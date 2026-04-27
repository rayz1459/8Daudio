#!/usr/bin/env python3
"""
Spatial 8D Audio — Mac App
Captures any audio device (e.g. BlackHole) and applies real-time
HRTF-based 8D binaural spatialisation using ILD + ITD + reverb.
"""

import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal
import threading
import time
import math
import sys
import tkinter as tk
from tkinter import ttk, font as tkfont

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100
BLOCK_SIZE    = 1024
CHANNELS      = 2
HEAD_RADIUS   = 0.0875   # metres (~8.75 cm)
SPEED_SOUND   = 343.0    # m/s
MAX_ITD       = HEAD_RADIUS / SPEED_SOUND  # ~0.000255 s

# ─── STATE ───────────────────────────────────────────────────────────────────
class State:
    enabled      = True
    speed        = 0.035     # radians/sec
    radius       = 12.0      # metres
    elevation    = 15.0      # degrees
    room_size    = 0.72
    volume       = 1.0       # user volume multiplier (0.0 – 2.0)
    angle        = 0.0
    input_device = None
    output_device= None
    stream       = None
    running      = False

state = State()

# ─── REVERB IMPULSE RESPONSE (concert hall) ──────────────────────────────────
REVERB_LEN = 22050   # 500 ms — wide hall feel without excessive CPU

def make_impulse_response(length=REVERB_LEN, decay_s=0.7, predelay_ms=25):
    ir = np.zeros((length, 2), dtype=np.float32)
    predelay = int(predelay_ms * SAMPLE_RATE / 1000)
    decay_samps = max(1, int(decay_s * SAMPLE_RATE))
    # Distinct reflection taps per ear — wider stereo, suggests room walls
    taps_l = [(0.009, 0.60), (0.018, 0.50), (0.031, 0.42),
              (0.047, 0.34), (0.069, 0.26), (0.097, 0.18), (0.138, 0.12)]
    taps_r = [(0.012, 0.58), (0.021, 0.52), (0.035, 0.40),
              (0.052, 0.32), (0.073, 0.24), (0.102, 0.17), (0.143, 0.11)]
    for ch, taps in enumerate((taps_l, taps_r)):
        for t_s, g in taps:
            i = int(t_s * SAMPLE_RATE)
            if i < length:
                ir[i, ch] += g * (1.0 if (i // 7) % 2 == 0 else -1.0)
        noise = np.random.randn(length).astype(np.float32)
        env = np.exp(-np.arange(length, dtype=np.float32) / decay_samps)
        env[:predelay] = 0.0
        ir[:, ch] += noise * env * 0.28
    # Energy-normalise then scale down — keeps tail from accumulating and clipping
    ir /= (np.sqrt(np.sum(ir ** 2, axis=0)) + 1e-9)
    ir *= 0.40
    return ir

IMPULSE = make_impulse_response()

# Persistent overlap-add tail (REVERB_LEN-1 samples carry to next block)
reverb_tail = np.zeros((REVERB_LEN - 1, 2), dtype=np.float32)

# ─── FILTERS (SOS form — numerically stable for long-running streams) ─────────
# SOS avoids the pole-zero cancellation issues of direct b/a form, which
# accumulates floating-point error and introduces a subtle "quality drop" over
# long listening sessions.

# Sub-bass warmth (+4 dB shelf, 80 Hz)
sos_low    = scipy_signal.butter(2,   80 / (SAMPLE_RATE / 2), btype='low',  output='sos')
# (presence shelf removed — was adding 3 kHz harshness that caused "bad speaker" sound)
# Air absorption for distance (4 kHz low-pass)
sos_air    = scipy_signal.butter(2, 4000 / (SAMPLE_RATE / 2), btype='low',  output='sos')
# Elevation cue (7 kHz high-shelf)
sos_hi     = scipy_signal.butter(2, 7000 / (SAMPLE_RATE / 2), btype='high', output='sos')
# Head shadow — 2.5 kHz LP on far ear (raised from 1.8 kHz: less muffled,
# still convincing binaural cue, and the source won't "disappear" at the sides)
sos_shadow = scipy_signal.butter(2, 2500 / (SAMPLE_RATE / 2), btype='low',  output='sos')

def _zi(sos):
    return scipy_signal.sosfilt_zi(sos)

# One state per channel (or per use-site for shadow/crossfeed)
zi_low     = [_zi(sos_low)    * 0 for _ in range(2)]
zi_air     = [_zi(sos_air)    * 0 for _ in range(2)]
zi_hi      = [_zi(sos_hi)     * 0 for _ in range(2)]
zi_shadow  = [_zi(sos_shadow) * 0 for _ in range(2)]
zi_xshadow = [_zi(sos_shadow) * 0 for _ in range(2)]

# Persistent ITD delay lines (one per ear) — holds the tail from previous block
MAX_ITD_SAMPLES = int(MAX_ITD * SAMPLE_RATE) + 2
itd_buffer = np.zeros((MAX_ITD_SAMPLES, 2), dtype=np.float32)

# Crossfeed delay buffer — feeds each ear a delayed, head-shadowed copy of the
# OPPOSITE channel. This simulates how each ear hears both speakers in real
# listening, which is what convinces the brain the source is outside the head.
XFEED_DELAY_SAMPLES = int(0.00028 * SAMPLE_RATE)   # ~280 µs (typical head-width ITD)
xfeed_buffer = np.zeros((XFEED_DELAY_SAMPLES + 1, 2), dtype=np.float32)

# ─── ADAPTIVE GAIN CONTROL ───────────────────────────────────────────────────
_AGC_TARGET = 0.26   # target output RMS (~-12 dBFS — noticeably louder)
_AGC_MAX    = 12.0   # allow larger boost for quiet sources
_AGC_MIN    = 0.8    # never attenuate below this
_agc_gain   = [2.5]  # starting estimate; adapts within the first few blocks

# ─── 8D SPATIALISER ───────────────────────────────────────────────────────────
def spatialise(block: np.ndarray, angle: float, elevation: float, room_size: float) -> np.ndarray:
    """
    Apply HRTF-approximated 8D spatialisation to a stereo block.
    
    Uses:
      - ITD  (interaural time delay)   — delays the far ear
      - ILD  (interaural level diff)   — attenuates the far ear
      - Elevation EQ                   — boosts/cuts based on vertical angle
      - Convolution reverb             — concert hall ambience
    """
    global zi_low, zi_air, zi_hi, zi_shadow, zi_xshadow
    global itd_buffer, reverb_tail, xfeed_buffer

    if block.shape[1] == 1:
        block = np.repeat(block, 2, axis=1)

    in_stereo = block.astype(np.float32, copy=True)
    n         = in_stereo.shape[0]

    # ── SPEAKER-TEST APPROACH: stereo → mono point source orbiting head ─────
    # RMS-preserving mix (avoids the -3 dB loss of a plain 0.5 sum on typical
    # decorrelated stereo — a key reason the output was quiet vs bypass).
    INV_SQRT2 = 1.0 / math.sqrt(2)
    mono = (in_stereo[:, 0] + in_stereo[:, 1]) * INV_SQRT2

    # ── Subtle sub-bass warmth (80 Hz, +2 dB) ──────────────────────────────
    # Keep it gentle — AirPods Pro 2 already have good bass extension.
    lows, zi_low[0] = scipy_signal.sosfilt(sos_low, mono, zi=zi_low[0])
    mono = mono + lows * 0.20

    az_sin = math.sin(angle)
    az_cos = math.cos(angle)

    # ── ILD: ±10 dB swing ───────────────────────────────────────────────────
    ild_db     = 10.0 * az_sin
    left_gain  = 10 ** ((-ild_db / 2) / 20)
    right_gain = 10 ** (( ild_db / 2) / 20)
    left  = mono * left_gain
    right = mono * right_gain

    # ── Head shadow on the FAR ear ──────────────────────────────────────────
    # Run both filters every block so state stays warm regardless of which side
    # the source is on — prevents a HF transient when sides swap at 0° / 180°.
    filt_l, zi_shadow[0] = scipy_signal.sosfilt(sos_shadow, left,  zi=zi_shadow[0])
    filt_r, zi_shadow[1] = scipy_signal.sosfilt(sos_shadow, right, zi=zi_shadow[1])
    shadow_strength = abs(az_sin) * 0.80
    if az_sin > 0:   # source right → muffle left ear
        left  = left  * (1 - shadow_strength) + filt_l * shadow_strength
    else:            # source left → muffle right ear
        right = right * (1 - shadow_strength) + filt_r * shadow_strength

    out = np.column_stack([left, right])

    # ── ITD — continuous delay line (no per-block clicks) ───────────────────
    itd_samples = int(abs(az_sin) * MAX_ITD * SAMPLE_RATE)
    itd_samples = min(itd_samples, MAX_ITD_SAMPLES - 1)
    delayed = np.empty_like(out)
    for ch in range(2):
        full = np.concatenate((itd_buffer[:, ch], out[:, ch]))
        delay_ch = itd_samples if ((az_sin > 0 and ch == 0) or (az_sin < 0 and ch == 1)) else 0
        s_ix = MAX_ITD_SAMPLES - delay_ch
        delayed[:, ch] = full[s_ix:s_ix + n]
        itd_buffer[:, ch] = full[-MAX_ITD_SAMPLES:]
    out = delayed

    # ── Crossfeed (glues the binaural image) ────────────────────────────────
    cf_l_in, zi_xshadow[0] = scipy_signal.sosfilt(sos_shadow, out[:, 1], zi=zi_xshadow[0])
    cf_r_in, zi_xshadow[1] = scipy_signal.sosfilt(sos_shadow, out[:, 0], zi=zi_xshadow[1])
    full_l = np.concatenate((xfeed_buffer[:, 0], cf_l_in))
    full_r = np.concatenate((xfeed_buffer[:, 1], cf_r_in))
    cf_l   = full_l[1:1 + n]
    cf_r   = full_r[1:1 + n]
    xfeed_buffer[:, 0] = full_l[-(XFEED_DELAY_SAMPLES + 1):]
    xfeed_buffer[:, 1] = full_r[-(XFEED_DELAY_SAMPLES + 1):]
    out[:, 0] += cf_l * 0.22
    out[:, 1] += cf_r * 0.22

    # ── Stereo width bleed — keeps instrument separation (15 %) ─────────────
    out[:, 0] = out[:, 0] * 0.88 + in_stereo[:, 0] * 0.15
    out[:, 1] = out[:, 1] * 0.88 + in_stereo[:, 1] * 0.15

    # ── Rear colouration: ONE air-abs pass only (no double-filter) ───────────
    # Single pass keeps the source audible at the back instead of disappearing.
    absorption = max(0.0, (state.radius - 6.0) / 80.0)
    if az_cos < -0.15:   # behind the head
        absorption = max(absorption, -az_cos * 0.18)
    absorption = min(absorption, 0.40)
    if absorption > 0.001:
        for ch in range(2):
            filt, zi_air[ch] = scipy_signal.sosfilt(sos_air, out[:, ch], zi=zi_air[ch])
            out[:, ch] = out[:, ch] * (1 - absorption) + filt * absorption

    # ── Elevation: 7 kHz presence cue on the "up" arc ───────────────────────
    elv_factor = max(0.0, math.sin(angle * 0.43)) * (elevation / 45.0)
    if elv_factor > 0.01:
        for ch in range(2):
            highs, zi_hi[ch] = scipy_signal.sosfilt(sos_hi, out[:, ch], zi=zi_hi[ch])
            out[:, ch] = out[:, ch] + highs * (0.28 * elv_factor)

    # ── Concert-hall reverb — FFT convolution ───────────────────────────────
    if room_size > 0.05:
        wet = np.zeros_like(out)
        for ch in range(2):
            conv = scipy_signal.fftconvolve(out[:, ch], IMPULSE[:, ch], mode='full')
            conv[:REVERB_LEN - 1] += reverb_tail[:, ch]
            wet[:, ch] = conv[:n]
            reverb_tail[:, ch] = np.clip(conv[n:n + REVERB_LEN - 1], -2.0, 2.0)
        # Wet ratio scales aggressively with distance so that at concert range
        # (30-100 m) the source is almost entirely enveloped in reverb — like
        # being in row 40 of a venue. Formula: ~30 % wet at 6 m, ~85 % at 100 m.
        dist_wet  = min(0.85, 0.20 + state.radius / 55.0)
        wet_gain  = room_size * dist_wet
        dry_gain  = max(0.05, 1.0 - room_size * dist_wet * 0.85)
        out = out * dry_gain + wet * wet_gain
    else:
        reverb_tail *= 0.0

    # ── Adaptive RMS gain × user volume ─────────────────────────────────────
    # AGC is compensated for distance so far-back seats sound like a concert
    # (heavy reverb, air absorption) without going quiet. "Distance" controls
    # spatial feel; "Volume" controls loudness independently.
    # dist_comp undoes inverse-square so the perceived level stays consistent.
    dist_comp = 1.0 + max(0.0, state.radius - 6.0) / 50.0
    in_rms = max(1e-6, float(np.sqrt(np.mean(mono ** 2))))
    desired_gain = min(_AGC_MAX, (_AGC_TARGET * state.volume * dist_comp) / in_rms)
    desired_gain = max(_AGC_MIN * state.volume, desired_gain)
    alpha = 0.97 if desired_gain > _agc_gain[0] else 0.88
    _agc_gain[0] = alpha * _agc_gain[0] + (1.0 - alpha) * desired_gain
    out *= _agc_gain[0]

    # ── True peak limiter (fires only on actual peaks, no constant distortion)
    np.clip(out, -0.95, 0.95, out=out)
    return out


# ─── AUDIO CALLBACK ──────────────────────────────────────────────────────────
last_time = [time.perf_counter()]

def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        pass  # ignore overflow/underflow in real-time

    if not state.enabled:
        outdata[:] = indata
        return

    # Advance angle
    now   = time.perf_counter()
    dt    = now - last_time[0]
    last_time[0] = now
    state.angle += state.speed * dt

    processed = spatialise(
        indata.copy(),
        state.angle,
        state.elevation,
        state.room_size
    )

    if processed.shape[1] == 1:
        outdata[:, 0] = processed[:, 0]
        outdata[:, 1] = processed[:, 0]
    else:
        outdata[:] = processed[:frames]


# ─── STREAM CONTROL ──────────────────────────────────────────────────────────
def start_stream():
    if state.stream:
        state.stream.close()

    try:
        state.stream = sd.Stream(
            samplerate   = SAMPLE_RATE,
            blocksize    = BLOCK_SIZE,
            device       = (state.input_device, state.output_device),
            channels     = CHANNELS,
            dtype        = 'float32',
            callback     = audio_callback,
            latency      = 'low',
        )
        state.stream.start()
        state.running = True
        return True
    except Exception as e:
        state.running = False
        return str(e)

def stop_stream():
    if state.stream:
        state.stream.close()
        state.stream  = None
    state.running = False


# ─── GUI ─────────────────────────────────────────────────────────────────────
BG      = '#0a0a0a'
GREEN   = '#1ed760'
DARK    = '#161616'
BORDER  = '#1a1a1a'
MUTED   = '#555555'
WHITE   = '#ffffff'

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('8D Spatial Audio')
        self.configure(bg=BG)
        self.resizable(False, False)
        self.geometry('320x660')

        self._build_ui()
        self._refresh_devices()
        self._animate()

    def _build_ui(self):
        pad = dict(padx=16, pady=0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill='x', padx=16, pady=(20, 0))

        tk.Label(hdr, text='8D SPATIAL AUDIO', font=('Courier', 11, 'bold'),
                 bg=BG, fg=GREEN).pack(side='left')
        self.status_dot = tk.Label(hdr, text='◉', font=('Courier', 11),
                                   bg=BG, fg=MUTED)
        self.status_dot.pack(side='right')

        # ── Orbit canvas ──────────────────────────────────────────────────────
        self.canvas = tk.Canvas(self, width=120, height=120, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(pady=(16, 8))
        cx, cy, r1, r2 = 60, 60, 48, 30
        self.canvas.create_oval(cx-r1, cy-r1, cx+r1, cy+r1,
                                outline='#1a1a1a', width=1)
        self.canvas.create_oval(cx-r2, cy-r2, cx+r2, cy+r2,
                                outline='#222222', width=1, dash=(3,3))
        self.canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill=WHITE, outline='')
        self.orbit_dot  = self.canvas.create_oval(0,0,10,10, fill=GREEN, outline='')
        self.orbit_glow = self.canvas.create_oval(0,0,18,18,
                                                   fill='', outline=GREEN,
                                                   width=1)

        # ── Device selectors ──────────────────────────────────────────────────
        dev_frame = tk.Frame(self, bg=DARK, highlightbackground=BORDER,
                             highlightthickness=1)
        dev_frame.pack(fill='x', padx=16, pady=(0, 12))

        tk.Label(dev_frame, text='INPUT (BlackHole)', font=('Courier', 8),
                 bg=DARK, fg=MUTED).pack(anchor='w', padx=10, pady=(8,2))
        self.in_var = tk.StringVar()
        self.in_menu = ttk.Combobox(dev_frame, textvariable=self.in_var,
                                    state='readonly', font=('Courier', 9))
        self.in_menu.pack(fill='x', padx=10, pady=(0,6))

        tk.Label(dev_frame, text='OUTPUT (AirPods)', font=('Courier', 8),
                 bg=DARK, fg=MUTED).pack(anchor='w', padx=10, pady=(4,2))
        self.out_var = tk.StringVar()
        self.out_menu = ttk.Combobox(dev_frame, textvariable=self.out_var,
                                     state='readonly', font=('Courier', 9))
        self.out_menu.pack(fill='x', padx=10, pady=(0,10))

        # ── Toggle + Start ────────────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(fill='x', padx=16, pady=(0, 12))

        self.start_btn = tk.Button(btn_row, text='START', font=('Courier', 10, 'bold'),
                                   bg=GREEN, fg='black', relief='flat',
                                   padx=12, pady=6, cursor='hand2',
                                   command=self._toggle_stream)
        self.start_btn.pack(side='left', fill='x', expand=True, padx=(0,6))

        self.bypass_btn = tk.Button(btn_row, text='BYPASS OFF', font=('Courier', 9),
                                    bg=DARK, fg=MUTED, relief='flat',
                                    padx=12, pady=6, cursor='hand2',
                                    command=self._toggle_bypass)
        self.bypass_btn.pack(side='right')

        # ── Sliders ───────────────────────────────────────────────────────────
        sliders = tk.Frame(self, bg=BG)
        sliders.pack(fill='x', padx=16)

        self.vol_val = self._make_slider(
            sliders, 'VOLUME', 50, 200, 100,
            lambda v: setattr(state, 'volume', float(v) / 100.0),
            lambda v: f'{float(v):.0f}%'
        )
        self.speed_val = self._make_slider(
            sliders, 'ROTATION SPEED', 0, 30, 7,
            lambda v: setattr(state, 'speed', float(v) / 20),
            lambda v: ('STILL' if float(v) < 0.5 else f'{float(v)/7:.1f}x')
        )
        self.radius_val = self._make_slider(
            sliders, 'DISTANCE', 2, 100, 12,
            lambda v: setattr(state, 'radius', float(v)),
            lambda v: (
                f'{float(v):.0f}m  CLOSE'    if float(v) < 15  else
                f'{float(v):.0f}m  STAGE'    if float(v) < 35  else
                f'{float(v):.0f}m  HALL'     if float(v) < 65  else
                f'{float(v):.0f}m  STADIUM'
            )
        )
        self.room_val = self._make_slider(
            sliders, 'ROOM SIZE', 0, 100, 72,
            lambda v: setattr(state, 'room_size', float(v)/100),
            lambda v: f'{float(v):.0f}%'
        )
        self.elv_val = self._make_slider(
            sliders, 'ELEVATION', 0, 45, 15,
            lambda v: setattr(state, 'elevation', float(v)),
            lambda v: f'{float(v):.0f}°'
        )

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_lbl = tk.Label(self, text='Select devices and press START',
                                   font=('Courier', 8), bg=BG, fg=MUTED)
        self.status_lbl.pack(pady=(8, 16))

    def _make_slider(self, parent, label, mn, mx, default, on_change, fmt):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill='x', pady=4)

        head = tk.Frame(row, bg=BG)
        head.pack(fill='x')
        tk.Label(head, text=label, font=('Courier', 8),
                 bg=BG, fg=MUTED).pack(side='left')
        val_lbl = tk.Label(head, text=fmt(default), font=('Courier', 8, 'bold'),
                           bg=BG, fg=GREEN)
        val_lbl.pack(side='right')

        var = tk.DoubleVar(value=default)

        def cmd(v):
            on_change(v)
            val_lbl.config(text=fmt(v))

        sl = tk.Scale(row, from_=mn, to=mx, orient='horizontal',
                      variable=var, command=cmd,
                      bg=BG, fg=GREEN, troughcolor=DARK,
                      highlightthickness=0, bd=0,
                      sliderrelief='flat', showvalue=False, length=280)
        sl.pack(fill='x')
        return val_lbl

    def _refresh_devices(self):
        devs = sd.query_devices()
        inputs  = [f"{i}: {d['name']}" for i, d in enumerate(devs) if d['max_input_channels'] > 0]
        outputs = [f"{i}: {d['name']}" for i, d in enumerate(devs) if d['max_output_channels'] > 0]

        self.in_menu['values']  = inputs
        self.out_menu['values'] = outputs

        # Auto-select BlackHole for input, AirPods for output
        for s in inputs:
            if 'blackhole' in s.lower() or 'BlackHole' in s:
                self.in_var.set(s); break
        for s in outputs:
            if 'airpod' in s.lower():
                self.out_var.set(s); break

    def _toggle_stream(self):
        if state.running:
            stop_stream()
            self.start_btn.config(text='START', bg=GREEN, fg='black')
            self.status_dot.config(fg=MUTED)
            self.status_lbl.config(text='Stopped', fg=MUTED)
        else:
            # Parse selected device indices
            try:
                state.input_device  = int(self.in_var.get().split(':')[0])
                state.output_device = int(self.out_var.get().split(':')[0])
            except:
                self.status_lbl.config(text='Select input and output devices', fg='#ff4444')
                return

            result = start_stream()
            if result is True:
                self.start_btn.config(text='STOP', bg='#333', fg=GREEN)
                self.status_dot.config(fg=GREEN)
                self.status_lbl.config(text='◉  8D Spatial Audio Active', fg=GREEN)
            else:
                self.status_lbl.config(text=f'Error: {result}', fg='#ff4444')

    def _toggle_bypass(self):
        state.enabled = not state.enabled
        if state.enabled:
            self.bypass_btn.config(text='BYPASS OFF', fg=MUTED)
        else:
            self.bypass_btn.config(text='BYPASS ON', fg='#ff4444')

    def _animate(self):
        cx, cy, r = 60, 60, 46
        x = cx + math.sin(state.angle) * r
        y = cy + math.cos(state.angle) * r
        self.canvas.coords(self.orbit_dot,  x-5,  y-5,  x+5,  y+5)
        self.canvas.coords(self.orbit_glow, x-9,  y-9,  x+9,  y+9)
        self.after(16, self._animate)  # ~60fps


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('8D Spatial Audio')
    print('─' * 40)
    print('Available devices:')
    print(sd.query_devices())
    print('─' * 40)

    app = App()
    app.mainloop()
    stop_stream()
