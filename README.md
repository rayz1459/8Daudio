# 🎧 8D Spatial Audio — Mac App

Real-time 8D binaural audio processor for Spotify (or any app).
Uses HRTF simulation with ILD, ITD, reverb, and air absorption.

## How it works

```
Spotify → BlackHole (virtual cable) → 8D App → AirPods Pro 2
```

The app captures audio from BlackHole (a free virtual audio driver),
applies rotating HRTF binaural processing, and outputs to your AirPods.

## Setup (5 minutes)

### Step 1 — Install BlackHole (free)
Download **BlackHole 2ch** from https://existential.audio/blackhole/
Install it. No account needed, completely free.

### Step 2 — Route Spotify through BlackHole
**System Settings → Sound → Output → BlackHole 2ch**

Spotify will now send its audio to BlackHole instead of your speakers.

### Step 3 — Install Python dependencies
Open Terminal, run:
```bash
pip3 install sounddevice numpy scipy
```

### Step 4 — Launch the app
Double-click `launch.sh` or run in Terminal:
```bash
bash launch.sh
```

### Step 5 — Configure the app
- **Input**: select `BlackHole 2ch`
- **Output**: select `Ray's AirPods Pro` (or your headphones)
- Press **START**
- Play a song in Spotify

## Controls

| Control | Description |
|---|---|
| Rotation Speed | How fast the sound orbits (slower = more concert-like) |
| Distance | How far away the source feels (higher = further back in crowd) |
| Room Size | Amount of reverb / concert hall ambience |
| Elevation | Vertical movement of the sound source |
| Bypass | Toggle 8D effect on/off for A/B comparison |

## After listening — restore audio

When done, go to **System Settings → Sound → Output** and switch
back to your AirPods or MacBook speakers.

## The audio processing

- **ITD** (Interaural Time Delay): delays the far ear to simulate direction
- **ILD** (Interaural Level Difference): attenuates the far ear
- **Air absorption**: rolls off high frequencies at distance
- **Elevation EQ**: boosts frequencies for vertical position cues  
- **Convolution reverb**: synthetic concert hall impulse response
- **Distance attenuation**: volume drops with distance (inverse law)
