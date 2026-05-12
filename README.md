# Bleep It (forked and reworked from Bleep That Sh\*t!)

Transcribe and censor words in audio/video files using Whisper ONNX models.
All processing is local — nothing is sent to a server.

## Prerequisites

- Node.js 18+
- [FFmpeg](https://ffmpeg.org/) installed on your system (for video processing)

## Installation

```bash
git clone <repo>
cd bleep-it
npm install
```

## Usage

```bash
# Basic usage — censor swear words in a video
npx tsx src/cli.ts --input video.mp4

# Custom word list
npx tsx src/cli.ts --input audio.mp3 --words "fuck,shit,damn"

# Specify output file
npx tsx src/cli.ts --input video.mp4 --output clean.mp4

# Transcribe only (no censoring)
npx tsx src/cli.ts --input audio.mp3 --transcribe-only

# Fuzzy matching
npx tsx src/cli.ts --input audio.mp3 --words "heck" --match-mode fuzzy --fuzzy-distance 2

# Different bleep sound
npx tsx src/cli.ts --input audio.mp3 --bleep-sound dolphin

# Full options
npx tsx src/cli.ts --input video.mp4 \
  --words "fuck,shit,bitch" \
  --bleep-sound brown \
  --bleep-volume 70 \
  --buffer 0.1 \
  --original-volume 0.0 \
  --model Xenova/whisper-small.en \
  --match-mode exact
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input audio/video file | (required) |
| `-o, --output` | Output file path | input-censored.ext |
| `--words` | Comma-separated words to censor | fuck,shit,damn,bitch,... |
| `--bleep-sound` | Bleep sound (bleep/brown/dolphin/trex/silence) | bleep |
| `--bleep-volume` | Bleep volume 0-100 | 80 |
| `--buffer` | Extra seconds around each bleep | 0 |
| `--original-volume` | Original audio volume during bleeps 0.0-1.0 | 0.0 |
| `--model` | Whisper model ID | Xenova/whisper-small.en |
| `--language` | Language code | en |
| `--match-mode` | Match mode (exact/partial/fuzzy) | exact |
| `--fuzzy-distance` | Max Levenshtein distance for fuzzy | 1 |
| `--transcribe-only` | Only transcribe, don't censor | false |
| `-h, --help` | Show help | |

## Supported Formats

- **Audio**: mp3, wav, ogg, flac, m4a, aac
- **Video**: mp4, mkv, avi, mov, webm, m4v

## License

Apache License 2.0
