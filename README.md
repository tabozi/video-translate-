# Video Translate

A Python script that translates YouTube videos into French.

## Features

- Downloads YouTube videos
- Extracts and transcribes audio using Whisper AI
- Translates text to French
- Generates French audio from translated text
- Combines video with new French audio
- GPU acceleration support (NVIDIA GPUs with NVENC)

## Requirements

- Python 3.10+
- FFmpeg
- NVIDIA GPU with NVENC support (optional, for hardware acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-translate.git
cd video-translate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# On Windows
# Download and install from https://ffmpeg.org/download.html
```

## Usage

```bash
python translate_video.py <youtube_url>
```

### Options

- `--keep-temp, -k`: Keep temporary files after processing

### Example

```bash
python translate_video.py https://www.youtube.com/watch?v=example
```

The translated video will be saved with the original title followed by "_fr" suffix.

## GPU Acceleration

The script automatically detects and uses NVIDIA GPU if available:
- Uses CUDA for Whisper AI transcription
- Uses NVENC for video encoding
- Falls back to CPU if GPU is not available or encounters errors

## Notes

- The script requires an internet connection for YouTube download and translation
- Video quality is preserved in the translation process
- Processing time depends on video length and available hardware
- GPU acceleration significantly improves processing speed 