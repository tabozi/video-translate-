import os
import logging
import time
import torch
import torch.cuda
import torch.backends.cudnn
from datetime import datetime
import whisper
from googletrans import Translator
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip
import yt_dlp
from pydub import AudioSegment
import argparse
from deep_translator import GoogleTranslator
import subprocess

# Global PyTorch CUDA configuration
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def setup_environment():
    """Configure folders, logging and CUDA."""
    # Create necessary directories
    os.makedirs('downloads', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Logging configuration
    log_filename = f"logs/translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # CUDA configuration
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # CUDA optimizations
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
        logger.info("GPU not available, using CPU")
    
    return logger, device

def download_youtube_video(url):
    """Download YouTube video."""
    video_id = url.split('watch?v=')[-1]
    output_path = os.path.join('downloads', f'{video_id}.%(ext)s')
    
    existing_files = [f for f in os.listdir('downloads') if f.startswith(video_id + '.')]
    if existing_files:
        logging.info(f"Video already downloaded: {existing_files[0]}")
        # Get video info even if already downloaded
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return os.path.join('downloads', existing_files[0]), info.get('title')
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height>=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info(f"Starting video download: {url}")
            info = ydl.extract_info(url, download=True)
            ext = info['ext']
            title = info.get('title')
            final_path = os.path.join('downloads', f'{video_id}.{ext}')
            logging.info(f"Video successfully downloaded: {final_path}")
            return final_path, title
    except Exception as e:
        logging.error(f"Error during download: {str(e)}")
        return None, None

def extract_audio(video_path):
    """Extract audio from video."""
    audio_path = os.path.join('downloads', 'original_audio.wav')
    try:
        logging.info(f"Extracting audio from: {video_path}")
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
        video.close()
        logging.info(f"Audio successfully extracted: {audio_path}")
        return audio_path
    except Exception as e:
        logging.error(f"Error during audio extraction: {str(e)}")
        return None

def transcribe_audio(audio_path, device):
    """Transcribe audio to text using Whisper."""
    logging.info("Starting audio transcription with Whisper")
    try:
        if device == "cuda":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        model = whisper.load_model(
            "tiny",
            device=device,
            download_root=os.path.join('downloads', 'models')
        ).to(device)
        
        logging.info(f"Whisper model loaded on {device}")
        
        # First, transcribe in original language
        options = {
            "task": "transcribe",
            "beam_size": 3,
            "best_of": 3,
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": True,
            "verbose": True,
            "fp16": True if device == "cuda" else False
        }
        
        result = model.transcribe(audio_path, **options)
        text = result["text"]
        
        logging.info("Transcription successful")
        logging.info(f"Transcribed text (start): {text[:100]}...")
        
        return text

    except Exception as e:
        logging.error(f"Error during Whisper transcription: {str(e)}")
        print(f"Transcription error: {str(e)}")
        return None
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.set_default_tensor_type('torch.FloatTensor')

def translate_text(text):
    """Translate text to French handling long texts."""
    try:
        # Split text into chunks of 4900 characters (safety margin)
        chunk_size = 4900
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        translator = GoogleTranslator(source='auto', target='fr')
        translated_chunks = []
        
        logging.info(f"Translating text in {len(chunks)} parts")
        
        for i, chunk in enumerate(chunks, 1):
            logging.info(f"Translating part {i}/{len(chunks)}")
            translated_chunk = translator.translate(chunk)
            if translated_chunk:
                translated_chunks.append(translated_chunk)
            else:
                logging.error(f"Translation failed for part {i}")
                translated_chunks.append(chunk)
        
        # Combine translated chunks
        final_translation = ' '.join(translated_chunks)
        
        if final_translation:
            logging.info(f"Complete translation done. Start: {final_translation[:100]}...")
            return final_translation
        else:
            logging.error("Translation completely failed")
            return text
            
    except Exception as e:
        logging.error(f"Error during translation: {str(e)}")
        return text

def text_to_speech(text):
    """Convert text to speech."""
    try:
        # Add parameters to improve voice quality
        tts = gTTS(text=text, lang='fr', slow=False)
        output_path = "translated_audio.mp3"
        tts.save(output_path)
        logging.info("French audio generated successfully")
        return output_path
    except Exception as e:
        logging.error(f"Error during audio generation: {str(e)}")
        return None

def combine_video_audio(video_path, audio_path, output_filename):
    """Combine video with new audio using NVENC."""
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        # Adjust audio duration
        if audio.duration > video.duration:
            audio = audio.subclip(0, video.duration)
        elif audio.duration < video.duration:
            repeats = int(video.duration / audio.duration) + 1
            audio = AudioFileClip(audio_path)
            audio = audio.loop(repeats)
            audio = audio.subclip(0, video.duration)
        
        # Save audio temporarily
        temp_audio = "temp_audio.aac"
        audio.write_audiofile(temp_audio, codec='aac', bitrate='192k')
        
        # Clean filename and ensure it ends with .mp4
        output_filename = "".join(x for x in output_filename if x.isalnum() or x in (' ', '-', '_'))
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'
        
        # FFmpeg command for NVENC encoding
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', temp_audio,
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',
            '-tune', 'hq',
            '-b:v', '8M',
            '-maxrate', '10M',
            '-bufsize', '10M',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-map', '0:v:0',
            '-map', '1:a:0',
            output_filename
        ]
        
        logging.info("Starting NVENC encoding")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            logging.info("Video successfully encoded using GPU (NVENC)")
        else:
            logging.warning(f"NVENC failed, falling back to CPU: {process.stderr}")
            # Fallback to CPU
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', temp_audio,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-b:v', '8M',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_filename
            ]
            subprocess.run(cmd, check=True)
            logging.info("Video successfully encoded using CPU")
        
        # Cleanup
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        return output_filename
        
    except Exception as e:
        logging.error(f"Error during video/audio combination: {str(e)}")
        return None
    finally:
        if 'video' in locals():
            video.close()
        if 'audio' in locals():
            audio.close()

def main():
    # Argument parser configuration
    parser = argparse.ArgumentParser(description='Translate YouTube video to French.')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--keep-temp', '-k', action='store_true',
                      help='Keep temporary files')
    
    args = parser.parse_args()
    
    logger, device = setup_environment()
    logger.info("Starting video translation program")
    
    logger.info(f"URL received: {args.url}")
    
    print("Downloading video...")
    video_path, video_title = download_youtube_video(args.url)
    
    if video_path is None:
        logger.error("Unable to download video. Stopping program.")
        return
    
    # Create output filename from video title
    output_filename = f"{video_title}_fr" if video_title else "video_traduite"
    logger.info(f"Output filename will be: {output_filename}")
    
    print("Extracting audio...")
    audio_path = extract_audio(video_path)
    if audio_path is None:
        return
    
    print("Transcribing audio...")
    text = transcribe_audio(audio_path, device)
    if text is None:
        return
    
    logger.info(f"Transcribed text: {text[:100]}...")
    
    print("Translating text...")
    translated_text = translate_text(text)
    logger.info(f"Translated text: {translated_text[:100]}...")
    
    print("Generating French audio...")
    translated_audio = text_to_speech(translated_text)
    
    print("Creating final video...")
    output_path = combine_video_audio(video_path, translated_audio, output_filename)
    
    logger.info(f"Translated video saved as: {output_path}")
    print(f"Translated video saved as: {output_path}")
    
    # Clean temporary files unless --keep-temp is used
    if not args.keep_temp:
        for file in [audio_path, translated_audio]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Temporary file removed: {file}")

if __name__ == "__main__":
    main() 