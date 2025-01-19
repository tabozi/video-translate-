import os
import logging
from datetime import datetime
import whisper
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip
import yt_dlp
import argparse
from deep_translator import GoogleTranslator
import subprocess

def setup_environment():
    """Configure folders and logging."""
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
    return logging.getLogger(__name__)

def download_youtube_video(url):
    """Download YouTube video."""
    video_id = url.split('watch?v=')[-1]
    output_path = os.path.join('downloads', f'{video_id}.%(ext)s')
    
    existing_files = [f for f in os.listdir('downloads') if f.startswith(video_id + '.')]
    if existing_files:
        logging.info(f"Video already downloaded: {existing_files[0]}")
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return os.path.join('downloads', existing_files[0]), info.get('title')
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
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

def transcribe_audio(audio_path):
    """Transcribe audio to text using Whisper."""
    logging.info("Starting audio transcription with Whisper")
    try:
        model = whisper.load_model("tiny")
        logging.info("Whisper model loaded")
        
        result = model.transcribe(
            audio_path,
            task="transcribe",
            verbose=True
        )
        
        text = result["text"]
        logging.info("Transcription successful")
        logging.info(f"Transcribed text (start): {text[:100]}...")
        
        return text
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return None

def combine_video_audio(video_path, audio_path, output_filename):
    """Combine video with new audio."""
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        if audio.duration > video.duration:
            audio = audio.subclip(0, video.duration)
        elif audio.duration < video.duration:
            repeats = int(video.duration / audio.duration) + 1
            audio = AudioFileClip(audio_path)
            audio = audio.loop(repeats)
            audio = audio.subclip(0, video.duration)
        
        temp_audio = "temp_audio.aac"
        audio.write_audiofile(temp_audio, codec='aac', bitrate='192k')
        
        output_filename = "".join(x for x in output_filename if x.isalnum() or x in (' ', '-', '_'))
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'
        
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
            '-progress', 'pipe:1',
            output_filename
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        duration = video.duration
        print("\nEncoding progress:")
        while True:
            line = process.stdout.readline()
            if not line:
                break
            
            if 'out_time_ms=' in line:
                time_ms = int(line.split('=')[1])
                current_time = time_ms / 1000000
                progress = min(100, (current_time / duration) * 100)
                
                bar_length = 50
                filled_length = int(bar_length * progress / 100)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                print(f'\r[{bar}] {progress:.1f}%', end='', flush=True)
        
        print()
        if process.wait() == 0:
            logging.info("Video successfully encoded")
        else:
            raise Exception("Video encoding failed")
        
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
    
    logger = setup_environment()
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
    
    print("Transcribing audio...")
    text = transcribe_audio(video_path)
    if text is None:
        return
    
    logger.info(f"Transcribed text: {text[:100]}...")
    
    print("Generating French audio...")
    translated_audio = text_to_speech(text)
    
    print("Creating final video...")
    output_path = combine_video_audio(video_path, translated_audio, output_filename)
    
    logger.info(f"Translated video saved as: {output_path}")
    print(f"Translated video saved as: {output_path}")
    
    # Clean temporary files unless --keep-temp is used
    if not args.keep_temp:
        for file in [video_path, translated_audio]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Temporary file removed: {file}")

if __name__ == "__main__":
    main() 