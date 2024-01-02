from faster_whisper import WhisperModel
import os, subprocess

YOUTUBE_ID = "aChceXG2txc" # Youtube ID
AUDIO_FILE_NAME = f"{YOUTUBE_ID}.mp3"
MODEL_SIZE = "large-v3"

# Download audio from Youtube
def dl_yt(yt_url):
    subprocess.run(f"yt-dlp -x --audio-format mp3 -o {AUDIO_FILE_NAME} {yt_url}", shell=True)

dl_yt(f"https://youtu.be/{YOUTUBE_ID}")


model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")

segments, info = model.transcribe(
	AUDIO_FILE_NAME,
	beam_size=5,
	vad_filter=True,
	without_timestamps=True,)
	
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    
os.remove(AUDIO_FILE_NAME)