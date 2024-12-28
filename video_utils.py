from moviepy import VideoFileClip
import os
import requests
from pytube import YouTube
import yt_dlp
from r2_manager import R2Manager


CLIP_DIR = "clip/"
if not os.path.exists(CLIP_DIR):
    os.makedirs(CLIP_DIR)


def get_youtube_video(cache_dir, yt_id):
    yt_url = f"https://www.youtube.com/watch?v={yt_id}"
    download_file = cache_dir + '/' + yt_id + ".mp4"

    if os.path.exists(download_file):
        print(f"{yt_url} already cached.")
        return download_file

    # try yt_dlp
    if try_yt_dlp_download(yt_url, download_file):
        return download_file

    # try pytube
    if try_pytube_download(yt_url, download_file):
        return download_file

    # Try Cobalt API
    if try_cobalt_api(yt_url, download_file):
        return download_file

    return None


def try_cobalt_api(yt_url, download_file):
    cobalt_api_url = "https://api.cobalt.tools/api/json"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "url": yt_url,
        "vCodec": "h264",
        "vQuality": "720",
        "aFormat": "mp3",
        "isAudioOnly": False
    }

    try:
        response = requests.post(cobalt_api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'success' and 'url' in data:
            video_url = data['url']
            video_response = requests.get(video_url)
            video_response.raise_for_status()

            with open(download_file, 'wb') as file:
                file.write(video_response.content)

            print(f"Video downloaded successfully using Cobalt API: "
                  f"{download_file}")
            return True
        else:
            print(f"Cobalt API Error: {data.get('text', 'Unknown error')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Cobalt API Error: Unable to process the YouTube URL. {str(e)}")
        return False


def try_pytube_download(yt_url, download_file):
    try:
        yt = YouTube(yt_url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by(
            'resolution').desc().first()
        video.download(filename=download_file)
        print(f"Video downloaded successfully using pytube: {download_file}")
        return True
    except Exception as e:
        print(f"Pytube Error: Unable to download the YouTube video. {str(e)}")
        return False


def try_yt_dlp_download(yt_url, download_file):
    ydl_opts = {
        'format': 'bestvideo[ext=h264]+bestaudio[ext=mp3]/best[ext=h264]/best',
        'outtmpl': download_file,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])
        print(f"Video downloaded successfully using yt-dlp: {download_file}")
        return True
    except Exception as e:
        print(f"yt-dlp Error: Unable to download the YouTube video. {str(e)}")
        return False


def generate_clips(cache_dir, info):
    yt_id = info['metadata']['youtube_id']
    download_file = get_youtube_video(cache_dir, yt_id)
    transcript = info['transcript']
    r2 = R2Manager()

    if download_file:
        video = VideoFileClip(download_file)

        for entry in transcript:
            start_time = entry['metadata']['start_timestamp']
            end_time = entry['metadata']['end_timestamp']

            # Adjust start and end times
            # Start 3 second earlier, but not before 0
            start_time = max(0, start_time - 3)
            end_time = min(video.duration, end_time +
                           3) if end_time != 0 else video.duration

            # Generate output filename
            output_filename = (
                f"{CLIP_DIR}{yt_id}-"
                f"{start_time}-{end_time}.mp4"
            )

            entry['metadata']['download'] = output_filename

            # Skip if already in R2
            if r2.file_exists(os.path.basename(output_filename)):
                continue

            # Skip if local file exists and already uploaded
            if os.path.exists(output_filename):
                # Try to upload existing file to R2
                if r2.upload_file(output_filename):
                    continue
            
            try:
                # Create clip using subclipped instead of subclip
                clip = video.subclipped(start_time, end_time)

                # Write the clip to a file
                clip.write_videofile(
                    output_filename, codec="libx264", audio_codec="aac")

                print(f"Generated clip: {output_filename}")

                # Upload to R2
                r2.upload_file(output_filename)
            except ValueError as e:
                if "start_time" in str(e) and "should be smaller than the clip's duration" in str(e):
                    print(f"Error: Start time {start_time} exceeds video duration {video.duration}. Skipping clip.")
                    entry['metadata']['download'] = None
                else:
                    raise e

        # Close the video to free up resources
        video.close()
    else:
        print(f"Failed to download video for YouTube ID: {yt_id}")
    
    return transcript

def main():
    youtube_id = "tCDvOQI3pco"
    print(f"Testing video download using YouTube ID: {youtube_id}...")
    video = get_youtube_video(youtube_id)
    if video:
        print(f"Downloaded video: {video}")
    else:
        print("Failed to download video.")


if __name__ == "__main__":
    main()
