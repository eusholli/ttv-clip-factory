import re
import asyncio
import json
import os
import traceback
from pyppeteer import launch
from bs4 import BeautifulSoup, NavigableString
from video_utils import get_youtube_video, generate_clips
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
import logging

# Set the TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure logging to suppress MoviePy's console output
logging.getLogger("moviepy").setLevel(logging.WARNING)


CACHE_DIR = "cache/"
DB_METADATA_FILE = os.path.join(CACHE_DIR, "db_metadata.json")
SUBJECTS = [
    " 5G ", " Automation ", " AI ", " Innovation ", " Network ", " Enterprise ", " Open RAN ",
    " TechCo ", " B2B ", " API ", " Infrastructure ", " Connectivity "
]

os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class TranscriptSegment:
    metadata: Dict[str, Optional[str]]
    text: str


@dataclass
class VideoInfo:
    metadata: Dict[str, Optional[str]]
    transcript: List[TranscriptSegment]


async def get_client_rendered_content(url: str, max_retries: int = 3) -> str:
    browser = None
    for attempt in range(max_retries):
        try:
            browser = await launch(
                handleSIGINT=False,
                handleSIGTERM=False,
                handleSIGHUP=False,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            page = await browser.newPage()
            
            # Set a reasonable timeout for navigation
            response = await page.goto(url, {
                'waitUntil': 'networkidle0',
                'timeout': 30000  # 30 seconds timeout
            })
            
            if not response.ok:
                raise Exception(f"Failed to load page: {response.status} {response.statusText}")
                
            # Wait for content to load
            await asyncio.sleep(5)
            
            # Get the content
            content = await page.content()
            
            # Clean shutdown
            await page.close()
            await browser.close()
            
            return content
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {str(e)}")
            
            if browser:
                try:
                    await browser.close()
                except Exception as close_error:
                    logger.error(f"Error closing browser: {str(close_error)}")
                finally:
                    browser = None
            
            if attempt == max_retries - 1:
                raise Exception(f"Failed to fetch content after {max_retries} attempts: {str(e)}")
            
            # Wait before retrying
            await asyncio.sleep(2 * (attempt + 1))
            
    raise Exception("Failed to fetch content: Max retries exceeded")

def extract_text_with_br(element):
    result = ['<br><br>']
    for child in element.descendants:
        if isinstance(child, NavigableString):
            result.append(child.strip())
        elif child.name == 'br':
            result.append('<br>')
    return ''.join(result).strip()


def extract_info(html_content: str) -> VideoInfo:
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string.strip() if soup.title else None
        date_elem = soup.find('p', class_='content-date')
        date = date_elem.find('span', class_='ng-binding').text.strip() if date_elem else None
        youtube_iframe = soup.find('iframe', src=lambda x: x and 'youtube' in x)
        youtube_url = youtube_iframe['src'] if youtube_iframe else None
        youtube_id = re.search(r'youtube.*\.com/embed/([^?]+)', youtube_url).group(1) if youtube_url else None
        if get_youtube_video(CACHE_DIR, youtube_id):
            transcript_elem = soup.find(id='transcript0')
            transcript = extract_text_with_br(transcript_elem) if transcript_elem else None
            return VideoInfo(
                metadata={'title': title, 'date': date, 'youtube_id': youtube_id},
                transcript=parse_transcript(transcript) if transcript else []
            )
        else:
            return None
    except Exception as e:
        logger.error(f"Error extracting information: {str(e)}")
        raise


def read_file(filename: str) -> Optional[str]:
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        raise


def extract_subject_info(text: str) -> List[str]:
    return [subject for subject in SUBJECTS if subject.lower() in text.lower()]


def extract_speaker_info(segment: str) -> Optional[Dict[str, Optional[str]]]:
    pattern = r'<br><br>(?:(?P<speaker>[^,(]+?)(?:,\s*(?P<company>[^(]+?))?)?\s*\((?P<timestamp>\d{2}:\d{2}:\d{2}|\d{2}:\d{2})\):<br>'

    match = re.match(pattern, segment)
    return {key: value.strip() if value else None for key, value in match.groupdict().items()} if match else None


def parse_transcript(content: str) -> List[TranscriptSegment]:
    parsed_segments = []
    saved_info = None

    segments = [segment.strip() for segment in re.split(r'(<br><br>.*?\((?:\d{2}:)?\d{2}:\d{2}\):<br>)',
                                                        content) if segment.strip()]

    for i, segment in enumerate(segments):
        speaker_info = extract_speaker_info(segment)
        if speaker_info:
            if speaker_info['speaker']:
                if saved_info:
                    text = segments[i-1] if i > 0 else ""
                    parsed_segments.append(TranscriptSegment(
                        metadata={
                            'speaker': saved_info['speaker'],
                            'company': saved_info['company'],
                            'start_timestamp': saved_info['timestamp'],
                            'end_timestamp': speaker_info['timestamp'],
                            'subjects': extract_subject_info(text)
                        },
                        text=text
                    ))
                saved_info = speaker_info
                if not saved_info['company']:
                    saved_info['company'] = "Unknown"
            else:
                if saved_info:
                    text = segments[i-1] if i > 0 else ""
                    parsed_segments.append(TranscriptSegment(
                        metadata={
                            'speaker': saved_info['speaker'],
                            'company': saved_info['company'],
                            'start_timestamp': saved_info['timestamp'],
                            'end_timestamp': speaker_info['timestamp'],
                            'subjects': extract_subject_info(text)
                        },
                        text=text
                    ))
                    saved_info['timestamp'] = speaker_info['timestamp']
        elif saved_info:
            continue

    if saved_info:
        text = segments[-1]
        parsed_segments.append(TranscriptSegment(
            metadata={
                'speaker': saved_info['speaker'],
                'company': saved_info['company'],
                'start_timestamp': saved_info['timestamp'],
                'end_timestamp': "00:00:00",
                'subjects': extract_subject_info(text)
            },
            text=text
        ))

    return parsed_segments


def get_cached_filename(url: str) -> str:
    return f"{CACHE_DIR}cached_{url.replace('://', '_').replace('/', '_')}"


async def process_url(url: str) -> Optional[VideoInfo]:
    try:
        cached_filename = get_cached_filename(url)
        html_filename = f"{cached_filename}.html"
        json_filename = f"{cached_filename}.json"

        if os.path.exists(json_filename):
            logger.info(f"Using cached JSON for {url}")
            with open(json_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return VideoInfo(
                    metadata=data['metadata'],
                    transcript=[TranscriptSegment(**segment) for segment in data['transcript']]
                )

        if os.path.exists(html_filename):
            logger.info(f"Using cached HTML for {url}")
            content = read_file(html_filename)
        else:
            logger.info(f"Fetching content from web for {url}")
            try:
                content = await get_client_rendered_content(url)
                # Only save content if successfully retrieved
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Failed to fetch content for {url}: {str(e)}")
                return None

        info = extract_info(content)
        if info is None:
            logger.warning(f"No valid information extracted from {url}")
            return None

        if info.transcript:
            logger.info(f"Generating clips for {url}")
            info_dict = asdict(info)
            try:
                info_dict['transcript'] = generate_clips(CACHE_DIR, info_dict)
                info = VideoInfo(
                    metadata=info_dict['metadata'],
                    transcript=[TranscriptSegment(**segment) for segment in info_dict['transcript']]
                )

                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(asdict(info), f, ensure_ascii=False, indent=4)

                logger.info(f"Information extracted and saved to {json_filename}")
            except Exception as e:
                logger.error(f"Error generating clips for {url}: {str(e)}")
                return None
        else:
            logger.warning(f"No transcript found for {url}")

        return info

    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}\n{traceback.format_exc()}")
        return None

async def process_urls(urls: List[str]) -> List[Optional[VideoInfo]]:
    return await asyncio.gather(*[process_url(url) for url in urls])


def db_save_metadata_sets(processed_urls: Set[str], speakers: Set[str],
                          companies: Dict[str, Set[str]],
                          sentiments: Set[str], subjects: Set[str]):
    metadata = {
        'processed_urls': list(processed_urls),
        'speakers': list(speakers),
        'companies': {company: list(speakers) for company, speakers in companies.items()},
        'sentiments': list(sentiments),
        'subjects': list(subjects)
    }

    with open(DB_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def db_load_metadata_sets() -> tuple:
    if os.path.exists(DB_METADATA_FILE):
        with open(DB_METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        return (
            set(metadata.get('processed_urls', [])),
            set(metadata.get('speakers', [])),
            {company: set(speakers) for company, speakers in metadata.get('companies', {}).items()},
            set(metadata.get('sentiments', [])),
            set(metadata.get('subjects', SUBJECTS))
        )

    return set(), set(), {}, set(), set(SUBJECTS)


async def main():
    url_file = "dsp-urls.txt"  # Changed from dsp-urls-one.txt to dsp-urls.txt

    if not os.path.exists(url_file):
        logger.error(f"Error: {url_file} not found.")
        return

    processed_urls, speakers, companies, sentiments, subjects = db_load_metadata_sets()

    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    total_urls = len(urls)
    for i, url in enumerate(urls, 1):
        if url in processed_urls:  # Uncommented this check to skip already processed URLs
            logger.info(f"[{i}/{total_urls}] {url} already processed")
            continue

        logger.info(f"[{i}/{total_urls}] Processing {url}")
        info = await process_url(url)
        if info is None:
            logger.warning(f"[{i}/{total_urls}] Failed to process {url}")
            continue

        for entry in info.transcript:
            metadata = {**info.metadata, **entry.metadata}
            company = metadata.get('company')
            speaker = metadata.get('speaker')
            entry_subjects = metadata.get('subjects', [])

            if speaker:
                speakers.add(speaker)
            subjects.update(entry_subjects)

            if company and speaker:
                companies.setdefault(company, set()).add(speaker)

        processed_urls.add(url)
        logger.info(f"[{i}/{total_urls}] Added new url: {url}")

    db_save_metadata_sets(processed_urls, speakers, companies, sentiments, subjects)

    logger.info("Processing complete. Check logs for any errors.")


if __name__ == "__main__":
    asyncio.run(main())
