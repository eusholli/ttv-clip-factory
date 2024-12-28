import json
import os
from glob import glob
import hashlib
import logging
from datetime import datetime
from hybrid_search import TranscriptSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptProcessor:
    def __init__(self):
        self.search = TranscriptSearch()

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert time string (MM:SS or HH:MM:SS) to float seconds."""
        try:
            parts = time_str.split(':')
            if len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            else:
                logger.warning(f"Invalid time format: {time_str}, using 0.0")
                return 0.0
        except (ValueError, AttributeError):
            logger.warning(f"Invalid time format: {time_str}, using 0.0")
            return 0.0

    def get_segment_hash(self, segment: dict, main_metadata: dict) -> str:
        hash_string = (
            f"{segment['text']}"
            f"{segment['metadata']['start_timestamp']}"
            f"{segment['metadata']['end_timestamp']}"
            f"{main_metadata.get('title', '')}"
            f"{main_metadata.get('date', '')}"
        )
        return hashlib.md5(hash_string.encode()).hexdigest()

    def process_transcript(self, json_data: dict) -> None:
        try:
            transcript = json_data['transcript']
            main_metadata = json_data.get('metadata', {})
            youtube_id = main_metadata.get('youtube_id')
            
            if youtube_id:
                logger.info(f"Deleting existing entries for YouTube ID: {youtube_id}")
                self.search.cursor.execute('DELETE FROM transcripts WHERE youtube_id = %s', (youtube_id,))
                self.search.conn.commit()
            
            new_count = 0
            skipped = 0
            
            logger.info(f"Processing transcript with {len(transcript)} segments...")
            
            # Parse date string to datetime object if exists
            date_str = main_metadata.get('date', '')
            date = None
            if date_str:
                try:
                    # Try different date formats
                    date_formats = ['%Y-%m-%d', '%b %d, %Y']
                    for fmt in date_formats:
                        try:
                            date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    if date is None:
                        logger.warning(f"Could not parse date: {date_str}")
                except Exception as e:
                    logger.warning(f"Error parsing date '{date_str}': {str(e)}")

            # Prepare batch data
            batch_data = []
            
            for segment in transcript:
                segment_hash = self.get_segment_hash(segment, main_metadata)
                
                batch_data.append({
                    'segment_hash': segment_hash,
                    'text': segment['text'],
                    'title': main_metadata.get('title', ''),
                    'date': date,
                    'youtube_id': main_metadata.get('youtube_id', ''),
                    'source': main_metadata.get('source', ''),
                    'speaker': segment['metadata']['speaker'],
                    'company': segment['metadata']['company'],
                    'start_time': self._time_to_seconds(segment['metadata']['start_timestamp']),
                    'end_time': self._time_to_seconds(segment['metadata']['end_timestamp']),
                    'duration': self._time_to_seconds(segment['metadata']['end_timestamp']) - self._time_to_seconds(segment['metadata']['start_timestamp']),
                    'subjects': segment['metadata']['subjects'],
                    'download': segment['metadata']['download']
                })
            
            try:
                self.search.add_transcripts_batch(batch_data)
                new_count = len(batch_data)
            except Exception as e:
                if "duplicate key value" in str(e):
                    # If we hit duplicates, rollback the failed transaction and fall back to individual inserts
                    self.search.conn.rollback()
                    new_count = 0
                    skipped = 0
                    for data in batch_data:
                        try:
                            self.search.add_transcript(**data)
                            new_count += 1
                        except Exception as e2:
                            if "duplicate key value" in str(e2):
                                skipped += 1
                                logger.info(f"Skipping duplicate segment: {data['segment_hash']}")
                            else:
                                # Rollback the current transaction before raising
                                self.search.conn.rollback()
                                raise e2
                else:
                    # Rollback the current transaction before raising
                    self.search.conn.rollback()
                    raise e
            
            logger.info(f"Added {new_count} new transcript segments")
            logger.info(f"Skipped {skipped} existing segments")
            
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}")
            raise

def process_directory(processor: TranscriptProcessor, directory: str) -> None:
    """Process all JSON files in the specified directory."""
    json_files = glob(os.path.join(directory, '*.json'))
    
    if not json_files:
        logger.warning(f"No JSON files found in directory: {directory}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    for file_path in json_files:
        logger.info(f"\nProcessing {file_path}...")
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            if not json_data.get('transcript'):
                logger.warning(f"No transcript found in {file_path}")
                continue
                
            processor.process_transcript(json_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

def main():
    import sys
    
    processor = TranscriptProcessor()
    
    # Check if argument is provided
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        # Check if it's a JSON file
        if os.path.isfile(path) and path.endswith('.json'):
            logger.info(f"Processing single JSON file: {path}")
            try:
                with open(path, 'r') as f:
                    json_data = json.load(f)
                
                if not json_data.get('transcript'):
                    logger.warning(f"No transcript found in {path}")
                    sys.exit(1)
                
                processor.process_transcript(json_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON file {path}: {str(e)}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error processing file {path}: {str(e)}")
                sys.exit(1)
        
        # Check if it's a directory
        elif os.path.isdir(path):
            logger.info(f"Processing JSON files from directory: {path}")
            process_directory(processor, path)
        
        else:
            logger.error(f"Invalid path: {path} - must be a JSON file or directory")
            sys.exit(1)
    else:
        logger.info("No path provided. Using test transcript data...")
        test_data = {
            "metadata": {
                "title": "How Boost Mobile is managing service assurance | TelecomTV",
                "date": "Nov 15, 2024",
                "youtube_id": "67aTlFZ8eq4"
            },
            "transcript": [
                {
                    "metadata": {
                        "speaker": "Guy Daniels",
                        "company": "TelecomTV",
                        "start_timestamp": "00:11",
                        "end_timestamp": "01:16",
                        "subjects": [" Network "],
                        "download": "clip/67aTlFZ8eq4-8-79.mp4"
                    },
                    "text": "Hello, you are watching Telecom TV and our special program on how Boost Mobile is leveraging Rakuten Symphony solutions to enhance network operations, service assurance, and customer experiences. I'm Guy Daniels, and in today's discussion we will look deeper into service assurance management through the partnership of Boost Mobile and Rakuten and discover the lessons learned along the way. Well, I'm delighted to say that joining me on the program today are Dawood Shahdad, who is VP Wireless Core Engineering at Boost Mobile, and Raul Atri, who is president of the OSS Business Unit at Rakuten Symphony. Good to see you both. Thanks so much for taking part in the program today. Can you provide an overview of the partnership between Rakuten and Boost Mobile and doward? Can I first ask you what were the primary goals and motivations behind this collaboration?"
                }
            ]
        }
        processor.process_transcript(test_data)

        # Verify the stored data
        logger.info("\nVerifying stored transcript data...")
        search = TranscriptSearch()
        results = search.hybrid_search(
            search_text="Boost Mobile Rakuten",
            filters={
                "youtube_id": "67aTlFZ8eq4",
                "speaker": "Guy Daniels"
            }
        )

        # Verify we got exactly one result
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        result = results[0]

        # Print the retrieved data
        logger.info("\nRetrieved transcript data:")
        logger.info(f"Title: {result['title']}")
        logger.info(f"Speaker: {result['speaker']}")
        logger.info(f"Company: {result['company']}")
        logger.info(f"YouTube ID: {result['youtube_id']}")
        logger.info(f"Start time: {result['start_time']}")
        logger.info(f"End time: {result['end_time']}")
        logger.info(f"Duration: {result['duration']}")
        logger.info(f"Download: {result['download']}")
        logger.info(f"Text: {result['text'][:100]}...")  # First 100 chars for brevity

        # Verify all fields match the original data
        assert result['title'] == test_data['metadata']['title'], "Title mismatch"
        assert result['youtube_id'] == test_data['metadata']['youtube_id'], "YouTube ID mismatch"
        assert result['speaker'] == test_data['transcript'][0]['metadata']['speaker'], "Speaker mismatch"
        assert result['company'] == test_data['transcript'][0]['metadata']['company'], "Company mismatch"
        assert result['download'] == test_data['transcript'][0]['metadata']['download'], "Download path mismatch"
        assert result['text'] == test_data['transcript'][0]['text'], "Transcript text mismatch"
        
        # Verify time calculations
        assert result['start_time'] == 11.0, "Start time calculation incorrect"  # 00:11 = 11 seconds
        assert result['end_time'] == 76.0, "End time calculation incorrect"  # 01:16 = 76 seconds
        assert result['duration'] == 65.0, "Duration calculation incorrect"  # 76 - 11 = 65 seconds

        logger.info("\nAll assertions passed - data was stored and retrieved correctly!")

if __name__ == "__main__":
    main()
