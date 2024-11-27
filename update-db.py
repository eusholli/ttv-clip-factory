import json
import os
import pickle
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import Dict
from glob import glob

class TranscriptSearchSystem:
    def __init__(self, index_path='transcript_search.index', metadata_path='transcript_metadata.pkl'):
        print("Initializing search system...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = 384
        self.load_or_create_index()

    def load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata, self.processed_hashes = pickle.load(f)
                print(f"Loaded existing index with {len(self.metadata)} entries")
            except Exception as e:
                print(f"Error loading index: {str(e)}")
                self.create_new_index()
        else:
            self.create_new_index()

    def create_new_index(self):
        print("Creating new index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.processed_hashes = set()

    def save_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump((self.metadata, self.processed_hashes), f)
            print("Index saved successfully")
        except Exception as e:
            print(f"Error saving index: {str(e)}")

    def get_segment_hash(self, segment: Dict, main_metadata: Dict) -> str:
        hash_string = (
            f"{segment['text']}"
            f"{segment['metadata']['start_timestamp']}"
            f"{segment['metadata']['end_timestamp']}"
            f"{main_metadata.get('title', '')}"
            f"{main_metadata.get('date', '')}"
        )
        return hashlib.md5(hash_string.encode()).hexdigest()

    def process_transcript(self, json_data: Dict) -> None:
        try:
            transcript = json_data['transcript']
            main_metadata = json_data.get('metadata', {})
            
            new_embeddings = []
            new_metadata = []
            skipped = 0
            
            print(f"Processing transcript with {len(transcript)} segments...")
            
            for segment in transcript:
                segment_hash = self.get_segment_hash(segment, main_metadata)
                
                if segment_hash in self.processed_hashes:
                    skipped += 1
                    continue
                    
                text = segment['text']
                metadata = {
                    'title': main_metadata.get('title', ''),
                    'date': main_metadata.get('date', ''),
                    'youtube_id': main_metadata.get('youtube_id', ''),
                    'source': main_metadata.get('source', ''),
                    'speaker': segment['metadata']['speaker'],
                    'company': segment['metadata']['company'],
                    'start_time': segment['metadata']['start_timestamp'],
                    'end_time': segment['metadata']['end_timestamp'],
                    'subjects': segment['metadata'].get('subjects', []),
                    'download': segment['metadata']['download'],
                    'text': text,
                    'segment_hash': segment_hash
                }
                
                embedding = self.model.encode([text])[0]
                new_embeddings.append(embedding)
                new_metadata.append(metadata)
                self.processed_hashes.add(segment_hash)
            
            if new_embeddings:
                self.index.add(np.array(new_embeddings).astype('float32'))
                self.metadata.extend(new_metadata)
                self.save_index()
                print(f"Added {len(new_embeddings)} new transcript segments")
            
            if skipped > 0:
                print(f"Skipped {skipped} existing segments")
            
        except Exception as e:
            print(f"Error processing transcript: {str(e)}")

def main():
    search_system = TranscriptSearchSystem()
    
    # Get all JSON files in cache directory that start with 'cached_'
    cache_files = glob('cache/cached_*.json')
    
    if not cache_files:
        print("No cache files found matching pattern 'cached_*.json' in cache directory")
        return
    
    print(f"Found {len(cache_files)} cache files to process")
    
    for file_path in cache_files:
        print(f"\nProcessing {file_path}...")
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            search_system.process_transcript(json_data)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

if __name__ == "__main__":
    main()
