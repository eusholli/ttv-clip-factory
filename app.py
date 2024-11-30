import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import hashlib
import os
import pickle
from r2_manager import R2Manager
from dotenv import load_dotenv
import stripe
import base64
import urllib.parse

# Load environment variables
load_dotenv()

# Initialize Stripe with API key from env vars or Streamlit secrets
stripe.api_key = os.getenv('STRIPE_SECRET_KEY') or st.secrets.get('STRIPE_SECRET_KEY', 'your_test_key_here')

# Get base URL from env vars or Streamlit secrets
BASE_URL = os.getenv('BASE_URL') or st.secrets.get('BASE_URL', 'http://localhost:8501')

class TranscriptSearchSystem:
    def __init__(self, index_path='transcript_search.index', metadata_path='transcript_metadata.pkl'):
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
            except Exception as e:
                st.error(f"Error loading index: {str(e)}")
                self.create_new_index()
        else:
            self.create_new_index()

    def create_new_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.processed_hashes = set()


    def search(self, query: str, top_k: int = 5, selected_speaker: List[str] = None, 
              selected_date: List[str] = None, selected_title: List[str] = None,
              selected_company: List[str] = None) -> List[Dict]:
        if not self.metadata:
            return []
        
        # Get initial embeddings for all entries that match the filters
        filtered_indices = []
        filtered_metadata = []
        
        for idx, meta in enumerate(self.metadata):
            # Apply filters
            if selected_speaker and meta['speaker'] not in selected_speaker:
                continue
            if selected_date and meta['date'] not in selected_date:
                continue
            if selected_title and meta['title'] not in selected_title:
                continue
            if selected_company and meta['company'] not in selected_company:
                continue
            
            filtered_indices.append(idx)
            filtered_metadata.append(meta)
        
        if not filtered_indices:
            return []
        
        # Create a temporary index with only the filtered entries
        temp_index = faiss.IndexFlatL2(self.dimension)
        temp_vectors = [self.index.reconstruct(i) for i in filtered_indices]
        temp_index.add(np.array(temp_vectors).astype('float32'))
        
        # Perform search on filtered index
        query_vector = self.model.encode([query])[0]
        distances, indices = temp_index.search(
            np.array([query_vector]).astype('float32'), 
            min(top_k, len(filtered_metadata))
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                result = filtered_metadata[idx].copy()
                result['score'] = 1 / (1 + distances[0][i])
                results.append(result)
                
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def get_metadata_by_hash(self, segment_hash: str) -> Dict:
        for meta in self.metadata:
            if meta['segment_hash'] == segment_hash:
                return meta
        return None

def format_timestamp(timestamp: str) -> str:
    time_parts = timestamp.split(':')
    if len(time_parts) == 2:
        return f"{time_parts[0]}m {time_parts[1]}s"
    return timestamp

def highlight_text(text: str, query: str) -> str:
    if not query:
        return text
    pattern = re.compile(f'({re.escape(query)})', re.IGNORECASE)
    return pattern.sub(r'**\1**', text)

def timestamp_to_seconds(timestamp):
    parts = timestamp.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        ts = h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(int, parts)
        ts = m * 60 + s
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    return ts

def create_checkout_session(clip_hash, search_state):
    try:
        # Add paid_clips to search state
        search_state['paid_clips'] = list(st.session_state.paid_clips)
        
        # Encode the search state
        encoded_state = base64.b64encode(json.dumps(search_state).encode()).decode()
        
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'Video Clip Download',
                        'description': 'Access to download the selected video clip',
                    },
                    'unit_amount': 500,  # $5.00 in cents
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{BASE_URL}?page=download&clip={clip_hash}&state={encoded_state}",
            cancel_url=f"{BASE_URL}?state={encoded_state}",
        )
        return checkout_session
    except Exception as e:
        st.error(f"Error creating checkout session: {str(e)}")
        return None

def show_download_page(clip_hash, search_state):
    st.title("⚠️ Important: Download Your Clip")
    st.warning("**Please Note:** Your purchase is only valid for this browser session. Make sure to download your clip before closing the browser.")
    
    # Get clip metadata
    clip_meta = st.session_state.search_system.get_metadata_by_hash(clip_hash)
    if not clip_meta:
        st.error("Clip not found!")
        return
    
    # Display clip information
    st.markdown(f"### {clip_meta['title']}")
    st.markdown(f"{clip_meta['speaker']} · {clip_meta['company']}")
    st.markdown(f"{format_timestamp(clip_meta['start_time'])} - {format_timestamp(clip_meta['end_time'])} · {clip_meta['date']}")
    st.markdown(clip_meta["text"])
    
    # Display YouTube preview
    st_ts = timestamp_to_seconds(clip_meta['start_time'])
    end_ts = timestamp_to_seconds(clip_meta['end_time'])
    st_ts = "0" if st_ts == 0 else st_ts-1
    end_url = "" if end_ts == 0 else f"&end={end_ts+1}"
    
    yt_url = (
        f"https://youtube.com/embed/{clip_meta['youtube_id']}"
        f"?start={st_ts}&{end_url}&autoplay=1&rel=0"
    )
    st.components.v1.iframe(src=yt_url, width=300, height=169, scrolling=True)
    
    # Handle download
    if 'download' in clip_meta:
        clip_filename = os.path.basename(clip_meta['download'])
        url, content = st.session_state.r2_manager.get_video_url_and_content(clip_filename)
        
        if content:
            st.download_button(
                label="Download Your Clip Now",
                data=content,
                file_name=clip_filename,
                mime="video/mp4",
                key=f"download_{clip_hash}"
            )
        else:
            st.error("Clip file not found in storage.")
    
    st.markdown("---")
    st.info("After downloading, you can always access this clip again from the main page during your current browser session.")
    
    # Return to main page button
    if st.button("Return to Main Page"):
        # Restore search state
        for key, value in search_state.items():
            if key == 'paid_clips':
                # Convert list back to set and update session state
                st.session_state.paid_clips.update(value)
            else:
                st.session_state[key] = value
        st.query_params.clear()
        st.rerun()

def main():
    st.set_page_config(
        page_title="TelecomTV Clip Factory",
        layout="wide"
    )
    
    # Initialize session state
    if 'search_system' not in st.session_state:
        st.session_state.search_system = TranscriptSearchSystem()
    
    if 'r2_manager' not in st.session_state:
        st.session_state.r2_manager = R2Manager()

    if 'paid_clips' not in st.session_state:
        st.session_state.paid_clips = set()

    # Check if we're on the download page after payment
    if "page" in st.query_params and st.query_params["page"] == "download":
        if "clip" in st.query_params and "state" in st.query_params:
            clip_hash = st.query_params["clip"]
            search_state = json.loads(base64.b64decode(st.query_params["state"]))
            
            # Restore paid_clips from search state and add new clip
            if 'paid_clips' in search_state:
                st.session_state.paid_clips.update(search_state['paid_clips'])
            st.session_state.paid_clips.add(clip_hash)
            
            show_download_page(clip_hash, search_state)
            return

    # Main page
    st.title("TelecomTV Clip Factory")

    # Display total number of entries
    total_entries = len(st.session_state.search_system.metadata)
    st.markdown(f"**Total Entries in Index:** {total_entries}")

    # Search interface
    search_query = st.text_input("Search transcripts", 
                                value=st.session_state.get('search_query', ''),
                                help="Enter your search query")
    
    # Store search query in session state
    st.session_state.search_query = search_query
    
    # Initialize filter variables with session state values
    if st.session_state.search_system.metadata:
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
        
        with col1:
            speakers = sorted(list(set(m['speaker'] for m in st.session_state.search_system.metadata)))
            selected_speaker = st.multiselect("Speaker", speakers, 
                                            default=st.session_state.get('selected_speaker', []))
            st.session_state.selected_speaker = selected_speaker
        
        with col2:
            # Convert dates to datetime objects for proper sorting
            dates = list(set(m['date'] for m in st.session_state.search_system.metadata))
            dates.sort(key=lambda x: datetime.strptime(x, "%b %d, %Y"), reverse=True)
            selected_date = st.multiselect("Date", dates,
                                         default=st.session_state.get('selected_date', []))
            st.session_state.selected_date = selected_date
            
        with col3:
            titles = sorted(list(set(m['title'] for m in st.session_state.search_system.metadata)))
            selected_title = st.multiselect("Title", titles,
                                          default=st.session_state.get('selected_title', []))
            st.session_state.selected_title = selected_title

        with col4:
            companies = sorted(list(set(m['company'] for m in st.session_state.search_system.metadata)))
            selected_company = st.multiselect("Company", companies,
                                            default=st.session_state.get('selected_company', []))
            st.session_state.selected_company = selected_company
            
        with col5:
            num_results = st.number_input(
                "Results",
                min_value=1,
                max_value=20,
                value=st.session_state.get('num_results', 5)
            )
            st.session_state.num_results = num_results
    
    # Show results if we have a search query
    if search_query or selected_speaker or selected_date or selected_title or selected_company:
        with st.spinner("Searching..."):
            results = st.session_state.search_system.search(
                search_query, 
                num_results,
                selected_speaker,
                selected_date,
                selected_title,
                selected_company
            )
            
            # Store search state
            search_state = {
                'search_query': search_query,
                'selected_speaker': selected_speaker,
                'selected_date': selected_date,
                'selected_title': selected_title,
                'selected_company': selected_company,
                'num_results': num_results,
                'paid_clips': list(st.session_state.paid_clips)  # Include paid_clips in search state
            }
            
            if results:
                st.markdown(f"Found {len(results)} results")
                
                for result in results:
                    st.markdown("---")                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        match_score = int(result['score'] * 100)
                        st.markdown(f"### {result['title']} (Match: {match_score}%)")
                        st.markdown(f"{result['speaker']} · {result['company']}")
                        st.markdown(f"{format_timestamp(result['start_time'])} - {format_timestamp(result['end_time'])} · {result['date']}")
                        st.markdown(highlight_text(result["text"], search_query))
                        if result['subjects']:
                            st.markdown(f"Tags: {', '.join(result['subjects'])}")
                    
                    with col2:
                        # Display YouTube preview with sandbox attributes
                        st_ts = timestamp_to_seconds(result['start_time'])
                        end_ts = timestamp_to_seconds(result['end_time'])
                        st_ts = "0" if st_ts == 0 else st_ts-1
                        end_url = "" if end_ts == 0 else f"&end={end_ts+1}"

                        yt_url = (
                        f"https://youtube.com/embed/{result['youtube_id']}"
                        f"?start={st_ts}&{end_url}&autoplay=1&rel=0"
                        )
                        st.components.v1.iframe(src=yt_url, width=300, height=169, scrolling=True)

                        # Handle video download with payment flow
                        if 'download' in result:
                            segment_hash = result['segment_hash']
                            if segment_hash in st.session_state.paid_clips:
                                clip_filename = os.path.basename(result['download'])
                                url, content = st.session_state.r2_manager.get_video_url_and_content(clip_filename)
                                
                                if content:
                                    st.download_button(
                                        label="Download Purchased Clip",
                                        data=content,
                                        file_name=clip_filename,
                                        mime="video/mp4",
                                        key=f"download_{segment_hash}"
                                    )
                                else:
                                    st.error("Clip file not found in storage.")
                            else:
                                if st.button("Buy Now ($5)", key=f"buy_{segment_hash}"):
                                    checkout_session = create_checkout_session(segment_hash, search_state)
                                    if checkout_session:
                                        st.link_button("Proceed to Payment", checkout_session.url)
            else:
                st.info("No results found. Try adjusting your search query or filters.")

if __name__ == "__main__":
    main()
