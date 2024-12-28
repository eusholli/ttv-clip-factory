# Transcript Search System

This system provides advanced semantic search capabilities for video transcripts, incorporating comprehensive text analysis including sentiment, emotion, technical complexity, and more.

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- MPS-enabled device (for Apple Silicon) or CUDA-capable GPU (for NVIDIA)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install spaCy's large English model:
```bash
python -m spacy download en_core_web_lg
```

## PostgreSQL Setup

1. Install PostgreSQL and pgvector:

On macOS:
```bash
brew install postgresql
brew services start postgresql
```

On Ubuntu:
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

2. Install pgvector extension:

On macOS:
```bash
brew install pgvector
```

On Ubuntu:
```bash
sudo apt-get install postgresql-14-pgvector  # Replace 14 with your PostgreSQL version
```

3. Create the database and enable pgvector:
```bash
psql postgres
CREATE DATABASE transcript_search;
\c transcript_search
CREATE EXTENSION vector;
```

## Environment Configuration

Create a `.env` file in the project root:
```
DATABASE_URL=postgresql://localhost/transcript_search
OPENAI_API_KEY=your_openai_api_key  # If using OpenAI models
```

## Project Structure

```
.
├── update_pg.py          # Database update and text analysis
├── test-pg.py           # Search functionality
├── requirements.txt     # Python dependencies
└── .env                # Environment variables
```

## Usage

1. Process transcripts and update the database:
```bash
python update_pg.py
```

2. Search for transcripts:
```bash
python test-pg.py "your search query here"
```

Example:
```bash
python test-pg.py "What are the best quotes about technology?"
```

## Requirements

The following packages are required (included in requirements.txt):
```
playwright
beautifulsoup4
yt_dlp
pytube
moviepy
requests
streamlit
sentence-transformers
numpy
pandas
python-dotenv
boto3
botocore
stripe
backoff
psycopg2-binary
sqlalchemy
pgvector
langchain
langchain-community
anthropic
textblob
nltk
transformers
spacy
```

## Text Analysis Features

The system performs comprehensive analysis on both stored transcripts and search queries:

1. Basic Metadata
   - Session title, date, speaker, company
   - Duration calculation
   - Subject tagging

2. Sentiment and Emotion
   - Sentiment analysis (positive/negative/neutral)
   - Emotion detection (joy, sadness, anger, fear, surprise, neutral)

3. Entity Analysis
   - Subject extraction
   - Object identification
   - Location detection

4. Content Analysis
   - Topic classification with confidence scores
   - Key phrase extraction
   - Intent classification
   - Technical complexity assessment
   - Readability metrics (Flesch-Kincaid)
   - Quote classification
   - Stakeholder impact analysis
   - Time reference extraction
   - Claim detection

## Database Schema

The system uses PostgreSQL with the following schema:

```sql
CREATE TABLE transcripts (
    segment_hash TEXT PRIMARY KEY,
    title TEXT,
    date TIMESTAMP,
    youtube_id TEXT,
    source TEXT,
    speaker TEXT,
    company TEXT,
    start_time FLOAT,
    end_time FLOAT,
    subjects TEXT[],
    download JSONB,
    text TEXT,
    embedding vector(384)
);

CREATE INDEX embedding_idx ON transcripts USING ivfflat (embedding vector_cosine_ops);
```

## Troubleshooting

1. If you encounter CUDA/MPS errors:
   - Ensure you have the correct version of PyTorch installed for your system
   - For Apple Silicon: Use MPS backend
   - For NVIDIA: Install appropriate CUDA toolkit

2. Database connection issues:
   - Verify PostgreSQL is running: `pg_isready`
   - Check DATABASE_URL in .env
   - Ensure pgvector extension is installed

3. Memory issues:
   - The spaCy model and transformers can use significant memory
   - Consider using smaller models if needed
   - Batch process large datasets

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
