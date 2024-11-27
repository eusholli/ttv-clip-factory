# TelecomTV Clip Factory

The TelecomTV Clip Factory is a powerful web application that enables users to search, preview, purchase, and download video clips from TelecomTV content. The application provides an intuitive interface for searching through video transcripts and finding specific moments in TelecomTV's video content.

## Features

### Advanced Search Capabilities
- **Transcript Search**: Search through video transcripts to find specific content
- **Multiple Filters**: Refine your search using filters for:
  - Speaker
  - Date
  - Title
- **Adjustable Results**: Control the number of search results (1-20) displayed

### Rich Preview Features
- **Video Preview**: Watch the relevant clip directly in the interface
- **Transcript Display**: Read the exact transcript text with highlighted search terms
- **Metadata**: View detailed information including:
  - Speaker name and company
  - Precise timestamps
  - Publication date
  - Subject tags

### Clip Purchase and Download
- **Secure Payments**: Integrated Stripe payment system for clip purchases
- **Session Management**: Access to purchased clips throughout your browser session
- **Direct Downloads**: Immediate access to purchased clips in MP4 format

## Using the Application

1. **Search for Content**
   - Enter your search terms in the search box
   - Use the filter dropdowns to narrow results by speaker, date, or title
   - Adjust the number of results to display

2. **Preview Content**
   - Each result shows a video preview with the relevant segment
   - Read the transcript text with highlighted search terms
   - View associated metadata (speaker, company, timestamps, etc.)

3. **Purchase and Download Clips**
   - Click "Buy Now ($5)" on any clip you wish to purchase
   - Complete the secure payment process through Stripe
   - Download your purchased clip immediately
   - Access your purchased clips throughout your current browser session

## Important Notes

- Purchased clips are only accessible during your current browser session
- Make sure to download your clips before closing your browser
- The application automatically creates preview segments with proper timestamps
- Search results show a match percentage to help identify the most relevant content

## Technical Details

The application is built using:
- Streamlit for the web interface
- FAISS for efficient transcript search
- Sentence transformers for semantic search capabilities
- Stripe for payment processing
- R2 storage for video clip management

## Best Practices

1. **Searching**
   - Use specific search terms for better results
   - Combine filters to narrow down content
   - Check the match percentage to identify the most relevant clips

2. **Purchasing**
   - Download your clips immediately after purchase
   - Keep your browser session open until you've downloaded all purchased clips
   - Verify your clips after downloading

3. **Viewing**
   - Use the video preview to ensure the clip contains the content you need
   - Read the transcript to confirm the context
   - Check the timestamps to verify the clip length
