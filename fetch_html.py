import asyncio
import logging
import os
from typing import Optional
from pathlib import Path
from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeoutError
import backoff

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = "cache/"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_filename(url: str) -> str:
    """Generate a filesystem-safe filename for caching that preserves path structure."""
    # Replace http:// or https:// with cached_http_ or cached_https_
    filename = f"cached_{url.replace('://', '_')}"
    
    # Split the path components
    parts = filename.split('/')
    
    # Get the domain and rest of the path
    domain_parts = parts[0].split('_', 2)  # Split into ['cached', 'http(s)', 'domain.com']
    path_parts = parts[1:] if len(parts) > 1 else []
    
    # Reconstruct the path preserving internal structure
    if path_parts:
        path = '_'.join(path_parts)
        filename = f"{domain_parts[0]}_{domain_parts[1]}_{domain_parts[2]}_{path}"
    else:
        filename = '_'.join(domain_parts)
    
    return Path(CACHE_DIR) / f"{filename}.html"

@backoff.on_exception(
    backoff.expo,
    (PlaywrightTimeoutError, Exception),
    max_tries=MAX_RETRIES
)
async def fetch_url(page: Page, url: str) -> Optional[str]:
    """
    Fetch a URL with retry logic and proper error handling using Playwright.
    
    Args:
        page: Playwright page instance
        url: URL to fetch
        
    Returns:
        Optional[str]: HTML content if successful, None if failed
    """
    try:
        # Navigate to the page and wait for network idle
        await page.goto(url, wait_until='networkidle')
        
        # Wait for potential dynamic content
        await page.wait_for_timeout(2000)  # Additional 2s wait for dynamic content
        
        # Get the full HTML after JavaScript execution
        content = await page.content()
        return content
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        raise

async def fetch_and_cache(url: str, page: Page) -> Optional[str]:
    """
    Fetch URL content and cache it to disk using Playwright.
    
    Args:
        url: URL to fetch
        page: Playwright page instance
        
    Returns:
        Optional[str]: HTML content if successful, None if failed
    """
    cache_file = get_cached_filename(url)
    
    # Check cache first
    if cache_file.exists():
        logger.info(f"Using cached version of {url}")
        return cache_file.read_text(encoding='utf-8')
    
    # Fetch if not cached
    logger.info(f"Fetching {url}")
    content = await fetch_url(page, url)
    
    if content:
        # Cache the content
        try:
            cache_file.write_text(content, encoding='utf-8')
            logger.info(f"Cached content for {url}")
            return content
        except Exception as e:
            logger.error(f"Failed to cache content for {url}: {str(e)}")
            return content
    return None

async def main():
    """Main entry point for the script."""
    url_file = "dsp-urls-one.txt"
    
    try:
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"URL file {url_file} not found")
        return
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            for url in urls:
                logger.info(f"Processing {url}")
                content = await fetch_and_cache(url, page)
                
                if content:
                    logger.info(f"Successfully fetched and cached {url}")
                else:
                    logger.error(f"Failed to fetch {url}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
