import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
import requests
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class R2Manager:
    def __init__(self):
        """Initialize R2 manager with credentials from environment variables or Streamlit secrets"""
        # Load environment variables
        load_dotenv()
        
        # Get credentials from env vars first, fall back to Streamlit secrets
        self.account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID') or st.secrets.get('CLOUDFLARE_ACCOUNT_ID')
        self.access_key_id = os.getenv('CLOUDFLARE_ACCESS_KEY_ID') or st.secrets.get('CLOUDFLARE_ACCESS_KEY_ID')
        self.secret_access_key = os.getenv('CLOUDFLARE_SECRET_ACCESS_KEY') or st.secrets.get('CLOUDFLARE_SECRET_ACCESS_KEY')
        self.bucket_name = os.getenv('CLOUDFLARE_BUCKET_NAME') or st.secrets.get('CLOUDFLARE_BUCKET_NAME')
        
        # Define storage limit (10GB in bytes)
        self.storage_limit = 10 * 1024 * 1024 * 1024
        
        # Validate required environment variables
        if not all([self.account_id, self.access_key_id, 
                   self.secret_access_key, self.bucket_name]):
            raise ValueError("Missing required credentials. Check environment variables or Streamlit secrets.")
        
        # Initialize R2 client
        self.s3_client = self._initialize_r2_client()

    def _initialize_r2_client(self):
        """Initialize the R2 client with credentials"""
        return boto3.client(
            service_name='s3',
            endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name='auto'  # R2 doesn't use regions, but boto3 requires this
        )

    def get_total_space_used(self) -> int:
        """
        Calculate total space used in the R2 bucket
        
        Returns:
            int: Total space used in bytes
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            total_size = 0
            
            for obj in response.get('Contents', []):
                total_size += obj['Size']
                
            return total_size
            
        except ClientError as e:
            logger.error(f"Error calculating total space: {str(e)}")
            return 0

    def file_exists(self, object_name: str) -> bool:
        """
        Check if a file already exists in the R2 bucket
        
        Args:
            object_name (str): The name of the object to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking file existence: {str(e)}")
            raise

    def _get_content_type(self, file_path: str) -> str:
        """Determine content type based on file extension"""
        extension = Path(file_path).suffix.lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.webm': 'video/webm'
        }
        return content_types.get(extension, 'application/octet-stream')

    def upload_file(self, file_path: str, object_name: str = None) -> bool:
        """
        Upload a file to Cloudflare R2 storage if it doesn't already exist and if under storage limit
        
        Args:
            file_path (str): Path to the file to upload
            object_name (str): S3 object name (if different from file_path)
            
        Returns:
            bool: True if file was uploaded successfully or already exists, False otherwise
        """
        # If object_name not specified, use file name
        if object_name is None:
            object_name = Path(file_path).name
            
        try:
            # Verify file exists and is readable
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # Check if file already exists in bucket
            if self.file_exists(object_name):
                logger.info(f"File {object_name} already exists in bucket {self.bucket_name}")
                return True

            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Check if adding this file would exceed storage limit
            total_used = self.get_total_space_used()
            if total_used + file_size > self.storage_limit:
                logger.error(f"Upload would exceed storage limit of {self.storage_limit/1024/1024/1024:.2f}GB. " +
                           f"Current usage: {total_used/1024/1024/1024:.2f}GB, " +
                           f"File size: {file_size/1024/1024/1024:.2f}GB")
                return False

            # Upload file with content type detection
            content_type = self._get_content_type(file_path)
            self.s3_client.upload_file(
                file_path, 
                self.bucket_name, 
                object_name,
                ExtraArgs={'ContentType': content_type}
            )
            
            logger.info(f"Successfully uploaded {file_path} to {self.bucket_name}/{object_name}")
            return True
            
        except ClientError as e:
            logger.error(f"Upload failed: {str(e)}")
            return False

    def list_videos(self):
        """
        List all videos in the bucket
        
        Returns:
            list: List of video filenames in the bucket
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            videos = []
            
            for obj in response.get('Contents', []):
                if obj['Key'].lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                    videos.append(obj['Key'])
                    
            return videos
            
        except ClientError as e:
            logger.error(f"Error listing videos: {str(e)}")
            return []

    def generate_presigned_url(self, object_name: str, expiration: int = 3600):
        """
        Generate a presigned URL for video access
        
        Args:
            object_name (str): Name of the object in the bucket
            expiration (int): URL expiration time in seconds (default: 1 hour)
            
        Returns:
            str: Presigned URL or None if error
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_name
                },
                ExpiresIn=expiration
            )
            return url
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            return None

    def get_video_content(self, url: str):
        """
        Download video content from presigned URL
        
        Args:
            url (str): Presigned URL to download from
            
        Returns:
            bytes: Video content or None if error
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"Error downloading video: {str(e)}")
            return None

    def get_video_url_and_content(self, object_name: str, expiration: int = 3600):
        """
        Get both presigned URL and video content for an object
        
        Args:
            object_name (str): Name of the object in the bucket
            expiration (int): URL expiration time in seconds (default: 1 hour)
            
        Returns:
            tuple: (presigned_url, video_content) or (None, None) if error
        """
        url = self.generate_presigned_url(object_name, expiration)
        if url:
            content = self.get_video_content(url)
            return url, content
        return None, None
