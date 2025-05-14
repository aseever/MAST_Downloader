#!/usr/bin/env python3
"""
mast_csv_downloader.py - Download FITS files from MAST CSV search results

This script handles downloading FITS files from a CSV file of MAST search results,
with support for retries, progress tracking, and organizing files by observation ID.

Example usage:
    python mast_csv_downloader.py -i search_results.csv -o download_directory -p proposal_id

"""

import os
import sys
import csv
import time
import argparse
import requests
import logging
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mast_download.log")
    ]
)
logger = logging.getLogger('mast_downloader')

# Constants
MAST_BASE_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_CHUNK_SIZE = 8192  # bytes

# Optional: Add your MAST API token here if you have one
MAST_API_TOKEN = None


class DownloadStats:
    """Class to track download statistics"""
    
    def __init__(self):
        self.total = 0
        self.downloaded = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = datetime.now()
        self.completed_urls = []
        self.failed_urls = []
        
    def update(self, url, status):
        """Update statistics based on download status"""
        if status:
            self.downloaded += 1
            self.completed_urls.append(url)
        else:
            self.failed += 1
            self.failed_urls.append(url)
    
    def elapsed_time(self):
        """Calculate elapsed time"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def summary(self):
        """Generate a summary of the download session"""
        elapsed = self.elapsed_time()
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        
        return (
            f"\n=== Download Summary ===\n"
            f"Total files: {self.total}\n"
            f"Downloaded: {self.downloaded}\n"
            f"Failed: {self.failed}\n"
            f"Skipped: {self.skipped}\n"
            f"Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"Download rate: {self.downloaded / elapsed:.2f} files/second\n"
        )


def process_download_url(url):
    """
    Process a MAST download URL to ensure it's in the correct format
    
    Args:
        url (str): URL or MAST URI
    
    Returns:
        str: Processed URL ready for downloading
    """
    if not url:
        raise ValueError("Empty URL provided")
    
    if url.startswith('http'):
        # URL is already in full form
        return url
    elif url.startswith('mast:'):
        # Convert MAST URI to download URL
        return f"{MAST_BASE_URL}{url}"
    else:
        # Assume it's a relative URI
        return f"{MAST_BASE_URL}mast:{url}"


def download_file(url, output_path, headers=None, chunk_size=DEFAULT_CHUNK_SIZE, timeout=DEFAULT_TIMEOUT, retries=MAX_RETRIES):
    """
    Download a file with progress bar and retry capability
    
    Args:
        url (str): URL to download
        output_path (str): Output file path
        headers (dict, optional): HTTP headers for the request
        chunk_size (int, optional): Size of chunks to download
        timeout (int, optional): Request timeout in seconds
        retries (int, optional): Number of retry attempts
    
    Returns:
        bool: True if download successful, False otherwise
    """
    # Process URL
    try:
        download_url = process_download_url(url)
    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        return False
    
    # Set up headers
    if headers is None:
        headers = {}
    
    if MAST_API_TOKEN:
        headers['Authorization'] = f"token {MAST_API_TOKEN}"
    
    # Create temporary file for download
    temp_path = f"{output_path}.part"
    
    # Check if file already exists at destination
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return True
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Retry loop
    for attempt in range(retries):
        try:
            # Start download
            response = requests.get(download_url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Get file size for progress bar
            file_size = int(response.headers.get('content-length', 0))
            
            # Progress bar for download
            with open(temp_path, 'wb') as f, tqdm(
                desc=os.path.basename(output_path),
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(len(chunk))
            
            # Move the temporary file to the final destination
            shutil.move(temp_path, output_path)
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt+1}/{retries} failed for {url}: {e}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            # Wait before retrying
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
        
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    # If we get here, all retries have failed
    logger.error(f"All {retries} download attempts failed for {url}")
    return False


def parse_csv_file(csv_file):
    """
    Parse a MAST CSV search results file
    
    Args:
        csv_file (str): Path to the CSV file
    
    Returns:
        list: List of observation dictionaries
    """
    observations = []
    
    try:
        with open(csv_file, 'r') as f:
            # Skip comment lines
            line = f.readline()
            while line.startswith('#'):
                line = f.readline()
            
            # Create CSV reader
            reader = csv.DictReader([line] + f.readlines())
            
            # Process each row
            for row in reader:
                observations.append(row)
        
        logger.info(f"Parsed {len(observations)} observations from {csv_file}")
        return observations
    
    except Exception as e:
        logger.error(f"Error parsing CSV file: {e}")
        return []


def filter_observations(observations, target_name=None, proposal_id=None, instrument=None):
    """
    Filter observations based on various criteria
    
    Args:
        observations (list): List of observation dictionaries
        target_name (str, optional): Filter by target name
        proposal_id (str, optional): Filter by proposal ID
        instrument (str, optional): Filter by instrument name
    
    Returns:
        list: Filtered list of observations
    """
    filtered = observations
    
    # Apply filters
    if target_name:
        filtered = [obs for obs in filtered 
                   if 'target_name' in obs and target_name.lower() in obs['target_name'].lower()]
        logger.info(f"Filtered to {len(filtered)} observations with target name containing '{target_name}'")
    
    if proposal_id:
        filtered = [obs for obs in filtered 
                   if 'proposal_id' in obs and proposal_id == obs['proposal_id']]
        logger.info(f"Filtered to {len(filtered)} observations with proposal ID '{proposal_id}'")
    
    if instrument:
        filtered = [obs for obs in filtered 
                   if 'instrument_name' in obs and instrument.lower() in obs['instrument_name'].lower()]
        logger.info(f"Filtered to {len(filtered)} observations with instrument name containing '{instrument}'")
    
    return filtered


def organize_download_path(output_dir, observation):
    """
    Organize download paths based on observation metadata
    
    Args:
        output_dir (str): Base output directory
        observation (dict): Observation dictionary
    
    Returns:
        str: Organized output path
    """
    # Extract relevant fields
    proposal_id = observation.get('proposal_id', 'unknown')
    target_name = observation.get('target_name', 'unknown')
    obs_id = observation.get('obs_id', 'unknown')
    instrument = observation.get('instrument_name', '').split('/')[0] if '/' in observation.get('instrument_name', '') else observation.get('instrument_name', 'unknown')
    
    # Clean up names for use as directory names
    target_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in target_name)
    
    # Extract filename from dataURL
    dataurl = observation.get('dataURL', '')
    filename = os.path.basename(urlparse(dataurl).path)
    
    # If no filename could be extracted, use the observation ID
    if not filename:
        filename = f"{obs_id}.fits"
        
    # Ensure filename ends with .fits
    if not filename.lower().endswith('.fits'):
        filename += '.fits'
    
    # Create directory structure: output_dir/proposal_id/target_name/instrument
    download_dir = os.path.join(output_dir, proposal_id, target_name, instrument)
    
    return os.path.join(download_dir, filename)


def download_observations(observations, output_dir, max_workers=4):
    """
    Download multiple observations with parallel processing
    
    Args:
        observations (list): List of observation dictionaries
        output_dir (str): Base output directory
        max_workers (int, optional): Maximum number of concurrent downloads
    
    Returns:
        DownloadStats: Download statistics
    """
    stats = DownloadStats()
    stats.total = len(observations)
    
    logger.info(f"Starting download of {stats.total} observations to {output_dir}")
    logger.info(f"Using up to {max_workers} concurrent downloads")
    
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a file to track download progress
    progress_file = os.path.join(output_dir, "download_progress.txt")
    
    # Function to download a single observation
    def download_observation(obs):
        # Extract the download URL
        download_url = obs.get('dataURL', '')
        if not download_url:
            logger.warning(f"No download URL for observation {obs.get('obs_id', 'unknown')}")
            stats.failed += 1
            return False
        
        # Organize the download path
        output_path = organize_download_path(output_dir, obs)
        
        # Check if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            stats.skipped += 1
            return True
        
        # Download the file
        logger.info(f"Downloading {download_url} to {output_path}")
        success = download_file(download_url, output_path)
        
        # Update statistics
        stats.update(download_url, success)
        
        # Update progress file
        with open(progress_file, 'a') as f:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{status}: {download_url} -> {output_path}\n")
        
        return success
    
    # Use ThreadPoolExecutor for parallel downloads
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            futures = [executor.submit(download_observation, obs) for obs in observations]
            
            # Process as they complete (optional, futures already handle waiting)
            for future in futures:
                future.result()  # This will re-raise any exceptions
    
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user. Progress has been saved.")
    
    except Exception as e:
        logger.error(f"Error during parallel downloads: {e}")
    
    # Print summary
    logger.info(stats.summary())
    
    # Create a summary file
    summary_file = os.path.join(output_dir, f"download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write(stats.summary())
        f.write("\nFailed downloads:\n")
        for url in stats.failed_urls:
            f.write(f"  {url}\n")
    
    return stats


def save_failed_list(failed_urls, output_dir):
    """
    Save a list of failed downloads for later retry
    
    Args:
        failed_urls (list): List of URLs that failed to download
        output_dir (str): Output directory
    
    Returns:
        str: Path to the failed URLs file
    """
    if not failed_urls:
        return None
    
    filename = os.path.join(output_dir, f"failed_downloads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(filename, 'w') as f:
        for url in failed_urls:
            f.write(f"{url}\n")
    
    logger.info(f"Saved {len(failed_urls)} failed downloads to {filename}")
    return filename


def retry_failed_downloads(failed_file, output_dir):
    """
    Retry downloading files from a list of failed downloads
    
    Args:
        failed_file (str): Path to file containing failed download URLs
        output_dir (str): Output directory
    
    Returns:
        DownloadStats: Download statistics
    """
    try:
        # Read failed URLs
        with open(failed_file, 'r') as f:
            failed_urls = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Retrying {len(failed_urls)} failed downloads")
        
        stats = DownloadStats()
        stats.total = len(failed_urls)
        
        # Retry each download
        for url in failed_urls:
            # Extract filename from URL
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"retry_{stats.total - len(failed_urls)}.fits"
            if not filename.lower().endswith('.fits'):
                filename += '.fits'
            
            # Use a retry directory
            retry_dir = os.path.join(output_dir, "retry")
            os.makedirs(retry_dir, exist_ok=True)
            
            output_path = os.path.join(retry_dir, filename)
            
            # Download the file
            logger.info(f"Retrying download of {url} to {output_path}")
            success = download_file(url, output_path, retries=5)  # More retries for failed files
            
            # Update statistics
            stats.update(url, success)
        
        # Print summary
        logger.info(stats.summary())
        return stats
    
    except Exception as e:
        logger.error(f"Error retrying downloads: {e}")
        return None


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Download FITS files from MAST CSV search results")
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, help="Input CSV file from MAST search")
    parser.add_argument('-o', '--output-dir', required=True, help="Output directory for downloads")
    
    # Optional filters
    parser.add_argument('-p', '--proposal-id', help="Filter by proposal ID")
    parser.add_argument('-t', '--target', help="Filter by target name")
    parser.add_argument('-n', '--instrument', help="Filter by instrument name")
    
    # Download options
    parser.add_argument('-w', '--workers', type=int, default=4, help="Maximum number of concurrent downloads")
    parser.add_argument('-r', '--retry', help="Retry downloads from a file of failed URLs")
    
    args = parser.parse_args()

    try:
        # Retry mode
        if args.retry:
            if not os.path.exists(args.retry):
                logger.error(f"Failed downloads file not found: {args.retry}")
                return 1
            
            retry_failed_downloads(args.retry, args.output_dir)
            return 0
        
        # Normal mode
        observations = parse_csv_file(args.input)
        if not observations:
            logger.error("No observations found in CSV file")
            return 1
        
        # Apply filters if specified
        filtered = filter_observations(
            observations,
            target_name=args.target,
            proposal_id=args.proposal_id,
            instrument=args.instrument
        )
        
        if not filtered:
            logger.error("No observations match the specified filters")
            return 1
        
        # Download observations
        stats = download_observations(filtered, args.output_dir, max_workers=args.workers)
        
        # Save failed downloads for later retry
        if stats.failed_urls:
            failed_file = save_failed_list(stats.failed_urls, args.output_dir)
            logger.info(f"To retry failed downloads, run:\n  python {sys.argv[0]} -o {args.output_dir} -r {failed_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())