#!/usr/bin/env python3
"""
astropy_csv_downloader.py - Simple CSV-based MAST file downloader

A streamlined script to download FITS files from MAST based on a CSV file,
using astroquery.mast for robust handling of astronomical data.

Example usage:
    python astropy_csv_downloader.py -i data/download_list.csv -o downloads

The CSV should contain at least one of these columns:
- dataURL or dataURI: direct download links
- obsID/obs_id and productID/product_id: identifiers for astroquery

The script handles resume capability and provides detailed progress feedback.
"""

import os
import sys
import csv
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import traceback
from urllib.parse import urlparse
import signal

# Astropy imports
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astroquery.mast import Observations
from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError
from astropy.utils.console import ProgressBar
from astropy.utils.data import download_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mast_download.log")
    ]
)
logger = logging.getLogger('astropy_downloader')

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DEFAULT_TIMEOUT = 180  # seconds

# Global variable to track if Ctrl+C was pressed
interrupted = False

class DownloadTracker:
    """Class to track download progress with resume capability"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.status_file = os.path.join(output_dir, "download_status.json")
        self.start_time = datetime.now()
        self.load_status()
        
    def load_status(self):
        """Load existing status or initialize new status"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    self.status = json.load(f)
                print(f"\nLoaded existing status: {self.status['downloaded']}/{self.status['total']} files downloaded")
                logger.info(f"Loaded existing status: {self.status['downloaded']}/{self.status['total']} files downloaded")
            except Exception as e:
                logger.warning(f"Error loading status file: {e}. Creating new status.")
                self.initialize_status()
        else:
            self.initialize_status()
            
    def initialize_status(self):
        """Initialize a new status dictionary"""
        self.status = {
            'timestamp': datetime.now().isoformat(),
            'total': 0,
            'downloaded': 0,
            'failed': 0,
            'completed_files': [],
            'failed_files': [],
            'in_progress': False
        }
        
    def save_status(self):
        """Save current status to file"""
        self.status['last_updated'] = datetime.now().isoformat()
        
        try:
            os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving status file: {e}")
            
    def update(self, filename, success):
        """Update status with download result"""
        if success:
            self.status['downloaded'] += 1
            if filename not in self.status['completed_files']:
                self.status['completed_files'].append(filename)
        else:
            self.status['failed'] += 1
            if filename not in self.status['failed_files']:
                self.status['failed_files'].append(filename)
        
        self.save_status()
        
    def is_completed(self, filename):
        """Check if a file was already downloaded successfully"""
        return filename in self.status['completed_files']
    
    def print_progress(self, current_file_index):
        """Print progress information"""
        if self.status['total'] <= 0:
            return
            
        current = self.status['downloaded'] + self.status['failed']
        percent = (current / self.status['total']) * 100
        
        # Calculate estimated time remaining
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if current > 0:
            estimated_total = elapsed * self.status['total'] / current
            remaining = estimated_total - elapsed
            time_remaining = str(timedelta(seconds=int(remaining)))
        else:
            time_remaining = "unknown"
            
        print(f"\r[{current}/{self.status['total']}] {percent:.1f}% complete | "
              f"File {current_file_index + 1}/{self.status['total']} | "
              f"ETA: {time_remaining}", end="", flush=True)
    
    def summary(self):
        """Get a summary of download status"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        
        return {
            'total': self.status['total'],
            'downloaded': self.status['downloaded'],
            'failed': self.status['failed'],
            'remaining': self.status['total'] - self.status['downloaded'] - self.status['failed'],
            'elapsed': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            'failed_files': self.status['failed_files']
        }

def signal_handler(sig, frame):
    """Handle Ctrl+C interruption"""
    global interrupted
    interrupted = True
    print("\n\nInterrupted by user. Cleaning up before exit...", flush=True)
    logger.warning("Download interrupted by user (Ctrl+C)")
    
# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def parse_csv_file(csv_file):
    """
    Parse a CSV file containing download information
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
        
    Returns:
    --------
    list : List of dictionaries with download information
    """
    downloads = []
    
    try:
        logger.info(f"Reading CSV file: {csv_file}")
        print(f"Reading CSV file: {csv_file}")
        
        # Try to open and read the CSV
        with open(csv_file, 'r') as f:
            # Skip comment lines at the beginning if present
            lines = []
            for line in f:
                if not line.startswith('#'):
                    lines.append(line)
            
            # Reset and create reader
            reader = csv.DictReader(lines)
            
            for row in reader:
                # Clean up row - convert empty strings to None and strip whitespace
                clean_row = {k: v.strip() if isinstance(v, str) else v for k, v in row.items() if v not in ('', None)}
                downloads.append(clean_row)
        
        logger.info(f"Found {len(downloads)} entries in CSV file")
        print(f"Found {len(downloads)} entries in CSV file")
        
        # Check for required fields
        if downloads:
            valid_entries = 0
            for i, entry in enumerate(downloads):
                has_url = any(field in entry for field in ['dataURL', 'dataURI', 'URI', 'URL'])
                has_id = ('obsID' in entry or 'obs_id' in entry) and ('productID' in entry or 'product_id' in entry)
                
                if has_url or has_id:
                    valid_entries += 1
                else:
                    logger.warning(f"Entry {i+1} lacks required fields (dataURL/dataURI or obsID+productID)")
            
            print(f"Valid entries for download: {valid_entries}/{len(downloads)}")
        
        return downloads
    
    except Exception as e:
        logger.error(f"Error parsing CSV file: {e}")
        traceback.print_exc()
        print(f"Error parsing CSV file: {e}")
        return []

def get_output_filename(entry, output_dir):
    """
    Determine the output filename for a download entry
    
    Parameters:
    -----------
    entry : dict
        Dictionary with download information
    output_dir : str
        Output directory
        
    Returns:
    --------
    str : Full path to the output file
    """
    # Check if the entry has a filename specified
    filename = None
    
    # Try various possible field names for the filename
    for field in ['filename', 'productFilename', 'file_name']:
        if field in entry and entry[field]:
            filename = entry[field]
            break
    
    # If no filename is specified, try to extract from URL
    if not filename and ('dataURL' in entry or 'dataURI' in entry):
        url = entry.get('dataURL', entry.get('dataURI', ''))
        try:
            filename = os.path.basename(urlparse(url).path)
        except:
            pass
    
    # If still no filename, construct one from IDs
    if not filename:
        obs_id = entry.get('obsID', entry.get('obs_id', ''))
        product_id = entry.get('productID', entry.get('product_id', ''))
        
        if obs_id and product_id:
            filename = f"{obs_id}_{product_id}.fits"
        elif obs_id:
            filename = f"{obs_id}.fits"
        elif product_id:
            filename = f"{product_id}.fits"
        else:
            # Last resort: random name
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"download_{timestamp}.fits"
    
    # Ensure filename ends with .fits
    if not filename.lower().endswith('.fits'):
        filename += '.fits'
    
    # Sanitize filename (replace invalid characters)
    filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    
    # Construct full path
    full_path = os.path.join(output_dir, filename)
    
    return full_path

def download_via_astroquery(entry, output_path, timeout=DEFAULT_TIMEOUT):
    """
    Download a file using astroquery.mast
    
    Parameters:
    -----------
    entry : dict
        Dictionary with download information
    output_path : str
        Full path where to save the file
    timeout : int
        Timeout in seconds
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    global interrupted
    
    # Set the timeout for Astroquery
    Observations.TIMEOUT = timeout
    
    try:
        # Check if we have obsID and productID
        obs_id = entry.get('obsID', entry.get('obs_id', ''))
        product_id = entry.get('productID', entry.get('product_id', ''))
        
        if not (obs_id and product_id):
            logger.error("Missing obsID or productID for astroquery download")
            print(f"\nMissing obsID or productID for astroquery download")
            return False
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a data products list
        products = Table()
        products['obsID'] = [obs_id]
        products['productID'] = [product_id]
        
        # Get the filename part of the output path
        filename = os.path.basename(output_path)
        target_dir = os.path.dirname(output_path)
        
        logger.info(f"Downloading obsID={obs_id}, productID={product_id} to {output_path}")
        print(f"\nDownloading obsID={obs_id}, productID={product_id}")
        
        # Download the file
        result = Observations.download_products(
            products,
            download_dir=target_dir,
            cache=False
        )
        
        if interrupted:
            return False
        
        # Check if the download was successful
        if len(result) > 0 and result['Local Path'][0]:
            downloaded_path = result['Local Path'][0]
            
            # If the downloaded file has a different name, rename it
            if os.path.basename(downloaded_path) != filename:
                try:
                    os.rename(downloaded_path, output_path)
                    logger.info(f"Renamed {os.path.basename(downloaded_path)} to {filename}")
                except Exception as e:
                    logger.warning(f"Error renaming file: {e}")
            
            print(f"\nSuccessfully downloaded {filename}")
            logger.info(f"Successfully downloaded to {output_path}")
            return True
        else:
            print(f"\nDownload failed: No files returned from Observations.download_products")
            logger.error("Download failed: No files returned from Observations.download_products")
            return False
            
    except Exception as e:
        print(f"\nError in astroquery download: {e}")
        logger.error(f"Error in astroquery download: {e}")
        traceback.print_exc()
        return False

def download_direct(url, output_path, retries=MAX_RETRIES):
    """
    Download a file directly from a URL
    
    Parameters:
    -----------
    url : str
        URL to download
    output_path : str
        Full path where to save the file
    retries : int
        Number of retries for failed downloads
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    global interrupted
    
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Downloading from URL: {url}")
        print(f"\nDownloading from URL: {url}")
        
        # Download with retries
        for attempt in range(retries):
            if interrupted:
                return False
                
            try:
                # Download to a temporary file first
                temp_path = f"{output_path}.part"
                
                # Use Astropy's download_file with progress bar
                print(f"\nDownload attempt {attempt+1}/{retries}:")
                dl_path = download_file(url, cache=False, show_progress=True, timeout=DEFAULT_TIMEOUT)
                
                if interrupted:
                    # Try to clean up if interrupted during download
                    if os.path.exists(dl_path):
                        try:
                            os.remove(dl_path)
                        except:
                            pass
                    return False
                
                # Move to the target location
                import shutil
                shutil.move(dl_path, output_path)
                
                print(f"\nSuccessfully downloaded to {os.path.basename(output_path)}")
                logger.info(f"Successfully downloaded to {output_path}")
                return True
                
            except Exception as e:
                print(f"\nDownload attempt {attempt+1}/{retries} failed: {e}")
                logger.warning(f"Download attempt {attempt+1}/{retries} failed: {e}")
                
                # Clean up partial file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # Wait before retrying (unless interrupted)
                if attempt < retries - 1 and not interrupted:
                    delay = RETRY_DELAY * (attempt + 1)
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
        
        # If we get here, all retries failed
        print(f"\nFailed to download after {retries} attempts")
        logger.error(f"Failed to download after {retries} attempts")
        return False
    
    except Exception as e:
        print(f"\nError in direct download: {e}")
        logger.error(f"Error in direct download: {e}")
        return False

def process_entry(entry, output_dir, tracker, index):
    """
    Process a single download entry
    
    Parameters:
    -----------
    entry : dict
        Dictionary with download information
    output_dir : str
        Base output directory
    tracker : DownloadTracker
        Download tracker instance
    index : int
        Current entry index for progress reporting
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    global interrupted
    
    if interrupted:
        return False
    
    # Print progress
    tracker.print_progress(index)
    
    # Determine output filename
    output_path = get_output_filename(entry, output_dir)
    filename = os.path.basename(output_path)
    
    # Check if already downloaded
    if tracker.is_completed(filename) or os.path.exists(output_path):
        logger.info(f"File already exists: {filename}")
        return True
    
    # Determine download method
    success = False
    
    # First try direct URL if available
    if any(field in entry for field in ['dataURL', 'dataURI', 'URL', 'URI']):
        url = entry.get('dataURL', entry.get('dataURI', entry.get('URL', entry.get('URI', ''))))
        if url:
            success = download_direct(url, output_path)
    
    # If direct download failed or no URL, try astroquery
    if not success and not interrupted and (('obsID' in entry or 'obs_id' in entry) and 
                                          ('productID' in entry or 'product_id' in entry)):
        success = download_via_astroquery(entry, output_path)
    
    # Update tracker if not interrupted
    if not interrupted:
        tracker.update(filename, success)
    
    return success

def download_from_csv(csv_file, output_dir, resume=True):
    """
    Download files listed in a CSV file
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with download information
    output_dir : str
        Output directory
    resume : bool
        Whether to resume previous downloads
        
    Returns:
    --------
    dict : Download summary
    """
    global interrupted
    interrupted = False
    
    # Initialize
    os.makedirs(output_dir, exist_ok=True)
    downloads = parse_csv_file(csv_file)
    
    if not downloads:
        print("No valid download entries found in CSV")
        logger.error("No valid download entries found in CSV")
        return None
    
    # Initialize download tracker
    tracker = DownloadTracker(output_dir)
    tracker.status['total'] = len(downloads)
    tracker.status['in_progress'] = True
    tracker.save_status()
    
    logger.info(f"Starting download of {len(downloads)} files to {output_dir}")
    print(f"\nStarting download of {len(downloads)} files to {output_dir}")
    print("Press Ctrl+C at any time to stop downloading safely\n")
    
    try:
        # Process entries sequentially
        for i, entry in enumerate(downloads):
            if interrupted:
                break
                
            # Skip already downloaded files if resuming
            if resume:
                output_path = get_output_filename(entry, output_dir)
                filename = os.path.basename(output_path)
                if tracker.is_completed(filename) or os.path.exists(output_path):
                    logger.info(f"Skipping already downloaded file: {filename}")
                    tracker.print_progress(i)
                    continue
            
            # Process the entry
            process_entry(entry, output_dir, tracker, i)
    
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        logger.warning("Download interrupted by user (KeyboardInterrupt)")
        interrupted = True
    
    except Exception as e:
        print(f"\n\nError during download process: {e}")
        logger.error(f"Error during download process: {e}")
        traceback.print_exc()
    
    finally:
        # Mark as complete
        tracker.status['in_progress'] = False
        tracker.save_status()
        
        # Print summary
        summary = tracker.summary()
        print("\n\n=== Download Summary ===")
        print(f"Total files: {summary['total']}")
        print(f"Downloaded: {summary['downloaded']}")
        print(f"Failed: {summary['failed']}")
        print(f"Remaining: {summary['remaining']}")
        print(f"Time elapsed: {summary['elapsed']}")
        
        logger.info("\n=== Download Summary ===")
        logger.info(f"Total files: {summary['total']}")
        logger.info(f"Downloaded: {summary['downloaded']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Remaining: {summary['remaining']}")
        logger.info(f"Time elapsed: {summary['elapsed']}")
        
        # List failed files
        if summary['failed_files']:
            print("\nFailed files:")
            for f in summary['failed_files'][:10]:
                print(f"  - {f}")
            
            if len(summary['failed_files']) > 10:
                print(f"  ... and {len(summary['failed_files']) - 10} more")
            
            print(f"\nTo retry failed downloads, run the script again with the same parameters.")
        
        if interrupted:
            print("\nDownload was interrupted. Run the script again with the same parameters to resume.")
        
        return summary

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Download FITS files from MAST using CSV input")
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, help="Input CSV file containing download information")
    parser.add_argument('-o', '--output-dir', required=True, help="Output directory for downloads")
    
    # Optional arguments
    parser.add_argument('-n', '--no-resume', action='store_true', help="Do not resume previous downloads")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Download files
    result = download_from_csv(
        args.input,
        args.output_dir,
        resume=not args.no_resume
    )
    
    if result is None:
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())