"""
Economic Times PDF Downloader
==============================

Downloads PDF editions of The Economic Times newspaper from epaperwave.com
Automatically checks for duplicates and saves to the data folder.

Features:
- Downloads multiple PDF editions
- Prevents duplicate downloads
- Organizes files by date
- Provides download progress
- Error handling and retry logic

Usage:
    python economic_times_downloader.py

Author: Market Prediction System
Date: October 2025
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

class EconomicTimesDownloader:
    """Download Economic Times PDFs from epaperwave.com"""
    
    def __init__(self, data_dir="../data", url_choice=1):
        """
        Initialize the downloader
        
        Args:
            data_dir: Directory to save downloaded PDFs
            url_choice: 1 for first URL, 2 for second URL, 3 for both
        """
        # Choose URL based on user preference
        if url_choice == 2:
            self.base_url = "https://epaperwave.com/download-the-economic-times-pdf-newspaper-free/"
            self.source_name = "ET_Free"
        elif url_choice == 3:
            self.base_url = None  # Will process both URLs
            self.source_name = "ET_Both"
        else:
            self.base_url = "https://epaperwave.com/the-economic-times-newspaper-download/"
            self.source_name = "ET"
            
        self.data_dir = Path(data_dir)
        self.pdfs_dir = self.data_dir / "economic_times_pdfs"
        self.metadata_file = self.data_dir / "downloaded_files.json"
        
        # Create directories
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.downloaded_files = self._load_metadata()
        
        # Request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
    
    def _load_metadata(self):
        """Load metadata of previously downloaded files"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save metadata of downloaded files"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.downloaded_files, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save metadata: {e}")
    
    def _get_file_hash(self, filepath):
        """Calculate MD5 hash of a file"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"⚠️ Could not calculate hash: {e}")
            return None
    
    def _is_duplicate(self, url, filepath):
        """Check if file is already downloaded"""
        # Check by URL
        if url in self.downloaded_files:
            return True
        
        # Check by filename
        if filepath.exists():
            print(f"  ⚠️ File already exists: {filepath.name}")
            return True
        
        return False
    
    def fetch_page(self, url=None):
        """Fetch the main page and extract PDF links with dates"""
        target_url = url or self.base_url
        print(f"🌐 Fetching page: {target_url}")
        
        try:
            response = requests.get(target_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all PDF links with dates
            pdf_links = []
            seen_urls = set()
            
            # Pattern to match dates like "31-07-2025:" or "31-07-2025 –"
            date_pattern = re.compile(r'(\d{2}-\d{2}-\d{4})\s*[:\-–]')
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = str(link['href'])
                
                # Check if it's a download link (Google Drive, direct PDF, etc.)
                if 'drive.google.com' in href or 'drive.usercontent.google.com' in href or '.pdf' in href.lower():
                    # Try to find the date context before this link
                    # Get the parent element's text
                    parent_text = link.parent.get_text() if link.parent else ""
                    
                    # Look for date pattern in the parent text
                    date_match = date_pattern.search(parent_text)
                    if date_match:
                        date_str = date_match.group(1)
                    else:
                        # Try to find date in the broader context (previous siblings)
                        try:
                            # Get text before the link
                            prev_text = ""
                            for prev in link.find_all_previous(text=True, limit=5):
                                prev_str = str(prev) if prev else ""
                                prev_text = prev_str + " " + prev_text
                                date_match = date_pattern.search(prev_text)
                                if date_match:
                                    date_str = date_match.group(1)
                                    break
                            else:
                                date_str = None
                        except:
                            date_str = None
                    
                    # Only add if we found a date and URL is unique
                    if date_str and href not in seen_urls:
                        full_url = urljoin(target_url, href)
                        title = link.get_text(strip=True) or "Economic Times"
                        
                        pdf_links.append({
                            'url': full_url,
                            'title': title,
                            'date': date_str,
                            'link_element': link
                        })
                        seen_urls.add(href)
            
            print(f"📄 Found {len(pdf_links)} PDF links with dates")
            return pdf_links
            
        except requests.RequestException as e:
            print(f"❌ Error fetching page: {e}")
            return []
    
    def extract_date_from_title(self, title):
        """Extract date from title if possible"""
        # Common date patterns
        patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD-MM-YYYY or DD/MM/YYYY
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})',  # DD Month YYYY
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return datetime.now().strftime('%Y-%m-%d')
    
    def _convert_google_drive_url(self, url):
        """Convert Google Drive view URL to direct download URL"""
        # Pattern 1: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        # Convert to: https://drive.usercontent.google.com/download?id=FILE_ID&export=download
        
        if 'drive.google.com/file/d/' in url:
            try:
                file_id = url.split('/d/')[1].split('/')[0].split('?')[0]
                return f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
            except:
                pass
        
        # Pattern 2: Already in usercontent format
        if 'drive.usercontent.google.com' in url and 'export=download' not in url:
            if '?' in url:
                return url + '&export=download'
            else:
                file_id = url.split('id=')[1].split('&')[0]
                return f"{url}&export=download"
        
        return url
    
    def download_pdf(self, pdf_info, retry=3):
        """
        Download a single PDF file
        
        Args:
            pdf_info: Dictionary with 'url', 'title', and 'date'
            retry: Number of retry attempts
        """
        url = pdf_info['url']
        title = pdf_info['title']
        date_str = pdf_info.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # Convert Google Drive URL to direct download
        url = self._convert_google_drive_url(url)
        
        # Convert date format from DD-MM-YYYY to YYYY-MM-DD for better sorting
        try:
            if '-' in date_str and len(date_str) == 10:
                parts = date_str.split('-')
                if len(parts) == 3 and len(parts[2]) == 4:  # DD-MM-YYYY
                    date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"  # Convert to YYYY-MM-DD
        except:
            pass
        
        # Create filename
        safe_title = re.sub(r'[^\w\s-]', '', title)[:30]
        safe_title = re.sub(r'[\s]+', '_', safe_title)
        filename = f"ET_{date_str}.pdf"
        filepath = self.pdfs_dir / filename
        
        # Check for duplicates
        if self._is_duplicate(url, filepath):
            print(f"  ⏭️ Skipping duplicate: {filename}")
            return False
        
        print(f"\n📥 Downloading: {date_str}")
        print(f"   URL: {url[:80]}...")
        print(f"   Saving as: {filename}")
        
        for attempt in range(retry):
            try:
                # Download with progress
                response = requests.get(url, headers=self.headers, stream=True, timeout=60)
                response.raise_for_status()
                
                # Check if it's actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    print(f"  ⚠️ Not a PDF file (content-type: {content_type})")
                    return False
                
                # Get file size
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar
                downloaded = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"  Progress: {percent:.1f}% ({downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB)", end='\r')
                
                print(f"\n  ✅ Downloaded successfully: {filepath.name}")
                
                # Save metadata
                file_hash = self._get_file_hash(filepath)
                self.downloaded_files[url] = {
                    'filename': filename,
                    'title': title,
                    'download_date': datetime.now().isoformat(),
                    'file_size': filepath.stat().st_size,
                    'hash': file_hash
                }
                self._save_metadata()
                
                return True
                
            except requests.RequestException as e:
                print(f"  ❌ Download failed (attempt {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    print(f"  ⏳ Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"  ❌ Failed after {retry} attempts")
                    return False
            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                return False
    
    def download_all(self, max_files=None):
        """
        Download all available PDFs
        
        Args:
            max_files: Maximum number of files to download (None = download all)
        """
        print("="*60)
        print("ECONOMIC TIMES PDF DOWNLOADER")
        print("="*60)
        
        # If url_choice=3, process both URLs
        if self.base_url is None:
            urls_to_process = [
                ("https://epaperwave.com/the-economic-times-newspaper-download/", "ET"),
                ("https://epaperwave.com/download-the-economic-times-pdf-newspaper-free/", "ET_Free")
            ]
        else:
            urls_to_process = [(self.base_url, self.source_name)]
        
        total_success = 0
        total_skip = 0
        total_fail = 0
        
        for url, source in urls_to_process:
            print(f"\n{'='*60}")
            print(f"Processing: {url}")
            print(f"Source name: {source}")
            print(f"{'='*60}")
            
            # Fetch available PDFs
            # Temporarily set source name for this URL
            old_source = self.source_name
            self.source_name = source
            
            pdf_links = self.fetch_page(url)
            
            if not pdf_links:
                print(f"❌ No PDF links found on {url}")
                self.source_name = old_source
                continue
            
            # Limit number of files if specified
            if max_files is not None:
                pdf_links = pdf_links[:max_files]
                print(f"\n📊 Will attempt to download {len(pdf_links)} PDFs")
            else:
                print(f"\n📊 Will attempt to download ALL {len(pdf_links)} PDFs")
            print(f"📁 Save location: {self.pdfs_dir.absolute()}")
            
            # Download each PDF
            success_count = 0
            skip_count = 0
            fail_count = 0
            
            for i, pdf_info in enumerate(pdf_links, 1):
                print(f"\n[{i}/{len(pdf_links)}]", end=' ')
                
                result = self.download_pdf(pdf_info)
                
                if result is True:
                    success_count += 1
                elif result is False:
                    skip_count += 1
                else:
                    fail_count += 1
                
                # Be nice to the server
                if i < len(pdf_links):
                    time.sleep(2)
            
            # Summary for this URL
            print(f"\n{'-'*60}")
            print(f"Source {source} Summary:")
            print(f"✅ Successfully downloaded: {success_count}")
            print(f"⏭️ Skipped (duplicates): {skip_count}")
            print(f"❌ Failed: {fail_count}")
            
            total_success += success_count
            total_skip += skip_count
            total_fail += fail_count
            
            # Restore source name
            self.source_name = old_source
        
        # Final Summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY (ALL SOURCES)")
        print("="*60)
        print(f"✅ Successfully downloaded: {total_success}")
        print(f"⏭️ Skipped (duplicates): {total_skip}")
        print(f"❌ Failed: {total_fail}")
        print(f"📁 Files saved to: {self.pdfs_dir.absolute()}")
        print(f"📋 Metadata saved to: {self.metadata_file.absolute()}")
        print("="*60)
    
    def list_downloaded_files(self):
        """List all downloaded files"""
        print("\n📚 Downloaded Files:")
        print("-" * 60)
        
        if not self.downloaded_files:
            print("No files downloaded yet.")
            return
        
        for i, (url, info) in enumerate(self.downloaded_files.items(), 1):
            print(f"{i}. {info['filename']}")
            print(f"   Title: {info['title']}")
            print(f"   Downloaded: {info['download_date']}")
            print(f"   Size: {info['file_size'] / 1024 / 1024:.2f} MB")
            print("-" * 60)

def main():
    """Main function"""
    # Ask user to choose URL
    print("="*60)
    print("ECONOMIC TIMES PDF DOWNLOADER")
    print("="*60)
    print("\nChoose source:")
    print("1. Economic Times Main Page (Original)")
    print("   URL: https://epaperwave.com/the-economic-times-newspaper-download/")
    print("2. Economic Times Free Download Page")
    print("   URL: https://epaperwave.com/download-the-economic-times-pdf-newspaper-free/")
    print("3. Both URLs (Download from both sources)")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            url_choice = int(choice)
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    # Initialize downloader with chosen URL
    downloader = EconomicTimesDownloader(data_dir="../data", url_choice=url_choice)
    
    # Show existing downloads
    if downloader.downloaded_files:
        print(f"\n📚 Found {len(downloader.downloaded_files)} previously downloaded files")
        response = input("Show list? (y/n): ")
        if response.lower() == 'y':
            downloader.list_downloaded_files()
    
    # Download all available files
    print("\n" + "="*60)
    response = input("Download ALL available PDFs? (y/n, default y): ").strip().lower()
    
    if response == 'n':
        max_files = input("How many PDFs to download? (enter number): ").strip()
        max_files = int(max_files) if max_files.isdigit() else None
    else:
        max_files = None  # Download all
    
    downloader.download_all(max_files=max_files)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
