"""
Economic Times PDF Downloader
Downloads PDF newspapers from epaperwave.com with duplicate checking
"""

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

class EconomicTimesPDFDownloader:
    """
    Download Economic Times PDF newspapers from epaperwave.com
    Includes duplicate checking and download tracking
    """
    
    def __init__(self, download_dir="economic_times_pdfs"):
        """
        Initialize downloader
        
        Args:
            download_dir: Directory to save downloaded PDFs
        """
        self.base_url = "https://epaperwave.com/the-economic-times-newspaper-download/"
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Track downloaded files
        self.tracking_file = self.download_dir / "download_history.json"
        self.download_history = self._load_history()
        
        # Setup session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    def _load_history(self):
        """Load download history from JSON file"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_history(self):
        """Save download history to JSON file"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.download_history, f, indent=2)
    
    def _get_file_hash(self, filepath):
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _is_duplicate(self, url, filepath):
        """Check if file is a duplicate"""
        # Check if URL already downloaded
        if url in self.download_history:
            print(f"   ⏭️ Already downloaded: {url}")
            return True
        
        # Check if file with same name exists
        if filepath.exists():
            print(f"   ⏭️ File already exists: {filepath.name}")
            return True
        
        return False
    
    def fetch_page_links(self):
        """
        Fetch all PDF download links from the main page
        
        Returns:
            List of tuples: (title, pdf_url, date_str)
        """
        print(f"🌐 Fetching page: {self.base_url}")
        
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            pdf_links = []
            
            # Find all PDF links (adjust selectors based on actual website structure)
            # This is a common pattern - may need adjustment
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Look for PDF links
                if '.pdf' in href.lower() or 'download' in href.lower():
                    # Extract date from link or text if possible
                    date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
                    date_str = date_match.group(1) if date_match else datetime.now().strftime('%Y-%m-%d')
                    
                    # Make absolute URL
                    if not href.startswith('http'):
                        href = requests.compat.urljoin(self.base_url, href)
                    
                    pdf_links.append((text, href, date_str))
            
            print(f"✅ Found {len(pdf_links)} potential PDF links")
            return pdf_links
            
        except Exception as e:
            print(f"❌ Error fetching page: {e}")
            return []
    
    def download_pdf(self, title, url, date_str):
        """
        Download a single PDF file
        
        Args:
            title: Title/description of the PDF
            url: Download URL
            date_str: Date string for filename
            
        Returns:
            Path to downloaded file or None if failed/skipped
        """
        # Create safe filename
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)[:50]
        
        filename = f"ET_{date_str}_{safe_title}.pdf"
        filepath = self.download_dir / filename
        
        # Check for duplicates
        if self._is_duplicate(url, filepath):
            return None
        
        print(f"\n📥 Downloading: {title}")
        print(f"   URL: {url}")
        print(f"   File: {filename}")
        
        try:
            # Download with progress
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = (downloaded / total_size) * 100
                            print(f"   Progress: {progress:.1f}%", end='\r')
            
            print(f"\n   ✅ Downloaded: {filepath.name} ({total_size / 1024 / 1024:.2f} MB)")
            
            # Update history
            file_hash = self._get_file_hash(filepath)
            self.download_history[url] = {
                'filename': filename,
                'title': title,
                'date': date_str,
                'download_time': datetime.now().isoformat(),
                'file_hash': file_hash,
                'size_mb': total_size / 1024 / 1024
            }
            self._save_history()
            
            return filepath
            
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return None
    
    def download_recent_newspapers(self, max_count=10, delay=2):
        """
        Download recent newspapers from the website
        
        Args:
            max_count: Maximum number of PDFs to download
            delay: Delay between downloads in seconds
        """
        print("=" * 70)
        print("ECONOMIC TIMES PDF DOWNLOADER")
        print("=" * 70)
        
        # Fetch available links
        pdf_links = self.fetch_page_links()
        
        if not pdf_links:
            print("\n⚠️ No PDF links found. The website structure may have changed.")
            return
        
        # Download PDFs
        downloaded = 0
        skipped = 0
        failed = 0
        
        for i, (title, url, date_str) in enumerate(pdf_links[:max_count], 1):
            print(f"\n[{i}/{min(len(pdf_links), max_count)}]")
            
            result = self.download_pdf(title, url, date_str)
            
            if result:
                downloaded += 1
            elif result is None and url in self.download_history:
                skipped += 1
            else:
                failed += 1
            
            # Delay between downloads
            if i < min(len(pdf_links), max_count):
                time.sleep(delay)
        
        # Summary
        print("\n" + "=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"✅ Downloaded: {downloaded} new PDFs")
        print(f"⏭️ Skipped: {skipped} (already downloaded)")
        print(f"❌ Failed: {failed}")
        print(f"📁 Save location: {self.download_dir.absolute()}")
        print(f"📊 Total tracked: {len(self.download_history)} files")
        print("=" * 70)
    
    def list_downloaded_files(self):
        """List all downloaded files"""
        print("\n📚 DOWNLOADED FILES:")
        print("-" * 70)
        
        if not self.download_history:
            print("No files downloaded yet.")
            return
        
        for url, info in sorted(self.download_history.items(), 
                               key=lambda x: x[1]['download_time'], 
                               reverse=True):
            print(f"\n📄 {info['filename']}")
            print(f"   Title: {info['title']}")
            print(f"   Date: {info['date']}")
            print(f"   Downloaded: {info['download_time']}")
            print(f"   Size: {info['size_mb']:.2f} MB")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download Economic Times PDF newspapers'
    )
    parser.add_argument(
        '--max-count', 
        type=int, 
        default=5,
        help='Maximum number of PDFs to download (default: 5)'
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=2,
        help='Delay between downloads in seconds (default: 2)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List already downloaded files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='economic_times_pdfs',
        help='Output directory for PDFs (default: economic_times_pdfs)'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = EconomicTimesPDFDownloader(download_dir=args.output_dir)
    
    if args.list:
        downloader.list_downloaded_files()
    else:
        downloader.download_recent_newspapers(
            max_count=args.max_count,
            delay=args.delay
        )


if __name__ == "__main__":
    main()
