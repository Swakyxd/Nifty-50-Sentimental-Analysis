# Economic Times PDF Downloader

Automated downloader for The Economic Times newspaper PDF editions from epaperwave.com

## 🚀 Features

- ✅ **Automatic PDF Download** - Downloads Economic Times newspaper PDFs
- ✅ **Duplicate Prevention** - Tracks downloaded files to avoid re-downloading
- ✅ **Organized Storage** - Saves PDFs with date-based filenames
- ✅ **Progress Tracking** - Shows download progress and statistics
- ✅ **Retry Logic** - Automatically retries failed downloads
- ✅ **Metadata Management** - Maintains a record of all downloads

## 📁 Directory Structure

```
newspaper/
├── scripts/
│   └── economic_times_downloader.py    # Main downloader script
├── data/
│   ├── economic_times_pdfs/            # Downloaded PDF files
│   └── downloaded_files.json           # Download metadata
└── README.md                            # This file
```

## 🔧 Installation

### Requirements
```bash
pip install beautifulsoup4 requests
```

Or if using the DL-Project environment:
```powershell
F:/anaconda/envs/DL-Project/python.exe -m pip install beautifulsoup4 requests
```

## 📖 Usage

### Basic Usage

```powershell
# Navigate to the scripts folder
cd F:\DL-Project\newspaper\scripts

# Run the downloader
F:/anaconda/envs/DL-Project/python.exe economic_times_downloader.py
```

The script will:
1. Check for previously downloaded files
2. Ask how many PDFs you want to download
3. Fetch available PDFs from the website
4. Download new files (skipping duplicates)
5. Save metadata for tracking

### Interactive Prompts

```
📚 Found X previously downloaded files
Show list? (y/n): y

============================================================
How many PDFs to download? (default 5): 10
```

### Output

PDFs are saved to: `F:\DL-Project\newspaper\data\economic_times_pdfs/`

Filename format: `ET_YYYY-MM-DD_Title.pdf`

Example: `ET_2025-10-26_Economic_Times_Delhi.pdf`

## 📊 Features in Detail

### Duplicate Prevention

The script maintains a `downloaded_files.json` that tracks:
- File URL
- Filename
- Download date
- File size
- MD5 hash

Files are skipped if:
- URL already exists in metadata
- File with same name already exists

### Progress Display

```
📥 Downloading: Economic Times - Delhi Edition
   URL: https://example.com/pdf/et-2025-10-26.pdf
   Saving as: ET_2025-10-26_Delhi_Edition.pdf
  Progress: 45.2% (2.35MB / 5.20MB)
  ✅ Downloaded successfully: ET_2025-10-26_Delhi_Edition.pdf
```

### Download Summary

```
============================================================
DOWNLOAD SUMMARY
============================================================
✅ Successfully downloaded: 8
⏭️ Skipped (duplicates): 2
❌ Failed: 0
📁 Files saved to: F:\DL-Project\newspaper\data\economic_times_pdfs
📋 Metadata saved to: F:\DL-Project\newspaper\data\downloaded_files.json
============================================================
```

## 🛠️ Advanced Usage

### Custom Data Directory

```python
from economic_times_downloader import EconomicTimesDownloader

# Use custom directory
downloader = EconomicTimesDownloader(data_dir="/path/to/custom/data")
downloader.download_all(max_files=10)
```

### List Downloaded Files

```python
downloader = EconomicTimesDownloader()
downloader.list_downloaded_files()
```

### Check for Specific Date

The script automatically extracts dates from PDF titles and organizes files accordingly.

## ⚙️ Configuration

Edit the script to customize:

```python
# Change base URL if needed
self.base_url = "https://epaperwave.com/the-economic-times-newspaper-download/"

# Adjust retry attempts
def download_pdf(self, pdf_info, retry=3):  # Change retry count

# Modify download delay
time.sleep(2)  # Seconds between downloads
```

## 🔍 Troubleshooting

### Issue: No PDFs found
**Solution:** The website structure may have changed. Check if the URL is still valid.

### Issue: Download fails repeatedly
**Solution:** 
- Check your internet connection
- Verify the website is accessible
- Increase retry attempts in the code

### Issue: "Not a PDF file" warning
**Solution:** The link may point to a webpage instead of a direct PDF. The script will skip these automatically.

### Issue: Permission error when saving
**Solution:** Ensure you have write permissions to the `data/` directory.

## 📝 Metadata Format

The `downloaded_files.json` stores:

```json
{
  "https://example.com/pdf/file.pdf": {
    "filename": "ET_2025-10-26_Delhi_Edition.pdf",
    "title": "Economic Times Delhi Edition",
    "download_date": "2025-10-26T14:30:00",
    "file_size": 5242880,
    "hash": "a1b2c3d4e5f6..."
  }
}
```

## 🚨 Important Notes

- **Rate Limiting**: Script includes 2-second delays between downloads to be respectful to the server
- **File Size**: PDF files can be 5-20MB each, plan storage accordingly
- **Legality**: Only download content you have rights to access
- **Updates**: Website structure may change; script may need updates

## 🎯 Tips

1. **Start Small**: Try downloading 2-3 PDFs first to verify it works
2. **Check Storage**: Ensure you have enough disk space (20-30MB per PDF)
3. **Regular Cleanup**: Delete old PDFs you no longer need
4. **Backup Metadata**: Keep a backup of `downloaded_files.json` to preserve history

## 🔄 Example Workflow

```powershell
# 1. Navigate to scripts
cd F:\DL-Project\newspaper\scripts

# 2. Run downloader
F:/anaconda/envs/DL-Project/python.exe economic_times_downloader.py

# 3. Choose number of files (e.g., 5)

# 4. Wait for downloads to complete

# 5. Check the data folder for PDFs
cd ../data/economic_times_pdfs
dir
```

## 📚 Related Scripts

- `news_getter.py` - Fetches text-based financial news via API
- `preview_news.py` - Previews downloaded news data

## 🤝 Integration

This downloader can be integrated with:
- Market prediction analysis (version-2)
- Sentiment analysis pipelines
- News summarization tools
- Financial data aggregation systems

## ⚡ Quick Start Command

```powershell
cd F:\DL-Project\newspaper\scripts && F:/anaconda/envs/DL-Project/python.exe economic_times_downloader.py
```

---

**Last Updated:** October 26, 2025  
**Version:** 1.0  
**Python:** 3.13+
