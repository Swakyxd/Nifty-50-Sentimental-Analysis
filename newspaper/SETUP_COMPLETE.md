# Economic Times PDF Downloader - Setup Complete! 

## ✅ What Was Created

### 📁 File Structure
```
F:\DL-Project\newspaper\
├── scripts\
│   ├── economic_times_downloader.py  # Main downloader (350+ lines)
│   └── test_downloader.py            # Test/demo script
├── data\
│   └── economic_times_pdfs\          # PDF storage (auto-created)
├── README.md                          # Complete documentation
└── requirements.txt                   # Package dependencies
```

## 🎯 Key Features Implemented

### ✅ Core Functionality
- **Web Scraping**: Fetches PDF links from epaperwave.com
- **Smart Downloads**: Automatically downloads Economic Times PDFs
- **Duplicate Prevention**: 
  - Tracks all downloads in `downloaded_files.json`
  - Checks both URL and filename before downloading
  - Calculates MD5 hash for verification
- **Progress Tracking**: Real-time download progress with file size
- **Error Handling**: Automatic retry (3 attempts) on failures
- **Metadata Management**: Complete record of all downloads

### ✅ Smart Organization
- **Date Extraction**: Automatically extracts dates from PDF titles
- **Organized Naming**: `ET_YYYY-MM-DD_Title.pdf` format
- **Batch Downloads**: Download multiple PDFs in one run
- **Rate Limiting**: 2-second delay between downloads (server-friendly)

## 🚀 Quick Start Guide

### 1. Install Dependencies (One-time)
```powershell
cd F:\DL-Project\newspaper
F:/anaconda/envs/DL-Project/python.exe -m pip install -r requirements.txt
```

### 2. Test the Downloader
```powershell
cd scripts
F:/anaconda/envs/DL-Project/python.exe test_downloader.py
```

This will:
- Show you available PDFs
- Let you test download 2 files
- Verify everything works

### 3. Run Full Downloader
```powershell
cd scripts
F:/anaconda/envs/DL-Project/python.exe economic_times_downloader.py
```

Follow the interactive prompts:
- Shows previously downloaded files
- Asks how many PDFs to download (default: 5)
- Downloads and saves to `data/economic_times_pdfs/`

## 📊 Usage Examples

### Example 1: First Time Use
```powershell
PS> python economic_times_downloader.py
📚 Found 0 previously downloaded files

============================================================
How many PDFs to download? (default 5): 3

🌐 Fetching page: https://epaperwave.com/...
📄 Found 15 potential PDF links

[1/3] 📥 Downloading: Economic Times - Delhi Edition
   Progress: 100% (5.23MB / 5.23MB)
   ✅ Downloaded successfully

[2/3] 📥 Downloading: Economic Times - Mumbai Edition
   Progress: 100% (4.87MB / 4.87MB)
   ✅ Downloaded successfully

============================================================
DOWNLOAD SUMMARY
============================================================
✅ Successfully downloaded: 2
⏭️ Skipped (duplicates): 0
❌ Failed: 1
```

### Example 2: Subsequent Use (with duplicates)
```powershell
PS> python economic_times_downloader.py
📚 Found 2 previously downloaded files
Show list? (y/n): y

1. ET_2025-10-26_Delhi_Edition.pdf
   Size: 5.23 MB
   Downloaded: 2025-10-26T14:30:00

[Downloading...]
  ⏭️ Skipping duplicate: ET_2025-10-26_Delhi_Edition.pdf
```

## 🔍 What Happens During Download

1. **Fetch Page**: Scrapes epaperwave.com for PDF links
2. **Extract Links**: Finds all download links and titles
3. **Check Duplicates**: Compares against metadata
4. **Download**: Streams PDF with progress bar
5. **Verify**: Checks content-type is PDF
6. **Save**: Stores with organized filename
7. **Record**: Updates metadata JSON file

## 📁 Output Structure

### Downloaded PDFs
```
data/economic_times_pdfs/
├── ET_2025-10-26_Delhi_Edition.pdf
├── ET_2025-10-25_Mumbai_Edition.pdf
├── ET_2025-10-24_Bangalore_Edition.pdf
└── ...
```

### Metadata File
```json
{
  "https://example.com/pdf/et-2025-10-26.pdf": {
    "filename": "ET_2025-10-26_Delhi_Edition.pdf",
    "title": "Economic Times Delhi Edition",
    "download_date": "2025-10-26T14:30:00",
    "file_size": 5484032,
    "hash": "a1b2c3d4e5f6..."
  }
}
```

## 🛠️ Advanced Usage

### Custom Configuration
```python
from economic_times_downloader import EconomicTimesDownloader

# Use custom directory
downloader = EconomicTimesDownloader(data_dir="/custom/path")

# Download specific number
downloader.download_all(max_files=10)

# List downloaded files
downloader.list_downloaded_files()
```

### Integration with Other Scripts
```python
# In your analysis script
from scripts.economic_times_downloader import EconomicTimesDownloader

# Download latest PDFs
downloader = EconomicTimesDownloader()
downloader.download_all(max_files=5)

# Process PDFs for sentiment analysis
# ... your code here ...
```

## 🔧 Configuration Options

Edit `economic_times_downloader.py` to customize:

```python
# Change URL
self.base_url = "https://your-custom-url.com"

# Adjust retry attempts
def download_pdf(self, pdf_info, retry=5):  # Default: 3

# Modify download delay
time.sleep(5)  # Default: 2 seconds

# Change timeout
response = requests.get(url, timeout=120)  # Default: 60
```

## 📊 Statistics & Tracking

The script tracks:
- ✅ **Total downloads**: How many files downloaded successfully
- ⏭️ **Duplicates skipped**: Files already downloaded
- ❌ **Failed downloads**: Downloads that failed after retries
- 📊 **File sizes**: Individual and total size
- 🕐 **Download dates**: When each file was downloaded
- 🔐 **File hashes**: MD5 hash for integrity verification

## ⚠️ Important Notes

1. **Rate Limiting**: 2-second delay between downloads (be respectful to servers)
2. **Storage**: Each PDF is typically 5-20MB, plan accordingly
3. **Network**: Requires stable internet connection
4. **Permissions**: Needs write access to `data/` directory
5. **Website Changes**: If website structure changes, script may need updates

## 🐛 Troubleshooting

### Problem: "No PDF links found"
**Solution**: Website structure may have changed. Check if URL is still valid.

### Problem: "Download failed" repeatedly
**Solutions**:
- Check internet connection
- Verify website is accessible in browser
- Increase retry attempts in code
- Check if server is rate-limiting you

### Problem: "Not a PDF file" warning
**Solution**: Link points to webpage, not PDF. Script will skip automatically.

### Problem: Permission error
**Solution**: Ensure write permissions to `data/` folder

## 🎯 Next Steps

1. **Test It**: Run `test_downloader.py` to verify setup
2. **Download PDFs**: Run main script to download newspapers
3. **Integrate**: Use with sentiment analysis or other tools
4. **Automate**: Schedule with Task Scheduler for daily downloads
5. **Analyze**: Process PDFs for market insights

## 🔄 Maintenance

### Regular Tasks
- Clean up old PDFs to save space
- Backup `downloaded_files.json` periodically
- Update script if website structure changes
- Monitor disk space usage

### Automation (Optional)
Create a Windows Task Scheduler job:
```powershell
# Task name: Daily ET Download
# Trigger: Daily at 7:00 AM
# Action: F:/anaconda/envs/DL-Project/python.exe
# Arguments: F:\DL-Project\newspaper\scripts\economic_times_downloader.py
```

## 📚 Related Files

- `README.md` - Complete documentation
- `requirements.txt` - Package dependencies
- `test_downloader.py` - Test/demo script
- `economic_times_downloader.py` - Main downloader

## ✅ Verification

To verify everything is working:

```powershell
# 1. Test imports
cd F:\DL-Project\newspaper\scripts
F:/anaconda/envs/DL-Project/python.exe -c "from economic_times_downloader import EconomicTimesDownloader; print('✅ OK')"

# 2. Run test
F:/anaconda/envs/DL-Project/python.exe test_downloader.py

# 3. Check output directory
dir ..\data\economic_times_pdfs
```

---

## 🎉 Ready to Use!

Your Economic Times PDF downloader is fully set up and ready to download newspaper PDFs from epaperwave.com!

**Quick Command:**
```powershell
cd F:\DL-Project\newspaper\scripts && F:/anaconda/envs/DL-Project/python.exe economic_times_downloader.py
```

---

**Created:** October 26, 2025  
**Location:** F:\DL-Project\newspaper\  
**Status:** ✅ Ready for Use
