# Nifty 50 News Extraction - Simplified Guide

## Overview
The `news_extract.py` script extracts news related to Nifty 50 index and its 50 constituent companies from Economic Times PDF newspapers and saves the data to a CSV file.

## Quick Start

### Run the Extractor
```bash
cd f:\DL-Project\newspaper\scripts
python news_extract.py
```

The script will:
1. Ask if you want to process ALL PDFs (default: yes)
2. Extract relevant paragraphs mentioning Nifty 50 companies or market keywords
3. Save results to `f:\DL-Project\newspaper\data\nifty50_news_extracted.csv`
4. Show statistics about extracted data

## Features

### ✅ Smart Text Extraction
- Handles PDF text extraction with error recovery
- Fixes spaced text issues (e.g., "J o i n" → "Join")
- Cleans and normalizes text for better readability
- Removes excessive whitespace and line breaks

### ✅ Comprehensive Company Tracking
Monitors all 50 Nifty companies:
- Reliance, HDFC Bank, Infosys, TCS, ICICI Bank
- SBI, Bharti Airtel, ITC, Wipro, Tata Motors
- And 40 more...

### ✅ Keyword Detection
Identifies market-related terms:
- Nifty 50, Nifty Index, NSE Nifty
- Sensex, Stock Market
- BSE, NSE
- Market Cap, Bull/Bear Market

### ✅ Enhanced CSV Output
Saves data with these columns:
| Column | Description |
|--------|-------------|
| `date` | Newspaper date (YYYY-MM-DD) |
| `source_file` | PDF filename |
| `companies_mentioned` | Semicolon-separated company list |
| `keywords_mentioned` | Semicolon-separated keyword list |
| `text` | Cleaned paragraph text (readable format) |
| `text_length` | Character count |
| `num_companies` | Number of companies mentioned |
| `num_keywords` | Number of keywords found |
| `extracted_at` | Extraction timestamp |

## Output Format

The CSV file is **clean and readable** with:
- ✅ No embedded line breaks in cells
- ✅ Normalized whitespace
- ✅ Semicolon-separated lists (easier to parse than commas)
- ✅ Additional metadata columns for analysis

### Example Output
```csv
date,source_file,companies_mentioned,keywords_mentioned,text,text_length,num_companies,num_keywords,extracted_at
2025-07-31,ET_2025-07-31.pdf,"Reliance; HDFC Bank; Infosys","nifty 50; stock market; sensex","Market overview showing strong performance...",245,3,3,2025-10-26 20:15:30
```

## Processing Statistics

When processing completes, you'll see:
```
============================================================
EXTRACTION SUMMARY
============================================================
📄 PDFs Processed: 501
✅ PDFs with Nifty 50 content: 45
⚠️ PDFs without extractable text: 320
📝 Total Relevant Paragraphs Found: 1,247
💾 Output File: f:\DL-Project\newspaper\data\nifty50_news_extracted.csv
============================================================

============================================================
DATA STATISTICS
============================================================
📊 Total Entries: 1,247

🏢 Top 10 Most Mentioned Companies:
   1. Reliance: 156 mentions
   2. HDFC Bank: 142 mentions
   3. Infosys: 128 mentions
   ...

🔑 Top Keywords:
   1. stock market: 324 mentions
   2. nifty 50: 289 mentions
   ...
============================================================
```

## Processing Options

### Process All PDFs (Default)
```bash
python news_extract.py
# Press Enter at prompt to process all PDFs
```

### Process Limited Number
```bash
python news_extract.py
# Enter 'n' when asked
# Then enter number of PDFs to process (e.g., 50)
```

### Overwrite Existing Output
If CSV file already exists:
```
📚 Output file already exists: ...
Overwrite existing file? (y/n, default n):
```
- Enter `y` to delete and recreate
- Enter `n` to cancel and keep existing file

## File Locations

```
newspaper/
├── data/
│   ├── economic_times_pdfs/              # Input: PDF files (501 files)
│   │   ├── ET_2024-02-11.pdf
│   │   ├── ET_2025-07-31.pdf
│   │   └── ...
│   └── nifty50_news_extracted.csv        # Output: Extracted data
└── scripts/
    └── news_extract.py                   # Main extractor script
```

## Performance

- **Speed**: Processes 3-10 PDFs per minute
- **Time Estimate**: 
  - 100 PDFs = ~10-30 minutes
  - 501 PDFs = ~50-180 minutes
- **Memory**: Low (<500 MB)
- **CPU**: Single-threaded

### Why Some PDFs are Slow/Skipped
- **Image-based PDFs**: No extractable text (scanned newspapers)
- **Corrupted PDFs**: Incomplete downloads or damaged files
- **Large PDFs**: 30-100 pages with images take longer
- **Text Encoding Issues**: Some PDFs have non-standard encodings

## Troubleshooting

### Issue: "No text extracted from PDF"
**Cause**: PDF is image-based (scanned) rather than text-based

**Solution**: These PDFs would require OCR (Optical Character Recognition) to extract text. The current script skips them automatically.

### Issue: "Stream has ended unexpectedly"
**Cause**: Corrupted or incomplete PDF file

**Solution**: The script automatically skips these files and continues processing.

### Issue: CSV file is empty or has no data
**Cause**: No PDFs contain mentions of Nifty 50 companies/keywords

**Solution**: 
- Check that PDFs contain actual news content (not just ads)
- Verify PDFs are from Economic Times business newspaper
- Try processing more PDFs

### Issue: CSV fields appear garbled
**Cause**: Excel may misinterpret CSV encoding

**Solution**:
1. Open CSV in Excel using "Data → From Text/CSV" import
2. Specify UTF-8 encoding
3. Or use Python/Pandas to read the CSV properly

## Reading the CSV File

### Using Python
```python
import pandas as pd

# Read CSV
df = pd.read_csv('nifty50_news_extracted.csv')

# View first few rows
print(df.head())

# Filter by company
reliance_news = df[df['companies_mentioned'].str.contains('Reliance', na=False)]

# Filter by date
july_news = df[df['date'].str.startswith('2025-07')]

# Get statistics
print(f"Total articles: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
```

### Using Excel
1. Open Excel
2. Go to: Data → From Text/CSV
3. Select the CSV file
4. Set delimiter to comma
5. Set encoding to UTF-8
6. Load data

## Customization

### Add More Companies
Edit `NIFTY_50_COMPANIES` list in `news_extract.py`:
```python
NIFTY_50_COMPANIES = [
    "Your Company Name",
    # ... existing companies
]
```

### Add Custom Keywords
Edit `NIFTY_KEYWORDS` list:
```python
NIFTY_KEYWORDS = [
    "your keyword",
    # ... existing keywords
]
```

### Change Minimum Text Length
Edit in `find_relevant_paragraphs()` method:
```python
if len(para) < 50:  # Change this value
    continue
```

## Next Steps

After extraction, you can:
1. **Analyze trends**: Which companies are mentioned most?
2. **Sentiment analysis**: Are mentions positive or negative?
3. **Time series**: How do mentions change over time?
4. **Correlation**: Do mentions correlate with stock prices?
5. **Model training**: Use extracted text for ML models

## Dependencies

Required Python packages:
```
pypdf
tqdm
```

Install with:
```bash
pip install pypdf tqdm
```

## Support

For issues or questions:
1. Check this README first
2. Verify input PDFs are valid and readable
3. Check terminal output for specific error messages
4. Review the generated CSV file structure

---

**Last Updated**: October 26, 2025
**Script Version**: 2.0 (Simplified Single Extractor)
