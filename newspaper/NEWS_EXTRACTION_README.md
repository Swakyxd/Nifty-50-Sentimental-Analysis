# Nifty 50 News Extraction Script

## Overview
This script extracts news articles and paragraphs related to Nifty 50 index and its constituent companies from Economic Times PDF newspapers.

## Features
- ✅ Extracts text from PDF files using `pypdf` library
- ✅ Identifies mentions of all 50 Nifty companies
- ✅ Detects Nifty 50 and stock market related keywords
- ✅ Saves extracted data to CSV format with metadata
- ✅ Provides statistics on most mentioned companies and keywords
- ✅ Handles PDF extraction errors gracefully

## Installation

### Required Packages
```bash
pip install pypdf tqdm
```

Or install using the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
Run the script from the `scripts` directory:
```bash
python news_extract.py
```

The script will:
1. Ask if you want to process ALL PDFs or limit the number
2. Extract relevant paragraphs mentioning Nifty 50 or its companies
3. Save results to `newspaper/data/nifty50_news_extracted.csv`
4. Display statistics about extracted data

### Testing with Limited PDFs
For testing purposes, you can process just a few PDFs:
```bash
python test_news_extract.py
```
This will process only 3 PDFs to verify the extraction is working correctly.

## Output Format

The script creates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `date` | Date of the newspaper (YYYY-MM-DD format) |
| `source_file` | Name of the PDF file |
| `companies_mentioned` | Comma-separated list of Nifty 50 companies mentioned |
| `keywords_mentioned` | Comma-separated list of relevant keywords found |
| `text` | Full text of the relevant paragraph/article |
| `extracted_at` | Timestamp when the extraction was performed |

### Example Output
```csv
date,source_file,companies_mentioned,keywords_mentioned,text,extracted_at
2025-08-02,ET_2025-08-02.pdf,"Reliance, HDFC Bank, Infosys","nifty 50, sensex, stock market","Market overview text...",2025-10-26 18:07:43
```

## Nifty 50 Companies Tracked

The script monitors mentions of all 50 companies in the Nifty index:

1. Adani Enterprises
2. Adani Ports
3. Apollo Hospitals
4. Asian Paints
5. Axis Bank
6. Bajaj Auto
7. Bajaj Finance
8. Bajaj Finserv
9. Bharti Airtel
10. BPCL
11. Britannia
12. Cipla
13. Coal India
14. Divi's Lab
15. Dr Reddy
16. Eicher Motors
17. Grasim
18. HCL Tech
19. HDFC Bank
20. HDFC Life
21. Hero MotoCorp
22. Hindalco
23. Hindustan Unilever
24. ICICI Bank
25. IndusInd Bank
26. Infosys
27. ITC
28. JSW Steel
29. Kotak Bank
30. Larsen & Toubro
31. LTI Mindtree
32. Mahindra & Mahindra
33. Maruti Suzuki
34. Nestle
35. NTPC
36. ONGC
37. Power Grid
38. Reliance
39. SBI
40. SBI Life
41. Shriram Finance
42. Sun Pharma
43. Tata Consumer
44. Tata Motors
45. Tata Steel
46. TCS
47. Tech Mahindra
48. Titan
49. UltraTech Cement
50. Wipro

## Keywords Tracked

The script also searches for general market-related terms:
- Nifty 50, Nifty50, Nifty Index
- NSE Nifty
- Sensex
- Stock Market
- BSE, NSE
- Market Rally, Market Crash
- Bull Market, Bear Market
- Stock Prices
- Market Cap
- Blue Chip

## How It Works

### 1. PDF Text Extraction
- Opens each PDF file in the `economic_times_pdfs` directory
- Extracts text from all pages using `pypdf` library
- Handles extraction errors gracefully (skips problematic pages)

### 2. Content Analysis
- Splits extracted text into paragraphs
- Searches each paragraph for company names using regex patterns
- Searches for market-related keywords
- Filters out very short paragraphs (< 50 characters)

### 3. Data Storage
- Saves relevant paragraphs with metadata to CSV
- Tracks which companies and keywords were mentioned
- Records source file and date for each entry

### 4. Statistics Generation
- Counts mentions per company
- Ranks companies by frequency
- Shows top keywords
- Displays processing summary

## File Structure

```
newspaper/
├── data/
│   ├── economic_times_pdfs/     # Input PDF files
│   │   ├── ET_2024-02-11.pdf
│   │   ├── ET_2024-02-12.pdf
│   │   └── ...
│   └── nifty50_news_extracted.csv   # Output CSV file
├── scripts/
│   ├── news_extract.py          # Main extraction script
│   ├── test_news_extract.py     # Test script (3 PDFs only)
│   └── view_csv_stats.py        # View CSV statistics
└── README.md                     # This file
```

## Performance

- Processes approximately 3-10 PDFs per minute (depending on PDF size and complexity)
- Each PDF typically contains 30-100 pages
- Extraction time varies based on:
  - PDF size and complexity
  - Number of images in PDF
  - Text density
  - System performance

## Error Handling

The script handles several types of errors:
- **Corrupted PDFs**: Skips problematic pages and continues
- **Missing dates**: Warns and skips files without valid dates
- **Empty PDFs**: Detects and skips PDFs with no extractable text
- **Large text fields**: Increases CSV field size limit automatically

## Customization

### Modify Company List
Edit the `NIFTY_50_COMPANIES` list in `news_extract.py`:
```python
NIFTY_50_COMPANIES = [
    "Your Company 1",
    "Your Company 2",
    # ... add more companies
]
```

### Add Custom Keywords
Edit the `NIFTY_KEYWORDS` list:
```python
NIFTY_KEYWORDS = [
    "your keyword 1",
    "your keyword 2",
    # ... add more keywords
]
```

### Change Output Location
Modify the `NewsExtractor` initialization:
```python
extractor = NewsExtractor(
    pdfs_dir="/path/to/pdfs",
    output_dir="/path/to/output"
)
```

## Troubleshooting

### Issue: "No PDF files found"
**Solution**: Ensure PDFs are in the correct directory (`newspaper/data/economic_times_pdfs/`)

### Issue: "No relevant data found"
**Solution**: The PDFs may not contain mentions of Nifty 50 companies. Try processing more PDFs or check if the company names match the list.

### Issue: CSV field size limit error
**Solution**: This is automatically handled, but if it persists, ensure you're using the latest version of the script.

### Issue: PDF extraction taking too long
**Solution**: Process PDFs in batches using the limit option, or upgrade your system resources.

## Sample Output Statistics

```
============================================================
EXTRACTION SUMMARY
============================================================
📄 PDFs Processed: 85
📝 Relevant Paragraphs Found: 1,247
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
   4. TCS: 115 mentions
   5. ICICI Bank: 98 mentions
   6. SBI: 87 mentions
   7. Bharti Airtel: 76 mentions
   8. ITC: 72 mentions
   9. Wipro: 65 mentions
   10. Tata Motors: 61 mentions

🔑 Top Keywords:
   1. stock market: 324 mentions
   2. nifty 50: 289 mentions
   3. sensex: 267 mentions
   4. market cap: 156 mentions
   5. nse: 142 mentions
============================================================
```

## Future Enhancements

Potential improvements for the script:
- [ ] Add sentiment analysis (positive/negative/neutral)
- [ ] Extract stock prices and percentage changes
- [ ] Generate daily/weekly/monthly summaries
- [ ] Create visualizations (charts, graphs)
- [ ] Add support for other newspaper formats
- [ ] Implement multi-threading for faster processing
- [ ] Add database storage option (SQLite/PostgreSQL)
- [ ] Generate automated reports (PDF/HTML)

## License

This script is part of the DL-Project for educational purposes.

## Contact

For issues or questions, please create an issue in the project repository.
