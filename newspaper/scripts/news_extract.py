"""
Extract news related to Nifty 50 and its constituent companies from Economic Times PDFs
"""

import os
import re
import csv
from pathlib import Path
from datetime import datetime
import io
try:
    import pypdf
    USE_PYPDF = True
except ImportError:
    import PyPDF2
    USE_PYPDF = False
from tqdm import tqdm

# Try to import PDF2Image and pytesseract for OCR
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Try to import pdfplumber as alternative extraction method
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Try to import PyMuPDF (fitz) as another alternative
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# List of Nifty 50 companies (as of October 2024)
NIFTY_50_COMPANIES = [
    "Adani Enterprises", "Adani Ports", "Apollo Hospitals", "Asian Paints", "Axis Bank",
    "Bajaj Auto", "Bajaj Finance", "Bajaj Finserv", "Bharti Airtel", "BPCL",
    "Britannia", "Cipla", "Coal India", "Divi's Lab", "Dr Reddy", "Eicher Motors",
    "Grasim", "HCL Tech", "HDFC Bank", "HDFC Life", "Hero MotoCorp", "Hindalco",
    "Hindustan Unilever", "ICICI Bank", "IndusInd Bank", "Infosys", "ITC",
    "JSW Steel", "Kotak Bank", "Larsen & Toubro", "LTI Mindtree", "Mahindra & Mahindra",
    "Maruti Suzuki", "Nestle", "NTPC", "ONGC", "Power Grid", "Reliance",
    "SBI", "SBI Life", "Shriram Finance", "Sun Pharma", "Tata Consumer", "Tata Motors",
    "Tata Steel", "TCS", "Tech Mahindra", "Titan", "UltraTech Cement", "Wipro"
]

# Additional keywords related to Nifty 50
NIFTY_KEYWORDS = [
    "nifty 50", "nifty50", "nifty index", "nse nifty", "sensex", "stock market",
    "bse", "nse", "market rally", "market crash", "bull market", "bear market",
    "stock prices", "market cap", "blue chip"
]

class NewsExtractor:
    """Extract Nifty 50 related news from Economic Times PDFs"""
    
    def __init__(self, pdfs_dir=None, output_dir=None, use_ocr=False):
        """
        Initialize the news extractor
        
        Args:
            pdfs_dir: Directory containing Economic Times PDFs (default: newspaper/data/economic_times_pdfs)
            output_dir: Directory to save the extracted data CSV (default: newspaper/data)
            use_ocr: Whether to use OCR for image-based PDFs (default: False)
        """
        # Get the script's directory
        script_dir = Path(__file__).parent
        
        # Default paths relative to script location
        if pdfs_dir is None:
            self.pdfs_dir = script_dir.parent / "data" / "economic_times_pdfs"
        else:
            self.pdfs_dir = Path(pdfs_dir)
            
        if output_dir is None:
            self.output_dir = script_dir.parent / "data"
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output CSV file
        self.output_file = self.output_dir / "nifty50_news_extracted.csv"
        
        # OCR settings
        self.use_ocr = use_ocr and OCR_AVAILABLE
        
        # Compile regex patterns for better performance
        self.company_patterns = self._compile_company_patterns()
        self.nifty_patterns = self._compile_nifty_patterns()
    
    def _print_available_methods(self):
        """Print available PDF extraction methods"""
        print("\n📚 Available Extraction Methods:")
        print(f"   ✅ pypdf: {'Yes' if USE_PYPDF else 'No'}")
        print(f"   ✅ PyPDF2: {'Yes' if not USE_PYPDF else 'No (using pypdf)'}")
        print(f"   ✅ PyMuPDF (fitz): {'Yes' if PYMUPDF_AVAILABLE else 'No'}")
        print(f"   ✅ pdfplumber: {'Yes' if PDFPLUMBER_AVAILABLE else 'No'}")
        print(f"   ✅ OCR (pytesseract): {'Yes' if OCR_AVAILABLE else 'No'}")
        if self.use_ocr:
            print(f"   🔍 OCR Mode: ENABLED")
        print()
    
    def _compile_company_patterns(self):
        """Compile regex patterns for company names"""
        patterns = []
        for company in NIFTY_50_COMPANIES:
            # Create flexible pattern that matches variations
            # e.g., "Reliance" matches "Reliance Industries", "RIL", etc.
            pattern = re.compile(r'\b' + re.escape(company) + r'\b', re.IGNORECASE)
            patterns.append((company, pattern))
        return patterns
    
    def _compile_nifty_patterns(self):
        """Compile regex patterns for Nifty keywords"""
        patterns = []
        for keyword in NIFTY_KEYWORDS:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            patterns.append((keyword, pattern))
        return patterns
    
    def extract_text_with_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF (fitz) - usually better quality"""
        try:
            text = ""
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            return ""
    
    def extract_text_with_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber - good for tables and structured data"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            return ""
    
    def extract_text_with_pypdf(self, pdf_path):
        """Extract text using pypdf/PyPDF2 - fallback method"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                if USE_PYPDF:
                    pdf_reader = pypdf.PdfReader(file)
                else:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        continue
            return text
        except Exception as e:
            return ""
    
    def extract_text_with_ocr(self, pdf_path, max_pages=10):
        """Extract text using OCR for image-based PDFs"""
        if not OCR_AVAILABLE:
            return ""
        
        try:
            text = ""
            # Convert PDF to images (limit pages for performance)
            images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)
            
            for i, image in enumerate(images):
                # Use pytesseract to extract text from image
                page_text = pytesseract.image_to_string(image)
                if page_text:
                    text += page_text + "\n"
            
            return text
        except Exception as e:
            return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file using multiple methods
        Tries multiple extraction strategies in order of quality/speed
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        extraction_method = "none"
        
        # Strategy 1: Try PyMuPDF (fitz) first - usually best quality
        if PYMUPDF_AVAILABLE:
            text = self.extract_text_with_pymupdf(pdf_path)
            if len(text.strip()) > 100:  # Meaningful text found
                extraction_method = "PyMuPDF"
        
        # Strategy 2: Try pdfplumber if PyMuPDF failed
        if not text and PDFPLUMBER_AVAILABLE:
            text = self.extract_text_with_pdfplumber(pdf_path)
            if len(text.strip()) > 100:
                extraction_method = "pdfplumber"
        
        # Strategy 3: Try pypdf/PyPDF2 as fallback
        if not text:
            text = self.extract_text_with_pypdf(pdf_path)
            if len(text.strip()) > 100:
                extraction_method = "pypdf"
        
        # Strategy 4: Try OCR if enabled and all text extraction failed
        if not text and self.use_ocr:
            text = self.extract_text_with_ocr(pdf_path)
            if len(text.strip()) > 100:
                extraction_method = "OCR"
        
        if not text:
            return ""
        
        # Post-processing: Fix common issues
        # Fix spaced text issue (e.g., "J o i n" -> "Join")
        text = re.sub(r'(?<=\w)\s(?=\w(?:\s|$))', '', text)
        
        # Remove excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def extract_date_from_filename(self, filename):
        """
        Extract date from filename like 'ET_2025-10-26.pdf'
        
        Args:
            filename: PDF filename
            
        Returns:
            Date string in YYYY-MM-DD format
        """
        # Pattern: ET_YYYY-MM-DD.pdf or ET_Free_YYYY-MM-DD.pdf
        pattern = r'(\d{4}-\d{2}-\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None
    
    def clean_text(self, text):
        """
        Clean extracted text by removing excessive whitespace and formatting issues
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with space
        text = re.sub(r'\n+', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def find_relevant_paragraphs(self, text, pdf_date, pdf_name):
        """
        Find paragraphs related to Nifty 50 and its companies
        
        Args:
            text: Full text from PDF
            pdf_date: Date of the newspaper
            pdf_name: Name of the PDF file
            
        Returns:
            List of relevant paragraphs with metadata
        """
        # Split text into paragraphs (sentences grouped together)
        paragraphs = text.split('\n\n')
        
        relevant_data = []
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:  # Skip very short paragraphs
                continue
            
            # Check if paragraph mentions any company
            mentioned_companies = []
            for company, pattern in self.company_patterns:
                if pattern.search(para):
                    mentioned_companies.append(company)
            
            # Check if paragraph mentions Nifty-related keywords
            mentioned_keywords = []
            for keyword, pattern in self.nifty_patterns:
                if pattern.search(para):
                    mentioned_keywords.append(keyword)
            
            # If relevant, add to results
            if mentioned_companies or mentioned_keywords:
                # Clean the text for better CSV readability
                cleaned_text = self.clean_text(para)
                
                # Skip if cleaned text is too short after cleaning
                if len(cleaned_text) < 100:
                    continue
                
                relevant_data.append({
                    'date': pdf_date,
                    'source_file': pdf_name,
                    'companies_mentioned': '; '.join(mentioned_companies) if mentioned_companies else '',
                    'keywords_mentioned': '; '.join(mentioned_keywords) if mentioned_keywords else '',
                    'text': cleaned_text,
                    'text_length': len(cleaned_text),
                    'num_companies': len(mentioned_companies),
                    'num_keywords': len(mentioned_keywords),
                    'extracted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return relevant_data
    
    def process_all_pdfs(self, limit=None):
        """
        Process all PDFs in the directory
        
        Args:
            limit: Maximum number of PDFs to process (None = all)
        """
        print("="*60)
        print("NIFTY 50 NEWS EXTRACTOR")
        print("="*60)
        print(f"📁 PDF Directory: {self.pdfs_dir.absolute()}")
        print(f"💾 Output File: {self.output_file.absolute()}")
        print("="*60)
        
        # Get all PDF files
        pdf_files = sorted(list(self.pdfs_dir.glob("*.pdf")))
        
        if not pdf_files:
            print("❌ No PDF files found!")
            return
        
        if limit:
            pdf_files = pdf_files[:limit]
        
        print(f"\n📊 Found {len(pdf_files)} PDF files to process")
        
        # Prepare CSV file
        csv_headers = ['date', 'source_file', 'companies_mentioned', 'keywords_mentioned', 
                       'text', 'text_length', 'num_companies', 'num_keywords', 'extracted_at']
        
        all_extracted_data = []
        
        # Track statistics
        pdfs_with_data = 0
        pdfs_without_text = 0
        pdfs_with_errors = 0
        
        # Process each PDF
        for i, pdf_path in enumerate(tqdm(pdf_files, desc="Processing PDFs"), 1):
            # Extract date from filename
            pdf_date = self.extract_date_from_filename(pdf_path.name)
            if not pdf_date:
                print(f"\n⚠️ Could not extract date from {pdf_path.name}, skipping...")
                continue
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                pdfs_without_text += 1
                continue
            
            # Find relevant paragraphs
            relevant_paragraphs = self.find_relevant_paragraphs(text, pdf_date, pdf_path.name)
            
            if relevant_paragraphs:
                all_extracted_data.extend(relevant_paragraphs)
                pdfs_with_data += 1
                tqdm.write(f"✅ {pdf_path.name}: Found {len(relevant_paragraphs)} relevant paragraphs")
        
        # Save to CSV
        if all_extracted_data:
            print(f"\n💾 Saving {len(all_extracted_data)} entries to CSV...")
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(all_extracted_data)
            
            print(f"✅ Successfully saved to: {self.output_file.absolute()}")
        else:
            print("\n❌ No relevant data found to save!")
        
        # Summary
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"📄 PDFs Processed: {len(pdf_files)}")
        print(f"✅ PDFs with Nifty 50 content: {pdfs_with_data}")
        print(f"⚠️ PDFs without extractable text: {pdfs_without_text}")
        print(f"📝 Total Relevant Paragraphs Found: {len(all_extracted_data)}")
        print(f"💾 Output File: {self.output_file.absolute()}")
        print("="*60)
    
    def show_statistics(self):
        """Show statistics of extracted data"""
        if not self.output_file.exists():
            print("❌ No extracted data file found!")
            return
        
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        
        # Increase CSV field size limit
        import sys
        csv.field_size_limit(sys.maxsize)
        
        with open(self.output_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)
        
        print(f"📊 Total Entries: {len(data)}")
        
        # Count by company
        company_counts = {}
        for row in data:
            if row['companies_mentioned']:
                companies = row['companies_mentioned'].split('; ')
                for company in companies:
                    if company.strip():
                        company_counts[company] = company_counts.get(company, 0) + 1
        
        if company_counts:
            print(f"\n🏢 Top 10 Most Mentioned Companies:")
            for i, (company, count) in enumerate(sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:10], 1):
                print(f"   {i}. {company}: {count} mentions")
        
        # Count by keyword
        keyword_counts = {}
        for row in data:
            if row['keywords_mentioned']:
                keywords = row['keywords_mentioned'].split('; ')
                for keyword in keywords:
                    if keyword.strip():
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        if keyword_counts:
            print(f"\n🔑 Top Keywords:")
            for i, (keyword, count) in enumerate(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True), 1):
                print(f"   {i}. {keyword}: {count} mentions")
        
        print("="*60)


def main():
    """Main function"""
    print("="*60)
    print("NIFTY 50 NEWS EXTRACTOR - ENHANCED")
    print("="*60)
    
    # Check for OCR capability
    use_ocr = False
    if OCR_AVAILABLE:
        print("\n🔍 OCR capability detected!")
        ocr_response = input("Enable OCR for image-based PDFs? (y/n, default n): ").strip().lower()
        use_ocr = (ocr_response == 'y')
        if use_ocr:
            print("⚠️ Note: OCR is slower but can extract from image-based PDFs")
    
    extractor = NewsExtractor(use_ocr=use_ocr)
    extractor._print_available_methods()
    
    # Show existing downloads
    if extractor.output_file.exists():
        print(f"📚 Output file already exists: {extractor.output_file}")
        response = input("Overwrite existing file? (y/n, default n): ").strip().lower()
        if response == 'y':
            extractor.output_file.unlink()
            print("✅ Existing file deleted\n")
    
    print("="*60)
    response = input("Process ALL PDFs? (y/n, default y): ").strip().lower()
    
    if response == 'n':
        limit_str = input("How many PDFs to process? (enter number): ").strip()
        limit = int(limit_str) if limit_str.isdigit() else None
    else:
        limit = None
    
    # Process PDFs
    print("\n🚀 Starting extraction process...")
    print("📊 Using multi-strategy extraction for maximum data recovery\n")
    extractor.process_all_pdfs(limit=limit)
    
    # Show statistics
    if extractor.output_file.exists():
        print("\n" + "="*60)
        show_stats = input("Show statistics? (y/n, default y): ").strip().lower()
        if show_stats != 'n':
            extractor.show_statistics()
    
    print("\n✅ Extraction complete!")
    print(f"📄 CSV file saved at: {extractor.output_file.absolute()}")


if __name__ == "__main__":
    main()
