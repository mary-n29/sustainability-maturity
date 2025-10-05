# PDF Text Processing Pipeline

A robust Python pipeline for extracting, cleaning, and chunking text from PDF documents. Perfect for preparing documents for NLP tasks, embeddings, RAG systems, or text analysis.

## 🌟 Features

- **Automated PDF Processing** - Batch process multiple PDFs with metadata tracking
- **Smart Text Cleaning** - Remove URLs, emails, dates, phone numbers, and other noise
- **Password Support** - Handle encrypted/password-protected PDFs automatically
- **Sentence-Based Chunking** - Split documents into meaningful chunks using spaCy
- **Multiple Output Formats** - Export to CSV, Excel, and manifest files
- **Self-Healing Dependencies** - Automatically installs required packages
- **Team-Friendly Code** - Well-documented classes for easy collaboration

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Input Format](#input-format)
- [Output Files](#output-files)
- [Usage Examples](#usage-examples)
- [Class Documentation](#class-documentation)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-text-processor.git
cd pdf-text-processor
```

### 2. Install Required Packages

```bash
pip install pandas pypdf spacy xlsxwriter
```

### 3. Download spaCy Model

The script will download this automatically, but you can install manually:

```bash
python -m spacy download en_core_web_sm
```

## ⚡ Quick Start

### Step 1: Prepare Your Metadata CSV

Create a file named `company_metadata.csv`:

```csv
Company,year,doc_type,file_name
Apple,2023,Annual Report,apple_2023_report.pdf
Microsoft,2023,Annual Report,msft_2023_report.pdf
Google,2022,10-K,google_10k_2022.pdf
```

### Step 2: Configure the Script

Open `TextCleaning_and_Chunking_Refactored.py` and edit the configuration section:

```python
# CONFIGURATION SECTION
METADATA_CSV = "company_metadata.csv"
PDF_ROOT_FOLDER = "."  # or "data/pdfs" if PDFs are in a subfolder
CHUNK_SIZE = 3
MIN_WORDS = 10
```

### Step 3: Run the Script

**In your IDE:** Simply press Run (F5) or click the Run button

**Or from terminal:**
```bash
python TextCleaning_and_Chunking_Refactored.py
```

### Step 4: Check Your Outputs

The script creates several output files:
- `cleaned_documents.csv` - Full document text with metadata
- `cleaned_chunks.csv` - Sentence-based chunks
- `cleaned_documents.xlsx` - Excel version of documents
- `cleaned_chunks.xlsx` - Excel version of chunks
- `cleaned_documents_manifest.csv` - Metadata only (no full text)

## ⚙️ Configuration

Edit these variables at the top of the script:

| Setting | Default | Description |
|---------|---------|-------------|
| `METADATA_CSV` | `"company_metadata.csv"` | Path to your metadata CSV file |
| `PDF_ROOT_FOLDER` | `"."` | Directory containing PDF files |
| `OUTPUT_DOCS_CSV` | `"cleaned_documents.csv"` | Output path for cleaned documents |
| `OUTPUT_CHUNKS_CSV` | `"cleaned_chunks.csv"` | Output path for text chunks |
| `CHUNK_SIZE` | `3` | Number of sentences per chunk |
| `MIN_WORDS` | `10` | Minimum words per chunk (filters out small chunks) |
| `MAX_CHARS` | `100000` | Max characters to process at once (prevents memory issues) |
| `SPACY_MODEL` | `"en_core_web_sm"` | spaCy model for sentence detection |
| `LOWERCASE` | `True` | Convert text to lowercase during cleaning |

## 📄 Input Format

### Metadata CSV Structure

Your metadata CSV should have these columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `Company` | Yes | Company/organization name | `"Apple"` |
| `year` | Yes | Year of the document | `2023` |
| `doc_type` | Yes | Type of document | `"Annual Report"`, `"10-K"` |
| `file_name` | Yes | PDF filename (with or without .pdf extension) | `"apple_2023.pdf"` |
| `pdf_password` | No | Password for encrypted PDFs | `"secret123"` |

**Example:**
```csv
Company,year,doc_type,file_name,pdf_password
Tesla,2023,Annual Report,tesla_annual_2023.pdf,
SpaceX,2022,Financial Report,spacex_financials.pdf,confidential123
```

## 📦 Output Files

### 1. `cleaned_documents.csv`
Full document-level data with cleaned text:
- `company` - Company name
- `year` - Document year
- `doc_type` - Document type
- `file_name` - PDF filename
- `clean_text` - Cleaned full text
- `word_count_doc` - Total words in document
- `char_count_doc` - Total characters in document

### 2. `cleaned_chunks.csv`
Sentence-based chunks for NLP processing:
- `document_id` - ID linking back to parent document
- `chunk_id` - Sequential chunk number within document
- `text_chunk` - Chunked text (N sentences)
- `word_count` - Words in this chunk
- `char_count` - Characters in this chunk
- All metadata columns from parent document

### 3. `cleaned_documents.xlsx` & `cleaned_chunks.xlsx`
Excel versions of the above for easy viewing

### 4. `cleaned_documents_manifest.csv`
Metadata only (no full text) for quick reference

## 💡 Usage Examples

### Example 1: Basic Usage

```python
# Edit configuration section
METADATA_CSV = "my_documents.csv"
PDF_ROOT_FOLDER = "data/pdfs"

# Run the script - that's it!
```

### Example 2: Large Chunks for Summarization

```python
CHUNK_SIZE = 10  # 10 sentences per chunk (larger chunks)
MIN_WORDS = 50   # Only keep substantial chunks
```

### Example 3: Processing Documents in Different Folder

```python
METADATA_CSV = "metadata/docs_list.csv"
PDF_ROOT_FOLDER = "raw_pdfs/company_reports"
OUTPUT_DOCS_CSV = "output/processed_docs.csv"
OUTPUT_CHUNKS_CSV = "output/processed_chunks.csv"
```

### Example 4: Programmatic Usage

You can also use the classes directly in your own scripts:

```python
from TextCleaning_and_Chunking_Refactored import PDFPipeline

# Create pipeline
pipeline = PDFPipeline(
    metadata_csv="data.csv",
    pdf_root="pdfs/",
    chunk_size=5,
    min_words=20
)

# Run pipeline
pipeline.run(
    output_docs_csv="results/documents.csv",
    output_chunks_csv="results/chunks.csv"
)
```

### Example 5: Using Individual Classes

```python
from TextCleaning_and_Chunking_Refactored import TextCleaner, PDFProcessor

# Clean some text
cleaner = TextCleaner(lowercase=True)
clean_text = cleaner.clean("Visit www.example.com on 2023-01-15")
print(clean_text)  # Output: "visit on"

# Read a PDF
pdf_text = PDFProcessor.read_pdf_text("document.pdf")
```

## 📚 Class Documentation

### `DependencyManager`
Handles automatic installation of required packages
- `ensure_spacy_model(model_name)` - Download/load spaCy model
- `ensure_cryptography()` - Install cryptography for encrypted PDFs

### `TextCleaner`
Cleans and normalizes text data
- `clean(text)` - Remove URLs, emails, dates, phone numbers, normalize whitespace

### `PDFProcessor`
Handles PDF file operations
- `read_pdf_text(path, password)` - Extract text from PDF (supports encryption)
- `ensure_pdf_extension(filename)` - Normalize filenames

### `DataFrameUtils`
Utility functions for data operations
- `save_to_csv(df, path)` - Save with Excel-compatible formatting
- `save_to_excel(df, path)` - Export to Excel workbook
- `compute_text_statistics(df)` - Add word/character counts

### `TextChunker`
Creates sentence-based chunks from documents
- `create_chunks(df, text_column)` - Split documents into chunks using spaCy

### `PDFPipeline`
Main orchestrator that coordinates all operations
- `run(output_docs_csv, output_chunks_csv)` - Execute complete pipeline

## 📦 Requirements

- **Python 3.7+**
- **pandas** - Data manipulation
- **pypdf** - PDF text extraction
- **spacy** - Sentence segmentation
- **xlsxwriter** - Excel export
- **cryptography** (auto-installed) - Encrypted PDF support

Install all at once:
```bash
pip install pandas pypdf spacy xlsxwriter
python -m spacy download en_core_web_sm
```

## 🔧 Troubleshooting

### "Metadata file not found"
- Check that `company_metadata.csv` exists in the same directory as the script
- Or update `METADATA_CSV` to point to the correct path

### "Missing file: [filename]"
- Verify PDF files are in the `PDF_ROOT_FOLDER` directory
- Check that filenames in metadata CSV match actual file names
- File names are case-sensitive on Linux/Mac

### "PDF is encrypted and a password is required"
- Add a `pdf_password` column to your metadata CSV
- Enter the password for that specific PDF

### Memory issues with large PDFs
- Increase `MAX_CHARS` if chunks are cut off mid-sentence
- Decrease `MAX_CHARS` if you're running out of memory
- Consider processing fewer files at once

### Empty chunks output
- Check that `MIN_WORDS` isn't set too high
- Verify PDFs contain extractable text (not scanned images)
- Some PDFs may have encoding issues - try opening them manually first

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - [Your GitHub](https://github.com/yourusername)
- Team Members - (add your team members here)

## 🙏 Acknowledgments

- Built with [spaCy](https://spacy.io/) for NLP processing
- PDF extraction powered by [pypdf](https://github.com/py-pdf/pypdf)
- Inspired by the need for clean, structured text data for NLP applications

## 📮 Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com]

---

**Star ⭐ this repository if you find it helpful!**
