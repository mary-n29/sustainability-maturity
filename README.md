# Sustainability Maturity Analysis Pipeline

A specialized Python pipeline for extracting, cleaning, and analyzing sustainability reports to classify corporate sustainability maturity using the **Gabler et al. (2023) Hierarchy of Sustainability Strategies**. Perfect for research on corporate sustainability, ESG analysis, and sustainability maturity assessment.

## 🌟 Features

- **Maturity Classification** - Automatically classifies text by sustainability strategy level (Regulatory → Cost Reduction → Value Proposition → Identification)
- **Smart Text Preservation** - Keeps metrics, years, targets, and commitments critical for maturity analysis
- **Multi-Dimensional Analysis** - Tags content by ESG dimensions (environmental, social, governance, economic)
- **Sentence-Based Chunking** - Creates contextual chunks with overlap for better analysis
- **Password-Protected PDFs** - Handles encrypted documents automatically
- **Multiple Export Formats** - Outputs to CSV, Excel, and manifest files
- **Self-Healing Dependencies** - Auto-installs required packages
- **Team-Friendly Code** - Well-documented classes for collaboration

## 📋 Table of Contents

- [Research Framework](#research-framework)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Input Format](#input-format)
- [Output Files](#output-files)
- [Maturity Classification](#maturity-classification)
- [Usage Examples](#usage-examples)
- [Class Documentation](#class-documentation)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

## 📚 Research Framework

This pipeline implements the **Hierarchy of Sustainability Strategies** from:

> Gabler, C.B., Landers, V.M., & Itani, O.S. (2023). Sustainability and professional sales: a review and future research agenda. *Journal of Personal Selling & Sales Management*.

### The Four-Level Hierarchy

```
┌─────────────────────────────────────────┐
│  Level 4: IDENTIFICATION STRATEGY       │
│  Core identity aligned with             │
│  sustainability (e.g., Patagonia)       │
├─────────────────────────────────────────┤
│  Level 3: VALUE PROPOSITION STRATEGY    │
│  Sustainability as customer value       │
│  (e.g., Nestle CSV)                     │
├─────────────────────────────────────────┤
│  Level 2: COST REDUCTION STRATEGY       │
│  Efficiency & long-term investment      │
│  (e.g., Nike, Google green energy)      │
├─────────────────────────────────────────┤
│  Level 1: REGULATORY STRATEGY           │
│  Compliance-driven to avoid penalties   │
│  (e.g., Unilever Sustainable Living)    │
└─────────────────────────────────────────┘
```

The pipeline automatically detects language patterns indicating each strategy level to classify corporate sustainability maturity.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sustainability-maturity-pipeline.git
cd sustainability-maturity-pipeline
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
Apple,2025,Environmental Report,Apple_Environmental_Progress_Report_2025.pdf
Nike,2024,Sustainability Data,Nike_Sustainability-Data.pdf
Patagonia,2024,B Corp Report,Patagonia_2023-2024-BCorp-Report.pdf
```

### Step 2: Configure the Script

Open the script and edit the configuration section:

```python
# CONFIGURATION SECTION
METADATA_CSV = "company_metadata.csv"
PDF_ROOT_FOLDER = "."
CHUNK_SIZE = 5  # Larger chunks for better context
OVERLAP_SENTENCES = 1  # Overlap for continuity
MIN_WORDS = 20  # Filter small chunks
PRESERVE_METRICS = True  # Keep numbers, years, targets
```

### Step 3: Run the Script

**In your IDE:** Press Run (F5) or click the Run button

**Or from terminal:**
```bash
python TextCleaning_and_Chunking_Sustainability.py
```

### Step 4: Analyze Your Results

Check the output files for:
- Maturity level classifications
- ESG dimension tagging
- Sustainability metrics
- Full text and chunks

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `METADATA_CSV` | `"company_metadata.csv"` | Path to metadata file |
| `PDF_ROOT_FOLDER` | `"."` | Directory with PDFs |
| `OUTPUT_DOCS_CSV` | `"cleaned_documents.csv"` | Document output path |
| `OUTPUT_CHUNKS_CSV` | `"cleaned_chunks.csv"` | Chunks output path |
| `CHUNK_SIZE` | `5` | Sentences per chunk (5-7 recommended for sustainability) |
| `MIN_WORDS` | `20` | Minimum words per chunk |
| `MAX_CHARS` | `100000` | Max chars to process at once |
| `SPACY_MODEL` | `"en_core_web_sm"` | spaCy model for sentence detection |
| `LOWERCASE` | `True` | Convert text to lowercase |
| `PRESERVE_METRICS` | `True` | Keep numbers, percentages, years |
| `PRESERVE_SECTION_HEADERS` | `True` | Identify section headers |
| `OVERLAP_SENTENCES` | `1` | Sentences overlap between chunks |

## 📄 Input Format

### Metadata CSV Structure

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `Company` | Yes | Organization name | `"Tesla"` |
| `year` | Yes | Report year | `2024` |
| `doc_type` | Yes | Document type | `"Impact Report"` |
| `file_name` | Yes | PDF filename | `"tesla_2024.pdf"` |
| `pdf_password` | No | Password for encrypted PDFs | `"secret123"` |

## 📦 Output Files

### 1. `cleaned_chunks.csv` - **Primary Analysis File**
Enhanced chunks with maturity classification:

**Standard Columns:**
- `company`, `year`, `doc_type`, `file_name`, `chunk_id`
- `text_chunk` - Chunked text
- `word_count`, `char_count` - Text statistics

**Sustainability Features:**
- `hsbs_dimensions` - ESG dimensions (e.g., "environmental,social")
- `has_metrics` - Contains quantitative data (True/False)
- `has_years` - Contains year references (True/False)
- `sustainability_score` - Count of sustainability keywords
- `top_keywords` - Top 5 sustainability terms found

**Maturity Classification (NEW):**
- `maturity_level` - Numeric level (0-4)
- `maturity_label` - Text label ("regulatory", "cost_reduction", etc.)
- `regulatory_score` - Count of regulatory indicators
- `cost_reduction_score` - Count of cost reduction indicators
- `value_prop_score` - Count of value proposition indicators
- `identification_score` - Count of identification indicators

### 2. `cleaned_documents.csv`
Full document-level data with cleaned text and statistics

### 3. `cleaned_documents.xlsx` & `cleaned_chunks.xlsx`
Excel versions for easy viewing in spreadsheet software

### 4. `cleaned_documents_manifest.csv`
Metadata only (no full text) for quick reference

## 🎯 Maturity Classification

### How It Works

The pipeline detects language patterns for each strategy level:

**Level 1 - Regulatory Strategy**
- Keywords: comply, compliance, regulation, mandate, penalty, minimum, legal
- Example: "We comply with environmental regulations to avoid sanctions"

**Level 2 - Cost Reduction Strategy**
- Keywords: efficiency, optimize, cost saving, ROI, investment, reduce waste
- Example: "Our energy efficiency initiatives reduced operational costs by 15%"

**Level 3 - Value Proposition Strategy**
- Keywords: customer value, competitive advantage, differentiation, brand, market
- Example: "Customers choose us for our commitment to sustainable packaging"

**Level 4 - Identification Strategy**
- Keywords: core identity, mission, purpose, leadership, transformation, who we are
- Example: "Sustainability is fundamental to who we are as an organization"

### Classification Methodology

1. Each chunk is scored across all four levels
2. The level with the highest score becomes the `dominant_level`
3. Individual scores allow for multi-level analysis
4. Companies can be classified by aggregating chunk-level data

## 💡 Usage Examples

### Example 1: Basic Company Classification

```python
import pandas as pd

# Load chunks from outputs folder
chunks = pd.read_csv('outputs/cleaned_chunks.csv')

# Classify companies by dominant maturity level
company_maturity = chunks.groupby('company')['maturity_level'].agg(
    lambda x: x.mode()[0] if not x.mode().empty else 0
)

print(company_maturity.sort_values(ascending=False))
```

### Example 2: Multi-Level Analysis

```python
# Load chunks
chunks = pd.read_csv('outputs/cleaned_chunks.csv')

# Aggregate maturity scores by company
maturity_profile = chunks.groupby('company').agg({
    'regulatory_score': 'sum',
    'cost_reduction_score': 'sum',
    'value_prop_score': 'sum',
    'identification_score': 'sum'
})

# Calculate percentages
maturity_profile['total'] = maturity_profile.sum(axis=1)
for col in ['regulatory', 'cost_reduction', 'value_prop', 'identification']:
    maturity_profile[f'{col}_pct'] = (
        maturity_profile[f'{col}_score'] / maturity_profile['total'] * 100
    )

print(maturity_profile)
```

### Example 3: Filter High-Quality Sustainability Content

```python
# Load data
chunks = pd.read_csv('outputs/cleaned_chunks.csv')

# Get chunks with strong sustainability signals
high_quality = chunks[
    (chunks['sustainability_score'] > 5) &  # Multiple keywords
    (chunks['has_metrics'] == True) &  # Quantitative data
    (chunks['maturity_level'] >= 3)  # Value prop or higher
]

print(f"Found {len(high_quality)} high-quality chunks")
```

### Example 4: Dimension-Specific Maturity

```python
# Load data
chunks = pd.read_csv('outputs/cleaned_chunks.csv')

# Environmental maturity
env_chunks = chunks[chunks['hsbs_dimensions'].str.contains('environmental')]
env_maturity = env_chunks.groupby('company')['maturity_level'].mean()

# Social maturity
social_chunks = chunks[chunks['hsbs_dimensions'].str.contains('social')]
social_maturity = social_chunks.groupby('company')['maturity_level'].mean()

comparison = pd.DataFrame({
    'environmental': env_maturity,
    'social': social_maturity
})
print(comparison)
```

### Example 5: Industry Benchmarking

```python
# Load data
chunks = pd.read_csv('outputs/cleaned_chunks.csv')

# Add industry classification to metadata
chunks['industry'] = chunks['company'].map({
    'Apple': 'Technology',
    'Nike': 'Apparel',
    'Tesla': 'Automotive',
    # ... add your companies
})

# Compare industry maturity
industry_maturity = chunks.groupby('industry').agg({
    'maturity_level': 'mean',
    'sustainability_score': 'mean',
    'has_metrics': lambda x: (x == True).sum() / len(x) * 100
})

print(industry_maturity.round(2))
```

## 📚 Class Documentation

### `DependencyManager`
Handles automatic installation of required packages
- `ensure_spacy_model(model_name)` - Download/load spaCy model
- `ensure_cryptography()` - Install cryptography for encrypted PDFs

### `SustainabilityTextCleaner`
Cleans text while preserving sustainability-relevant information
- `clean(text)` - Remove noise, keep metrics/years/targets
- `identify_sustainability_content(text)` - Detect sustainability features

### `PDFProcessor`
Handles PDF file operations
- `read_pdf_text(path, password)` - Extract text (supports encryption)
- `ensure_pdf_extension(filename)` - Normalize filenames

### `MaturityLevelDetector` ⭐ NEW
Detects sustainability maturity using Gabler et al. hierarchy
- `detect_maturity_level(text)` - Classify text by strategy level
- Returns scores for all four levels plus dominant level

### `SectionDetector`
Identifies ESG dimensions in text
- `detect_section_type(text)` - Tag by environmental/social/governance/economic

### `DataFrameUtils`
Utility functions for data operations
- `save_to_csv(df, path)` - Save with Excel compatibility
- `save_to_excel(df, path)` - Export to Excel
- `compute_text_statistics(df)` - Add word/character counts

### `SustainabilityChunker`
Creates contextually-aware chunks
- `create_chunks(df, text_column)` - Split with overlap and metadata

### `SustainabilityPipeline`
Main orchestrator coordinating all operations
- `run(output_docs_csv, output_chunks_csv)` - Execute pipeline

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
- Verify `company_metadata.csv` exists in the script directory
- Update `METADATA_CSV` configuration to the correct path

### "Missing file: [filename]"
- Check PDFs are in `PDF_ROOT_FOLDER` directory
- Verify filenames in CSV match actual files (case-sensitive on Linux/Mac)

### "PDF is encrypted and a password is required"
- Add `pdf_password` column to metadata CSV
- Enter password for that specific PDF

### Low maturity scores across all documents
- Check if reports are truly sustainability-focused
- Adjust `MIN_WORDS` if chunks are too small
- Verify PDFs contain extractable text (not scanned images)

### Memory issues with large PDFs
- Increase `MAX_CHARS` if chunks cut off mid-sentence
- Decrease `MAX_CHARS` if running out of memory
- Process fewer files at once

### Unexpected maturity classifications
- Review the maturity indicator patterns in `MaturityLevelDetector`
- Check if company uses non-standard terminology
- Consider that companies may have mixed strategies

## 📊 Output Summary Example

When you run the pipeline, you'll see:

```
============================================================
SUSTAINABILITY PDF TEXT PROCESSING PIPELINE
Optimized for Maturity Analysis (Gabler et al. 2023)
============================================================
[INFO] Initializing sustainability-focused pipeline...
[INFO] Successfully processed 23 documents

[SUMMARY] Created 2,847 chunks from 23 documents
[SUMMARY] Average sustainability keywords per chunk: 4.3

[SUMMARY] HSBS dimension distribution:
  - Environmental: 1,892 chunks (66.5%)
  - Social: 1,245 chunks (43.7%)
  - Governance: 678 chunks (23.8%)
  - Economic: 534 chunks (18.8%)

[SUMMARY] Maturity Level Distribution (Gabler et al. Hierarchy):
  - Level Identification: 342 chunks (12.0%)
  - Level Value Proposition: 934 chunks (32.8%)
  - Level Cost Reduction: 1,016 chunks (35.7%)
  - Level Regulatory: 487 chunks (17.1%)
  - Level None Identified: 68 chunks (2.4%)

[DONE] Sustainability processing complete!
============================================================
```

## 🤝 Contributing

Contributions welcome! For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add maturity indicators'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📖 Citation

If you use this pipeline in your research, please cite:

**The Framework:**
```
Gabler, C.B., Landers, V.M., & Itani, O.S. (2023). Sustainability and 
professional sales: a review and future research agenda. Journal of 
Personal Selling & Sales Management.
```

**This Tool:**
```
[Your Name]. (2025). Sustainability Maturity Analysis Pipeline. 
GitHub repository: https://github.com/yourusername/repo-name
```

## 👥 Authors

- Your Name - [Your GitHub](https://github.com/yourusername)
- Team Members - (add your team here)

## 🙏 Acknowledgments

- Framework based on Gabler, Landers, & Itani (2023)
- Built with [spaCy](https://spacy.io/) for NLP
- PDF extraction via [pypdf](https://github.com/py-pdf/pypdf)
- Designed for sustainability research and ESG analysis

## 📮 Contact

Questions or support? Open an issue on GitHub or contact [your-email@example.com]

---

**⭐ Star this repository if you find it helpful for your sustainability research!**
