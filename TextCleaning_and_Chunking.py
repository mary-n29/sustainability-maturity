# -*- coding: utf-8 -*-
"""
Sustainability-Focused PDF Text Processing Pipeline

Enhanced for HSBS (Horizontal Sustainability Balanced Scorecard) analysis.
Preserves critical information like metrics, dates, and sustainability keywords.
"""

import os
import re
import sys
import subprocess
import csv
import html
from typing import List, Dict, Optional, Set, Tuple

import pandas as pd
from pypdf import PdfReader
import spacy
from spacy.cli import download as spacy_download

# ============================================================
# CONFIGURATION SECTION - Edit these settings before running
# ============================================================

# INPUT SETTINGS
METADATA_CSV = "data/company_metadata.csv"
PDF_ROOT_FOLDER = "data"

# OUTPUT SETTINGS
OUTPUT_DOCS_CSV = "outputs/cleaned_documents.csv"
OUTPUT_CHUNKS_CSV = "outputs/cleaned_chunks.csv"

# PROCESSING SETTINGS
CHUNK_SIZE = 5  # Larger chunks for sustainability context (5-7 sentences recommended)
MIN_WORDS = 20  # Higher minimum for meaningful sustainability content
MAX_CHARS = 100_000
SPACY_MODEL = "en_core_web_sm"
LOWERCASE = True
PRESERVE_METRICS = True  # Keep numbers, percentages, years
PRESERVE_SECTION_HEADERS = True  # Identify and tag section headers
OVERLAP_SENTENCES = 1  # Overlap chunks by 1 sentence for context continuity


# ============================================================
# CLASS: DependencyManager
# ============================================================
class DependencyManager:
    """Handles automatic installation of required dependencies."""

    @staticmethod
    def ensure_spacy_model(model_name: str = "en_core_web_sm"):
        """Load spaCy model, downloading if needed."""
        try:
            return spacy.load(model_name)
        except Exception:
            print(f"[INFO] spaCy model '{model_name}' not found. Downloading…")
            spacy_download(model_name)
            return spacy.load(model_name)

    @staticmethod
    def ensure_cryptography():
        """Install cryptography library for encrypted PDFs."""
        try:
            import cryptography  # noqa: F401
        except Exception:
            print("[INFO] Installing 'cryptography' for encrypted PDFs…")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "cryptography>=3.4"
            ])


# ============================================================
# CLASS: SustainabilityTextCleaner
# ============================================================
class SustainabilityTextCleaner:
    """
    Specialized text cleaner for sustainability reports.

    Preserves critical information like:
    - Years and dates (for targets and commitments)
    - Numbers and percentages (for metrics)
    - Sustainability keywords
    - Section structure
    """

    # Sustainability-related keywords to identify important sections
    SUSTAINABILITY_KEYWORDS = {
        # Environmental
        'emissions', 'carbon', 'ghg', 'greenhouse', 'climate', 'renewable',
        'energy', 'waste', 'water', 'biodiversity', 'circular', 'recycling',
        'sustainability', 'environmental', 'pollution', 'footprint',

        # Social
        'diversity', 'equity', 'inclusion', 'dei', 'labor', 'human rights',
        'employee', 'workforce', 'safety', 'health', 'community', 'social',
        'wellbeing', 'training', 'engagement',

        # Governance
        'governance', 'ethics', 'compliance', 'transparency', 'accountability',
        'board', 'stakeholder', 'risk', 'esg', 'reporting', 'disclosure',

        # Metrics and commitments
        'target', 'goal', 'commitment', 'reduction', 'increase', 'improve',
        'achieve', 'initiative', 'program', 'strategy', 'framework', 'policy',
        'net zero', 'carbon neutral', 'science-based', 'kpi'
    }

    # Patterns for content to remove
    URL_PATTERN = r"(?i)\b(?:https?://|www\.)\S+"
    EMAIL_PATTERN = r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    PHONE_PATTERN = r"(?i)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?){2}\d{4}"

    def __init__(self, lowercase: bool = True, preserve_metrics: bool = True):
        """
        Initialize the sustainability-focused cleaner.

        Args:
            lowercase: Convert text to lowercase (default: True)
            preserve_metrics: Keep numbers, years, percentages (default: True)
        """
        self.lowercase = lowercase
        self.preserve_metrics = preserve_metrics

    def clean(self, text: str) -> str:
        """
        Clean text while preserving sustainability-relevant information.

        This method:
        1. Handles None/non-string inputs
        2. Unescapes HTML entities
        3. Removes URLs, emails, phone numbers (not relevant for analysis)
        4. PRESERVES years, dates, numbers, percentages (critical for metrics)
        5. Normalizes whitespace
        6. Optionally lowercases (while preserving numerical context)

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text with metrics preserved
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)

        # Decode HTML entities and normalize spaces
        text = html.unescape(text).replace("\u00A0", " ")

        # Remove URLs (not relevant for sustainability analysis)
        text = re.sub(self.URL_PATTERN, " ", text)

        # Remove email addresses
        text = re.sub(self.EMAIL_PATTERN, " ", text)

        # Remove phone numbers
        text = re.sub(self.PHONE_PATTERN, " ", text)

        # NOTE: Unlike generic cleaner, we PRESERVE:
        # - Years (e.g., 2050, 2030) - critical for targets
        # - Dates (e.g., 2023-01-15) - for commitments
        # - Numbers and percentages - for metrics
        # - Time expressions - for timelines

        # Clean up spacing around punctuation
        text = re.sub(r"\s,\s", ", ", text)
        text = re.sub(r"\s\.\s", ". ", text)

        # Normalize multiple spaces to single space
        text = re.sub(r"\s+", " ", text).strip()

        # Lowercase if specified (but metrics remain readable)
        if self.lowercase:
            text = text.lower()

        return text

    def identify_sustainability_content(self, text: str) -> Dict[str, any]:
        """
        Analyze text to identify sustainability-relevant features.

        Returns a dictionary with:
        - has_metrics: Boolean indicating presence of numbers/percentages
        - has_years: Boolean indicating presence of year references
        - sustainability_score: Count of sustainability keywords
        - keywords_found: Set of sustainability keywords present

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sustainability feature flags
        """
        text_lower = text.lower()

        # Check for metrics (numbers, percentages)
        has_numbers = bool(re.search(r'\b\d+', text))
        has_percentages = bool(re.search(r'\d+%|\bpercent\b', text_lower))

        # Check for years (1900-2100 range for historical and future targets)
        has_years = bool(re.search(r'\b(19|20)\d{2}\b', text))

        # Count sustainability keywords
        keywords_found = set()
        for keyword in self.SUSTAINABILITY_KEYWORDS:
            if keyword in text_lower:
                keywords_found.add(keyword)

        return {
            'has_metrics': has_numbers or has_percentages,
            'has_years': has_years,
            'sustainability_score': len(keywords_found),
            'keywords_found': keywords_found
        }


# ============================================================
# CLASS: PDFProcessor
# ============================================================
class PDFProcessor:
    """Handles PDF file operations including reading and password handling."""

    @staticmethod
    def ensure_pdf_extension(filename: str) -> str:
        """Ensure filename has proper .pdf extension."""
        if not isinstance(filename, str):
            return ""
        name = filename.strip()
        name = re.sub(r"(?i)\.pdf(\.pdf)+$", ".pdf", name)
        if not re.search(r"(?i)\.pdf$", name):
            name = f"{name}.pdf"
        return name

    @staticmethod
    def read_pdf_text(pdf_path: str, password: Optional[str] = None,
                      _retried: bool = False) -> str:
        """Extract all text from a PDF file."""
        try:
            reader = PdfReader(pdf_path)

            if getattr(reader, "is_encrypted", False):
                success = True
                try:
                    result = reader.decrypt(password or "")
                    success = bool(result)
                except Exception:
                    success = False

                if not success:
                    raise RuntimeError("PDF is encrypted and password is required.")

            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

            return " ".join(text_parts)

        except Exception as e:
            error_msg = str(e).lower()
            if ("cryptography" in error_msg or "aes" in error_msg) and not _retried:
                DependencyManager.ensure_cryptography()
                return PDFProcessor.read_pdf_text(pdf_path, password=password, _retried=True)
            raise


class MaturityLevelDetector:
    """
    Detects sustainability maturity level based on the Gabler et al. hierarchy.

    Four levels (bottom to top):
    1. Regulatory - Compliance-driven
    2. Cost Reduction - Efficiency-focused
    3. Value Proposition - Customer-value driven
    4. Identification - Identity-aligned
    """

    # Level 1: Regulatory Strategy indicators
    REGULATORY_PATTERNS = {
        'compliance': r'\b(comply|compliance|regulation|regulatory|mandate|requirement|standard|law|legal|required)\b',
        'avoid_penalties': r'\b(penalty|penalties|fine|fines|sanction|violation|enforce)\b',
        'minimum': r'\b(minimum|baseline|basic|fundamental|essential|mandatory)\b',
        'response': r'\b(respond|response|react|address|meet requirement)\b'
    }

    # Level 2: Cost Reduction Strategy indicators
    COST_REDUCTION_PATTERNS = {
        'efficiency': r'\b(efficiency|efficient|optimize|streamline|reduce cost|cost saving)\b',
        'investment': r'\b(invest|investment|long.?term|roi|return on investment)\b',
        'resource': r'\b(resource reduction|waste reduction|energy efficiency|lean)\b',
        'operational': r'\b(operational|operations|process improvement|supply chain optimization)\b'
    }

    # Level 3: Value Proposition Strategy indicators
    VALUE_PROP_PATTERNS = {
        'customer_value': r'\b(customer value|consumer preference|market demand|competitive advantage)\b',
        'differentiation': r'\b(differentiate|differentiation|unique|distinctive|brand)\b',
        'market': r'\b(market position|market share|customer satisfaction|loyalty)\b',
        'communication': r'\b(communicate|convey|promote|advertise|message|stakeholder)\b'
    }

    # Level 4: Identification Strategy indicators
    IDENTIFICATION_PATTERNS = {
        'identity': r'\b(identity|core value|who we are|embedded|integral|fundamental)\b',
        'purpose': r'\b(purpose|mission|vision|commitment|dedicated|passionate)\b',
        'leadership': r'\b(leader|leadership|pioneer|champion|advocate|activist)\b',
        'transformation': r'\b(transform|transformation|revolutionary|radical|systemic change)\b'
    }

    @classmethod
    def detect_maturity_level(cls, text: str) -> Dict[str, any]:
        """
        Analyze text to determine sustainability maturity level.

        Returns dictionary with:
        - dominant_level: Highest scoring level (1-4)
        - level_scores: Dict of scores for each level
        - level_indicators: Specific patterns found per level

        Args:
            text: Text to analyze

        Returns:
            Dictionary with maturity assessment
        """
        text_lower = text.lower()

        level_scores = {
            'regulatory': 0,
            'cost_reduction': 0,
            'value_proposition': 0,
            'identification': 0
        }

        level_indicators = {
            'regulatory': [],
            'cost_reduction': [],
            'value_proposition': [],
            'identification': []
        }

        # Score Level 1: Regulatory
        for category, pattern in cls.REGULATORY_PATTERNS.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                level_scores['regulatory'] += matches
                level_indicators['regulatory'].append(category)

        # Score Level 2: Cost Reduction
        for category, pattern in cls.COST_REDUCTION_PATTERNS.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                level_scores['cost_reduction'] += matches
                level_indicators['cost_reduction'].append(category)

        # Score Level 3: Value Proposition
        for category, pattern in cls.VALUE_PROP_PATTERNS.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                level_scores['value_proposition'] += matches
                level_indicators['value_proposition'].append(category)

        # Score Level 4: Identification
        for category, pattern in cls.IDENTIFICATION_PATTERNS.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                level_scores['identification'] += matches
                level_indicators['identification'].append(category)

        # Determine dominant level
        if max(level_scores.values()) == 0:
            dominant_level = 0  # No clear sustainability strategy
        else:
            level_mapping = {
                'regulatory': 1,
                'cost_reduction': 2,
                'value_proposition': 3,
                'identification': 4
            }
            dominant_key = max(level_scores, key=level_scores.get)
            dominant_level = level_mapping[dominant_key]

        return {
            'dominant_level': dominant_level,
            'level_scores': level_scores,
            'level_indicators': level_indicators,
            'maturity_label': cls._get_level_label(dominant_level)
        }

    @staticmethod
    def _get_level_label(level: int) -> str:
        """Convert numeric level to descriptive label."""
        labels = {
            0: 'none_identified',
            1: 'regulatory',
            2: 'cost_reduction',
            3: 'value_proposition',
            4: 'identification'
        }
        return labels.get(level, 'unknown')


# ============================================================
# CLASS: SectionDetector
# ============================================================
class SectionDetector:
    """
    Detects and categorizes sections in sustainability reports.

    Helps identify which HSBS dimension a chunk belongs to.
    """

    # Section patterns for HSBS dimensions
    ENVIRONMENTAL_PATTERNS = [
        r'\benvironmental\b', r'\bclimate\b', r'\bemissions\b', r'\bcarbon\b',
        r'\benergy\b', r'\bwaste\b', r'\bwater\b', r'\bbiodiversity\b'
    ]

    SOCIAL_PATTERNS = [
        r'\bsocial\b', r'\bemployee\b', r'\blabor\b', r'\bdiversity\b',
        r'\bequity\b', r'\binclusion\b', r'\bcommunity\b', r'\bhuman rights\b',
        r'\bworkforce\b', r'\bsafety\b', r'\bhealth\b'
    ]

    GOVERNANCE_PATTERNS = [
        r'\bgovernance\b', r'\bboard\b', r'\bethics\b', r'\bcompliance\b',
        r'\brisk\b', r'\btransparency\b', r'\baccountability\b', r'\bstakeholder\b'
    ]

    ECONOMIC_PATTERNS = [
        r'\beconomic\b', r'\bfinancial\b', r'\bvalue\b', r'\bgrowth\b',
        r'\bperformance\b', r'\brevenue\b', r'\binvestment\b'
    ]

    @classmethod
    def detect_section_type(cls, text: str) -> List[str]:
        """
        Identify which HSBS dimensions are discussed in text.

        Args:
            text: Text to analyze

        Returns:
            List of dimension tags (e.g., ['environmental', 'social'])
        """
        text_lower = text.lower()
        dimensions = []

        # Check each dimension
        if any(re.search(pattern, text_lower) for pattern in cls.ENVIRONMENTAL_PATTERNS):
            dimensions.append('environmental')

        if any(re.search(pattern, text_lower) for pattern in cls.SOCIAL_PATTERNS):
            dimensions.append('social')

        if any(re.search(pattern, text_lower) for pattern in cls.GOVERNANCE_PATTERNS):
            dimensions.append('governance')

        if any(re.search(pattern, text_lower) for pattern in cls.ECONOMIC_PATTERNS):
            dimensions.append('economic')

        return dimensions if dimensions else ['general']


# ============================================================
# CLASS: DataFrameUtils
# ============================================================
class DataFrameUtils:
    """Utility functions for DataFrame operations and file I/O."""

    @staticmethod
    def save_to_csv(df: pd.DataFrame, filepath: str):
        """Save DataFrame to CSV with Excel-compatible formatting."""
        df.to_csv(
            filepath,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n"
        )

    @staticmethod
    def save_to_excel(df: pd.DataFrame, filepath: str, sheet_name: str = "Sheet1"):
        """Save DataFrame to Excel workbook."""
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    @staticmethod
    def compute_text_statistics(df: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
        """Add word count and character count columns."""

        def calculate_stats(text: str) -> pd.Series:
            text = text if isinstance(text, str) else ""
            word_count = len(re.findall(r"\b\w+\b", text))
            char_count = len(text)
            return pd.Series({
                "word_count_doc": word_count,
                "char_count_doc": char_count
            })

        result_df = df.copy()
        stats = result_df[text_column].fillna("").apply(calculate_stats)
        result_df[["word_count_doc", "char_count_doc"]] = stats
        return result_df


# ============================================================
# CLASS: SustainabilityChunker
# ============================================================
class SustainabilityChunker:
    """
    Creates contextually-aware chunks for sustainability analysis.

    Features:
    - Overlapping chunks to preserve context
    - Section/dimension tagging
    - Sustainability feature detection
    - Metric preservation
    """

    def __init__(self, nlp_model, text_cleaner, chunk_size: int = 5,
                 overlap: int = 1, max_chars: int = 100_000):
        """
        Initialize the sustainability-focused chunker.

        Args:
            nlp_model: Loaded spaCy language model
            text_cleaner: SustainabilityTextCleaner instance
            chunk_size: Number of sentences per chunk (default: 5)
            overlap: Number of sentences to overlap between chunks (default: 1)
            max_chars: Maximum characters to process at once
        """
        self.nlp = nlp_model
        self.text_cleaner = text_cleaner
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chars = max_chars

    def create_chunks(self, documents_df: pd.DataFrame,
                      text_column: str = "clean_text") -> pd.DataFrame:
        """
        Split documents into overlapping, contextually-aware chunks.

        Adds sustainability-specific metadata:
        - hsbs_dimensions: Which dimensions are discussed (environmental, social, etc.)
        - has_metrics: Whether chunk contains quantitative data
        - has_years: Whether chunk contains year references
        - sustainability_score: Count of sustainability keywords

        Args:
            documents_df: DataFrame with cleaned documents
            text_column: Column containing text to chunk

        Returns:
            DataFrame with enriched chunks
        """
        all_chunks: List[Dict] = []

        for doc_idx, row in documents_df.iterrows():
            text = row.get(text_column, "") or ""
            if not text:
                continue

            # Split very long texts into segments
            text_segments = [
                text[i:i + self.max_chars]
                for i in range(0, len(text), self.max_chars)
            ]

            for segment in text_segments:
                # Use spaCy for sentence segmentation
                doc = self.nlp(segment)
                sentences = [sent.text.strip() for sent in doc.sents]

                # Create overlapping chunks
                step_size = max(1, self.chunk_size - self.overlap)

                for i in range(0, len(sentences), step_size):
                    chunk_sentences = sentences[i:i + self.chunk_size]
                    chunk_text = " ".join(chunk_sentences).strip()

                    if not chunk_text:
                        continue

                    # Analyze chunk for sustainability features
                    features = self.text_cleaner.identify_sustainability_content(chunk_text)
                    dimensions = SectionDetector.detect_section_type(chunk_text)
                    maturity = MaturityLevelDetector.detect_maturity_level(chunk_text)

                    # Create enriched chunk record
                    chunk_record = {
                        "document_id": doc_idx,
                        "text_chunk": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "hsbs_dimensions": ",".join(dimensions),
                        "has_metrics": features['has_metrics'],
                        "has_years": features['has_years'],
                        "sustainability_score": features['sustainability_score'],
                        "top_keywords": ",".join(list(features['keywords_found'])[:5]),
                        "maturity_level": maturity['dominant_level'],
                        "maturity_label": maturity['maturity_label'],
                        "regulatory_score": maturity['level_scores']['regulatory'],
                        "cost_reduction_score": maturity['level_scores']['cost_reduction'],
                        "value_prop_score": maturity['level_scores']['value_proposition'],
                        "identification_score": maturity['level_scores']['identification']
                    }

                    # Copy metadata from parent document
                    for col in documents_df.columns:
                        if col != text_column:
                            chunk_record[col] = row[col]

                    all_chunks.append(chunk_record)

        chunks_df = pd.DataFrame(all_chunks)

        if chunks_df.empty:
            return chunks_df

        # Sort chunks
        sort_columns = [
            col for col in ["company", "year", "file_name", "document_id"]
            if col in chunks_df.columns
        ]
        if sort_columns:
            chunks_df = chunks_df.sort_values(
                sort_columns, kind="stable"
            ).reset_index(drop=True)

        # Add chunk IDs
        if "file_name" in chunks_df.columns:
            chunks_df["chunk_id"] = chunks_df.groupby("file_name").cumcount() + 1
        else:
            chunks_df["chunk_id"] = range(1, len(chunks_df) + 1)

        return chunks_df


# ============================================================
# CLASS: SustainabilityPipeline
# ============================================================
class SustainabilityPipeline:
    """
    Main orchestrator for sustainability-focused PDF processing.

    Optimized for HSBS framework analysis with enhanced metadata.
    """

    def __init__(self, metadata_csv: str = "company_metadata.csv",
                 pdf_root: str = ".", chunk_size: int = 5,
                 overlap: int = 1, min_words: int = 20,
                 max_chars: int = 100_000, spacy_model: str = "en_core_web_sm",
                 lowercase: bool = True, preserve_metrics: bool = True):
        """Initialize the sustainability pipeline."""
        self.metadata_csv = metadata_csv
        self.pdf_root = pdf_root
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words
        self.max_chars = max_chars
        self.lowercase = lowercase

        print("[INFO] Initializing sustainability-focused pipeline...")
        self.nlp = DependencyManager.ensure_spacy_model(spacy_model)
        DependencyManager.ensure_cryptography()
        self.text_cleaner = SustainabilityTextCleaner(
            lowercase=lowercase,
            preserve_metrics=preserve_metrics
        )
        self.chunker = SustainabilityChunker(
            self.nlp, self.text_cleaner,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chars=max_chars
        )

    def run(self, output_docs_csv: str = "cleaned_documents.csv",
            output_chunks_csv: str = "cleaned_chunks.csv"):
        """Execute the complete sustainability processing pipeline."""
        print(f"[INFO] Starting pipeline with metadata: {self.metadata_csv}")

        documents_data = self._load_and_process_pdfs()

        if not documents_data:
            print("[INFO] No valid documents processed. Exiting.")
            return

        docs_df = self._prepare_documents_dataframe(documents_data)
        self._save_document_outputs(docs_df, output_docs_csv)
        self._create_and_save_chunks(docs_df, output_chunks_csv)

        print("[DONE] Sustainability processing complete!")

    def _load_and_process_pdfs(self) -> List[Dict]:
        """Load metadata and process all PDFs."""
        if not os.path.exists(self.metadata_csv):
            print(f"[ERROR] Metadata file not found: {self.metadata_csv}")
            return []

        metadata_df = pd.read_csv(self.metadata_csv)
        print(f"[INFO] Loaded {len(metadata_df)} entries from metadata")

        documents_data: List[Dict] = []

        for idx, row in metadata_df.iterrows():
            raw_filename = str(row.get("file_name", "")).strip()
            if not raw_filename:
                continue

            filename = PDFProcessor.ensure_pdf_extension(raw_filename)
            pdf_path = filename if os.path.isabs(filename) else os.path.join(self.pdf_root, filename)

            if not os.path.exists(pdf_path):
                print(f"[WARN] Missing file: {pdf_path}")
                continue

            try:
                print(f"[INFO] Processing: {filename}")
                raw_text = PDFProcessor.read_pdf_text(pdf_path, password=row.get("pdf_password"))
                cleaned_text = self.text_cleaner.clean(raw_text)

                documents_data.append({
                    "company": row.get("Company"),
                    "year": row.get("year"),
                    "doc_type": row.get("doc_type"),
                    "file_name": filename,
                    "clean_text": cleaned_text
                })

            except RuntimeError:
                print(f"[WARN] Skipping password-locked PDF: {filename}")
                continue
            except Exception as e:
                print(f"[ERROR] {filename}: {e}")
                continue

        print(f"[INFO] Successfully processed {len(documents_data)} documents")
        return documents_data

    def _prepare_documents_dataframe(self, documents_data: List[Dict]) -> pd.DataFrame:
        """Convert document data to DataFrame and add statistics."""
        df = pd.DataFrame(documents_data)
        df["file_name"] = df["file_name"].map(PDFProcessor.ensure_pdf_extension)

        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")

        df = DataFrameUtils.compute_text_statistics(df, "clean_text")

        sort_cols = [col for col in ["company", "year", "file_name"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

        return df

    def _save_document_outputs(self, docs_df: pd.DataFrame, output_csv: str):
        """Save document-level results."""
        print("[INFO] Saving document-level outputs...")
        DataFrameUtils.save_to_csv(docs_df, output_csv)
        print(f"[OK] Saved: {output_csv}")

        manifest_cols = ["company", "year", "doc_type", "file_name"]
        manifest_df = docs_df[[col for col in manifest_cols if col in docs_df.columns]]
        DataFrameUtils.save_to_csv(manifest_df, "outputs/cleaned_documents_manifest.csv")

        print("[OK] Saved: cleaned_documents_manifest.csv")

        DataFrameUtils.save_to_excel(docs_df, "outputs/cleaned_documents.xlsx", sheet_name="documents")

        print("[OK] Saved: cleaned_documents.xlsx")

    def _create_and_save_chunks(self, docs_df: pd.DataFrame, output_csv: str):
        """Create sustainability-aware chunks and save results."""
        print("[INFO] Creating sustainability-aware chunks...")

        chunks_df = self.chunker.create_chunks(docs_df, text_column="clean_text")

        if chunks_df.empty:
            print("[INFO] No chunks created.")
            return

        print(f"[INFO] Filtering chunks with < {self.min_words} words...")
        chunks_df = chunks_df[chunks_df["word_count"] >= self.min_words].reset_index(drop=True)

        sort_cols = [
            col for col in ["company", "year", "file_name", "document_id", "chunk_id"]
            if col in chunks_df.columns
        ]
        if sort_cols:
            chunks_df = chunks_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

        if "file_name" in chunks_df.columns:
            chunks_df["chunk_id"] = chunks_df.groupby("file_name").cumcount() + 1

        DataFrameUtils.save_to_csv(chunks_df, output_csv)
        print(f"[OK] Saved: {output_csv}")

        DataFrameUtils.save_to_excel(chunks_df, "outputs/cleaned_chunks.xlsx", sheet_name="chunks")
        print("[OK] Saved: cleaned_chunks.xlsx")

        # Print summary statistics
        print(f"\n[SUMMARY] Created {len(chunks_df)} chunks from {len(docs_df)} documents")
        if 'sustainability_score' in chunks_df.columns:
            avg_score = chunks_df['sustainability_score'].mean()
            print(f"[SUMMARY] Average sustainability keywords per chunk: {avg_score:.2f}")
        if 'hsbs_dimensions' in chunks_df.columns:
            print(f"[SUMMARY] HSBS dimension distribution:")
            for dim in ['environmental', 'social', 'governance', 'economic']:
                count = chunks_df['hsbs_dimensions'].str.contains(dim, case=False).sum()
                print(f"  - {dim.capitalize()}: {count} chunks ({count / len(chunks_df) * 100:.1f}%)")
        if 'maturity_level' in chunks_df.columns:
            print(f"\n[SUMMARY] Maturity Level Distribution (Gabler et al. Hierarchy):")
            level_counts = chunks_df['maturity_label'].value_counts()
            for level in ['identification', 'value_proposition', 'cost_reduction', 'regulatory', 'none_identified']:
                count = level_counts.get(level, 0)
                pct = (count / len(chunks_df) * 100) if len(chunks_df) > 0 else 0
                print(f"  - Level {level.replace('_', ' ').title()}: {count} chunks ({pct:.1f}%)")


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    """Run the sustainability-focused PDF processing pipeline."""
    print("=" * 60)
    print("SUSTAINABILITY PDF TEXT PROCESSING PIPELINE")
    print("Optimized for HSBS Framework Analysis")
    print("=" * 60)

    pipeline = SustainabilityPipeline(
        metadata_csv=METADATA_CSV,
        pdf_root=PDF_ROOT_FOLDER,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP_SENTENCES,
        min_words=MIN_WORDS,
        max_chars=MAX_CHARS,
        spacy_model=SPACY_MODEL,
        lowercase=LOWERCASE,
        preserve_metrics=PRESERVE_METRICS
    )

    pipeline.run(
        output_docs_csv=OUTPUT_DOCS_CSV,
        output_chunks_csv=OUTPUT_CHUNKS_CSV
    )

    print("=" * 60)


if __name__ == "__main__":
    main()