"""PDF parser for clinical research documents."""
import re
from pathlib import Path
from typing import Optional
import logging
logger = logging.getLogger(__name__)
try:
    import pdfplumber
    import fitz
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

def extract_text_blocks(pdf_path: str) -> list[dict]:
    if not PDF_AVAILABLE:
        return _demo_blocks()
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                blocks.append({"text": str(table), "page": page_num+1, "block_type": "table", "raw_table": table})
            full_text = page.extract_text(x_tolerance=3, layout=True)
            if full_text:
                blocks.append({"text": full_text, "page": page_num+1, "block_type": "text"})
    return blocks

def extract_tables_structured(pdf_path: str) -> list[dict]:
    if not PDF_AVAILABLE:
        return []
    tables_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            settings = {"vertical_strategy":"lines","horizontal_strategy":"lines","snap_tolerance":5,"join_tolerance":5}
            tables = page.extract_tables(settings)
            for idx, table in enumerate(tables):
                if not table or not table[0]:
                    continue
                tables_out.append({"page":page_num+1,"table_index":idx,"headers":[str(c or "").strip() for c in table[0]],"rows":[[str(c or "").strip() for c in row] for row in table[1:]]})
    return tables_out

def detect_document_type(text_sample: str) -> str:
    patterns = {"rct":r"random(iz|is)ed|placebo.control","meta_analysis":r"meta.anal|systematic review","regulatory":r"clinical evaluation report|IVDR|CER"}
    for doc_type, pattern in patterns.items():
        if re.search(pattern, text_sample, re.IGNORECASE):
            return doc_type
    return "unknown"

def _demo_blocks() -> list[dict]:
    return [
        {"text": "PHASE III RANDOMIZED CONTROLLED TRIAL\nN=486 patients\nPrimary endpoint: Overall survival at 24 months", "page":1, "block_type":"text"},
        {"text": "[['Treatment','n','AE Grade 3-4','p-value'],['Drug A 10mg','243','18 (7.4%)','<0.001'],['Placebo','243','12 (4.9%)','ref']]", "page":3, "block_type":"table"},
    ]
