# clinical-pdf-extractor

LangGraph-based pipeline for extracting structured clinical trial data from research PDFs.

## Architecture

```
PDF -> [parse_pdf] -> [classify_doc] -> [extract_clinical/GPT-4o] -> [validate_tables] -> JSON
```

## Setup

```bash
pip install -r requirements.txt
python main.py --pdf your_paper.pdf --mode langgraph
```

## Demo mode (no API key needed)

```bash
python main.py --mode demo
```

## Output

```json
{
  "clinical_data": {"title": "...", "phase": "III", "sample_size": 486, "p_value": "0.003"},
  "tables": [...],
  "extraction_confidence": 0.85
}
```

## Design Notes

- LangGraph state machine: each node independently testable and retryable
- Pydantic output validation on every LLM response
- Dual-strategy table extraction (line-detect + bbox fallback)
- Confidence scoring per extraction for downstream gating

Built by Dr. Sandeep Grover | https://github.com/Sandyyy123
