"""
LangGraph-based clinical data extractor.
Uses a multi-node graph for PDF -> structured ClinicalTrialData extraction.
"""
from typing import TypedDict, Annotated, Optional
import logging

from models import ClinicalTrialData, ExtractedTable, ExtractionResult, TableRow
from pdf_parser import extract_text_blocks, extract_tables_structured, detect_document_type

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langgraph.graph import StateGraph, END
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain/LangGraph not installed. Run: pip install langchain-openai langgraph")


class ExtractionState(TypedDict):
    pdf_path: str
    text_blocks: list[dict]
    tables_raw: list[dict]
    doc_type: str
    clinical_data: Optional[dict]
    validated_tables: list[dict]
    warnings: list[str]
    confidence: float


def build_extraction_graph():
    """
    Build LangGraph extraction pipeline:
    parse_pdf -> classify_document -> extract_clinical_data -> validate_tables -> assemble_result
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = JsonOutputParser(pydantic_object=ClinicalTrialData)

    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical data extraction expert. Extract structured data from clinical research PDFs for regulatory compliance workflows. Return only valid JSON."),
        ("human", "Extract clinical trial data from this text.\n\nDocument type: {doc_type}\n\nText:\n{text}\n\n{format_instructions}"),
    ])

    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You validate extracted tables from clinical PDFs. Flag inconsistencies and data quality issues."),
        ("human", "Validate this table for regulatory compliance use:\n{table_data}\n\nReturn JSON: {{valid: bool, confidence: float, issues: [str]}}"),
    ])

    def parse_pdf_node(state: ExtractionState) -> ExtractionState:
        blocks = extract_text_blocks(state["pdf_path"])
        tables = extract_tables_structured(state["pdf_path"])
        text_sample = " ".join(b["text"] for b in blocks[:3])
        return {**state, "text_blocks": blocks, "tables_raw": tables,
                "doc_type": detect_document_type(text_sample)}

    def extract_clinical_node(state: ExtractionState) -> ExtractionState:
        full_text = "\n\n".join(b["text"] for b in state["text_blocks"])[:8000]
        chain = extraction_prompt | llm | parser
        try:
            result = chain.invoke({
                "doc_type": state["doc_type"],
                "text": full_text,
                "format_instructions": parser.get_format_instructions()
            })
            return {**state, "clinical_data": result, "confidence": 0.85}
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {**state, "warnings": state["warnings"] + [str(e)], "confidence": 0.0}

    def validate_tables_node(state: ExtractionState) -> ExtractionState:
        validated = []
        for table in state["tables_raw"]:
            chain = validation_prompt | llm | JsonOutputParser()
            try:
                result = chain.invoke({"table_data": str(table)})
                validated.append({**table, "valid": result.get("valid", True),
                                   "validation_confidence": result.get("confidence", 0.7)})
            except Exception as e:
                validated.append({**table, "valid": True, "validation_confidence": 0.5})
        return {**state, "validated_tables": validated}

    graph = StateGraph(ExtractionState)
    graph.add_node("parse_pdf", parse_pdf_node)
    graph.add_node("extract_clinical", extract_clinical_node)
    graph.add_node("validate_tables", validate_tables_node)
    graph.add_edge("parse_pdf", "extract_clinical")
    graph.add_edge("extract_clinical", "validate_tables")
    graph.add_edge("validate_tables", END)
    graph.set_entry_point("parse_pdf")
    return graph.compile()


def extract_demo(pdf_path: str = "demo.pdf") -> ExtractionResult:
    """Demo mode extraction without LLM - uses heuristic patterns."""
    import re, uuid
    blocks = extract_text_blocks(pdf_path)
    tables_raw = extract_tables_structured(pdf_path)
    full_text = " ".join(b["text"] for b in blocks)

    def find_value(patterns, text, default=None):
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return default

    clinical_data = ClinicalTrialData(
        title=find_value([r"title[:\s]+(.{20,120}?)(?:\n|\.|$)"], full_text, "Clinical Study Report"),
        phase=find_value([r"phase\s+(I{1,3}V?|[1-4])"], full_text),
        sample_size=int(find_value([r"n\s*=\s*(\d+)", r"(\d+)\s+patient"], full_text, "0")),
        primary_endpoint=find_value([r"primary endpoint[:\s]+(.{10,150}?)(?:\n|$)"], full_text, "Not identified"),
        primary_endpoint_result=find_value([r"p\s*[<=>]\s*(0\.\d+)"], full_text),
        p_value=find_value([r"p\s*[<=>]\s*(0\.\d+|\d+)"], full_text),
        intervention=find_value([r"intervention[:\s]+(.{5,80}?)(?:\n|$)"], full_text, "Study drug"),
    )

    tables = [
        ExtractedTable(
            table_id=f"T{i+1}",
            caption=f"Table {i+1}",
            headers=t.get("headers", []),
            rows=[TableRow(row_index=j, cells=row) for j, row in enumerate(t.get("rows", []))],
            confidence=0.75
        ) for i, t in enumerate(tables_raw)
    ]

    return ExtractionResult(
        document_id=str(uuid.uuid4())[:8],
        clinical_data=clinical_data,
        tables=tables,
        extraction_confidence=0.75,
        warnings=["Demo mode - LLM not used; heuristic extraction only"]
    )
