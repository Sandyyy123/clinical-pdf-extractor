#!/usr/bin/env python3
"""Clinical PDF Extractor - python main.py --pdf paper.pdf [--mode langgraph|demo]"""
import argparse, json, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Extract structured clinical data from research PDFs")
    parser.add_argument("--pdf", default="demo.pdf")
    parser.add_argument("--mode", choices=["langgraph","demo"], default="demo")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "langgraph":
        try:
            from extractor import build_extraction_graph
            graph = build_extraction_graph()
            state = graph.invoke({"pdf_path": args.pdf, "text_blocks":[], "tables_raw":[], "doc_type":"", "clinical_data":None, "validated_tables":[], "warnings":[], "confidence":0.0})
            result_dict = {"clinical_data": state["clinical_data"], "tables_extracted": len(state["validated_tables"]), "confidence": state["confidence"], "warnings": state["warnings"], "doc_type": state["doc_type"]}
        except Exception as e:
            logger.error(f"LangGraph failed: {e}. Falling back to demo.")
            args.mode = "demo"

    if args.mode == "demo":
        from extractor import extract_demo
        result = extract_demo(args.pdf)
        result_dict = result.model_dump()

    output_json = json.dumps(result_dict, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(output_json)
        logger.info(f"Saved to {args.output}")
    else:
        print(output_json)

if __name__ == "__main__":
    main()
