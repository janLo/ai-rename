#!uv run

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pymupdf",
#     "openai",
#     "pydantic",
# ]
# ///

import argparse
import fitz  # PyMuPDF
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# Define your desired schema for the filename components
class DocumentMeta(BaseModel):
    date: str      # Expected format: YYYYMMDD
    entity: str    # The sender or receiver that is NOT the document owner
    summary: str   # Short 1-5 word summary

def extract_text(pdf_path: Path, max_pages: int) -> str:
    """Extracts text from the first few pages of the OCR'd PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(min(max_pages, len(doc))):
            text += doc[page_num].get_text()
        return text
    except Exception as e:
        print(f"[!] Error reading {pdf_path.name}: {e}")
        return ""

def generate_metadata(client: OpenAI, model: str, text: str) -> DocumentMeta | None:
    """Sends text to local LLM and returns structured metadata."""
    prompt = """
    You are a document classification assistant. Analyze the following OCR text from a document.
    Extract the date of the document, the entity (the sender or receiver that is NOT the document owner/we), and a short 1-5 word summary.
    Respond ONLY with a valid JSON object matching this schema:
    {"date": "YYYYMMDD", "entity": "string", "summary": "string"}
    If a date cannot be found, use "00000000". Note the date format has no dashes.
    We are Jan Losinski, Gemma Jones-Losinski and Albert Losinski.
    Keep the Summary in the language of the document.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                # Truncate text to roughly 4000 characters to keep context clean and fast
                {"role": "user", "content": f"Document text:\n{text[:4000]}"} 
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Parse the JSON string into our Pydantic model
        result_json = response.choices[0].message.content.strip()
        
        # Clean markdown formatting if the LLM includes it
        if result_json.startswith("```"):
            result_json = result_json.strip("` \n")
            if result_json.lower().startswith("json"):
                result_json = result_json[4:].strip()
                
        return DocumentMeta.model_validate_json(result_json)
        
    except ValidationError as e:
        print(f"[!] LLM returned invalid schema: {e}")
        import pdb; pdb.set_trace()
        return None
    except Exception as e:
        print(f"[!] LLM request failed: {e}")
        return None

def sanitize_filename_part(part: str) -> str:
    """Removes unsafe characters for file paths but keeps spaces."""
    clean = "".join(c for c in part if c.isalnum() or c in " -_").strip()
    return " ".join(clean.split()) # Normalizes multiple spaces into a single space

def process_paths(paths: list[Path], client: OpenAI, model: str, max_pages: int, dry_run: bool):
    """Iterates through specified files/directories, finding PDFs recursively, and renames them."""
    pdf_files = []
    seen = set()

    for path in paths:
        if not path.exists():
            print(f"[!] Warning: Path '{path}' does not exist. Skipping.")
            continue

        if path.is_file():
            if path not in seen:
                seen.add(path)
                pdf_files.append(path)
        elif path.is_dir():
            found_pdfs = []
            for p in path.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".pdf":
                    found_pdfs.append(p)
            found_pdfs.sort()
            for p in found_pdfs:
                if p not in seen:
                    seen.add(p)
                    pdf_files.append(p)
        else:
            print(f"[!] Warning: Path '{path}' is neither a file nor a directory. Skipping.")

    if not pdf_files:
        print("[-] No PDFs found in the specified paths.")
        return

    print(f"[*] Found {len(pdf_files)} PDFs. Processing with model '{model}'...")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path}")
        
        text = extract_text(pdf_path, max_pages)
        if not text.strip():
            print("  -> No text found (maybe not OCR'd properly?), skipping.")
            continue
            
        meta = generate_metadata(client, model, text)
        if not meta:
            print("  -> Failed to extract metadata, skipping.")
            continue
        if meta.date.startswith("0000"):
            print("  -> No valid date found, skipping")
            continue
            
        # Clean strings for safe filenames
        safe_entity = sanitize_filename_part(meta.entity)
        safe_summary = sanitize_filename_part(meta.summary)
        
        # Apply pattern: YYYYMMDDZ - ENTITY - SUMMARY
        new_filename = f"{meta.date}Z - {safe_entity} - {safe_summary}.pdf"
        new_path = pdf_path.with_name(new_filename)
        
        # Prevent overwriting
        if new_path.exists() and new_path != pdf_path:
            print(f"  -> File '{new_filename}' already exists. Skipping rename.")
            continue
            
        if dry_run:
            print(f"  -> [DRY RUN] Would rename to: {new_filename}")
        else:
            if new_path != pdf_path:
                print(f"  -> Renaming to: {new_filename}")
                pdf_path.rename(new_path)
            else:
                print("  -> Filename is already correct. Skipping.")

def main():
    parser = argparse.ArgumentParser(
        description="Rename OCR'd PDFs based on their text content using a local LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Positional argument for the target paths
    parser.add_argument("paths", type=Path, nargs="+", help="Paths to files or directories containing PDFs.")
    
    # Optional configurations
    parser.add_argument("--model", type=str, default="gemma4-it:e4b", help="FastFlowLM model to use.")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="Lemonade server API Base URL.")
    parser.add_argument("--max-pages", type=int, default=2, help="Number of pages to read per PDF.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned renames without actually changing files.")

    args = parser.parse_args()

    # Configure local Lemonade Server client
    client = OpenAI(
        base_url=args.url,
        api_key="sk-no-key-needed" # Local APIs usually ignore this, but the OpenAI client requires it to be set
    )

    process_paths(
        paths=args.paths,
        client=client,
        model=args.model,
        max_pages=args.max_pages,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
