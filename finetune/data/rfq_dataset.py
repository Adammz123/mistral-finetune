"""
RFQ Dataset Processor for extracting text from quotation PDFs.
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

import pdfplumber
import pandas as pd


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RFQDatasetProcessor:
    """Process RFQ quotations from PDFs and ground truth XLSX files."""
    
    def __init__(self, pdf_folder: Path, xlsx_path: Path, output_path: Path):
        """
        Initialize the RFQ dataset processor.
        
        Args:
            pdf_folder: Path to folder containing quotation PDFs
            xlsx_path: Path to XLSX file with ground truth data
            output_path: Path to output JSONL file
        """
        self.pdf_folder = Path(pdf_folder)
        self.xlsx_path = Path(xlsx_path)
        self.output_path = Path(output_path)
        
        if not self.pdf_folder.exists():
            raise ValueError(f"PDF folder does not exist: {self.pdf_folder}")
        if not self.xlsx_path.exists():
            raise ValueError(f"XLSX file does not exist: {self.xlsx_path}")
    
    @staticmethod
    def get_standard_quotation_with_price_breaks_prompt(text: str) -> str:
        """Get prompt for standard quotation parsing with price breaks support."""
        return (
            f"Extract the following information from the quotation below and return only a JSON object. "
            f"The JSON should include these top-level keys: supplier_name, quotation_id, valid_until_date, incoterms, payment_terms, warranty_terms, currency, and additional_notes. "
            f"It should also include a key called parts, which is an array of objects. "
            f"Each object in parts should contain part_number, part_description, manufacturer, unit_of_measure, lead_time_days_weeks (e.g., '5 days' or '2 weeks' or 'in stock', 'in stock, 5 weeks for additional stock'), tariff_charge (e.g., '12.5%' or '0%' or 'N/A'), and a key called price_breaks. "
            f"The price_breaks key should be an array of objects, where each object contains quantity, minimum_order_quantity, price_per_unit, and total_price. "
            f"If there are no price breaks (just single pricing), create one price_breaks object with the available pricing information.\n\n"
            f"Quotation:\n{text}\n\n"
            f"JSON Response:"
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text from PDF."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = text.strip()
        return text
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text from a single PDF file using PDF Plumber.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        result = {
            'file_path': str(pdf_path),
            'file_name': pdf_path.name,
            'extraction_timestamp': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'text_content': '',
            'page_count': 0,
            'text_length': 0,
            'pages': []
        }
        
        try:
            logger.info(f"Extracting text from: {pdf_path.name}")
            
            with pdfplumber.open(pdf_path) as pdf:
                result['page_count'] = len(pdf.pages)
                
                all_text = []
                page_texts = []
                
                for page_number, page in enumerate(pdf.pages, start=1):
                    try:
                        # Extract text from the page
                        page_text = page.extract_text()
                        
                        if page_text:
                            # Clean up the text
                            cleaned_text = self._clean_text(page_text)
                            page_texts.append({
                                'page_number': page_number,
                                'text': cleaned_text,
                                'text_length': len(cleaned_text)
                            })
                            all_text.append(cleaned_text)
                        else:
                            # Page has no extractable text
                            page_texts.append({
                                'page_number': page_number,
                                'text': '',
                                'text_length': 0
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_number} of {pdf_path.name}: {e}")
                        page_texts.append({
                            'page_number': page_number,
                            'text': '',
                            'text_length': 0,
                            'error': str(e)
                        })
                
                # Combine all text
                full_text = '\n\n--- Page Break ---\n\n'.join(all_text)
                result['text_content'] = full_text
                result['text_length'] = len(full_text)
                result['pages'] = page_texts
                result['success'] = True
                
                logger.info(f"Successfully extracted {result['text_length']} characters from {result['page_count']} pages")
                
        except Exception as e:
            error_msg = f"Error processing PDF {pdf_path.name}: {str(e)}"
            logger.error(error_msg)
            result['error'] = error_msg
            result['success'] = False
        
        return result
    
    def load_ground_truth_from_xlsx(self) -> pd.DataFrame:
        """
        Load ground truth data from XLSX file.
        
        Returns:
            DataFrame containing ground truth data
        """
        logger.info(f"Loading ground truth from: {self.xlsx_path}")
        df = pd.read_excel(self.xlsx_path)
        logger.info(f"Loaded {len(df)} rows from ground truth file")
        return df
    
    def format_ground_truth_to_json(self, row: pd.Series) -> Dict[str, Any]:
        """
        Format a ground truth row into the expected JSON structure.
        
        Args:
            row: A row from the ground truth DataFrame
            
        Returns:
            Dictionary in the expected JSON format
        """
        # Create the base structure
        quotation_data = {
            "supplier_name": row.get('supplier_name', None),
            "quotation_id": str(row.get('quotation_id', '')),
            "valid_until_date": row.get('valid_until_date', None),
            "incoterms": row.get('incoterms', None),
            "payment_terms": row.get('payment_terms', None),
            "warranty_terms": row.get('warranty_terms', None),
            "currency": row.get('currency', 'USD'),
            "additional_notes": row.get('additional_notes', None),
            "parts": []
        }
        
        # Parse parts information - assuming parts are stored as JSON string in the XLSX
        # or multiple columns with part information
        if 'parts' in row and pd.notna(row['parts']):
            try:
                # If parts is already a JSON string
                if isinstance(row['parts'], str):
                    quotation_data['parts'] = json.loads(row['parts'])
                else:
                    quotation_data['parts'] = row['parts']
            except json.JSONDecodeError:
                logger.warning(f"Could not parse parts JSON for quotation {row.get('quotation_id', 'unknown')}")
        else:
            # Build parts array from individual columns
            part = {
                "part_number": row.get('part_number', ''),
                "part_description": row.get('part_description', ''),
                "manufacturer": row.get('manufacturer', ''),
                "unit_of_measure": row.get('unit_of_measure', ''),
                "lead_time_days_weeks": row.get('lead_time_days_weeks', ''),
                "tariff_charge": row.get('tariff_charge', 'N/A'),
                "price_breaks": []
            }
            
            # Parse price breaks - assuming they're in JSON format or separate columns
            if 'price_breaks' in row and pd.notna(row['price_breaks']):
                try:
                    if isinstance(row['price_breaks'], str):
                        part['price_breaks'] = json.loads(row['price_breaks'])
                    else:
                        part['price_breaks'] = row['price_breaks']
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse price_breaks for part {part['part_number']}")
            else:
                # Build price break from individual columns
                price_break = {
                    "quantity": int(row.get('quantity', 0)) if pd.notna(row.get('quantity')) else 0,
                    "minimum_order_quantity": int(row.get('minimum_order_quantity', 0)) if pd.notna(row.get('minimum_order_quantity')) else 0,
                    "price_per_unit": float(row.get('price_per_unit', 0.0)) if pd.notna(row.get('price_per_unit')) else 0.0,
                    "total_price": float(row.get('total_price', 0.0)) if pd.notna(row.get('total_price')) else 0.0
                }
                part['price_breaks'].append(price_break)
            
            quotation_data['parts'].append(part)
        
        return quotation_data
    
    def generate_prompt_id(self, text: str) -> str:
        """Generate a unique prompt ID from text content."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def create_training_sample(self, pdf_text: str, ground_truth_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a training sample in the ultrachat format.
        
        Args:
            pdf_text: Extracted text from PDF
            ground_truth_json: Ground truth data in JSON format
            
        Returns:
            Dictionary in ultrachat JSONL format
        """
        # Create the full prompt with the quotation text
        full_prompt = self.get_standard_quotation_with_price_breaks_prompt(pdf_text)
        
        # Convert ground truth to JSON string for the assistant response
        assistant_response = json.dumps(ground_truth_json, indent=2)
        
        # Create the training sample
        sample = {
            "prompt": full_prompt,
            "prompt_id": self.generate_prompt_id(pdf_text),
            "messages": [
                {
                    "content": full_prompt,
                    "role": "user"
                },
                {
                    "content": assistant_response,
                    "role": "assistant"
                }
            ]
        }
        
        return sample
    
    def process_dataset(self) -> None:
        """Process all PDFs and create training dataset."""
        logger.info("Starting dataset processing...")
        
        # Load ground truth
        ground_truth_df = self.load_ground_truth_from_xlsx()
        
        # Get all PDF files
        pdf_files = list(self.pdf_folder.glob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process each PDF and create training samples
        training_samples = []
        
        for pdf_path in pdf_files:
            # Extract text from PDF
            extraction_result = self.extract_text_from_pdf(pdf_path)
            
            if not extraction_result['success']:
                logger.warning(f"Skipping {pdf_path.name} due to extraction error")
                continue
            
            pdf_text = extraction_result['text_content']
            
            # Find matching ground truth - match by filename or quotation_id
            # Assuming the PDF filename contains the quotation_id
            pdf_stem = pdf_path.stem  # filename without extension
            
            # Try to find matching row in ground truth
            matching_rows = ground_truth_df[
                ground_truth_df['quotation_id'].astype(str).str.contains(pdf_stem, case=False, na=False) |
                ground_truth_df.get('file_name', pd.Series(dtype=str)).str.contains(pdf_stem, case=False, na=False)
            ]
            
            if matching_rows.empty:
                logger.warning(f"No ground truth found for {pdf_path.name}")
                continue
            
            # Process each matching row (there might be multiple parts)
            for idx, row in matching_rows.iterrows():
                ground_truth_json = self.format_ground_truth_to_json(row)
                training_sample = self.create_training_sample(pdf_text, ground_truth_json)
                training_samples.append(training_sample)
        
        # Write to JSONL file
        logger.info(f"Writing {len(training_samples)} training samples to {self.output_path}")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Dataset processing complete! Output saved to {self.output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Process RFQ quotation PDFs and ground truth XLSX to create training dataset'
    )
    
    parser.add_argument(
        '--pdf-folder',
        type=str,
        required=True,
        help='Path to folder containing quotation PDF files'
    )
    
    parser.add_argument(
        '--xlsx-path',
        type=str,
        required=True,
        help='Path to XLSX file containing ground truth data'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to output JSONL file'
    )
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = RFQDatasetProcessor(
        pdf_folder=Path(args.pdf_folder),
        xlsx_path=Path(args.xlsx_path),
        output_path=Path(args.output_path)
    )
    
    processor.process_dataset()


if __name__ == '__main__':
    main()
