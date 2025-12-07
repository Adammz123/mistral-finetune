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
    
    def __init__(self, pdf_parent_folder: Path, xlsx_path: Path, output_path: Path):
        """
        Initialize the RFQ dataset processor.
        
        Args:
            pdf_parent_folder: Path to parent folder containing subdirectories of quotation PDFs
                              Each subdirectory represents a different PDF format/template
            xlsx_path: Path to XLSX file with ground truth data
            output_path: Path to output JSONL file
        """
        self.pdf_parent_folder = Path(pdf_parent_folder)
        self.xlsx_path = Path(xlsx_path)
        self.output_path = Path(output_path)
        
        # Validate parent folder exists
        if not self.pdf_parent_folder.exists():
            raise ValueError(f"PDF parent folder does not exist: {self.pdf_parent_folder}")
        
        if not self.xlsx_path.exists():
            raise ValueError(f"XLSX file does not exist: {self.xlsx_path}")
        
        # Automatically discover subdirectories containing PDFs
        self.pdf_folders = self._discover_pdf_folders()
        
        if not self.pdf_folders:
            raise ValueError(f"No subdirectories with PDF files found in {self.pdf_parent_folder}")
        
        logger.info(f"Discovered {len(self.pdf_folders)} PDF template folder(s):")
        for folder in self.pdf_folders:
            pdf_count = len(list(folder.glob('*.pdf')))
            logger.info(f"  - {folder.name}: {pdf_count} PDF(s)")
    
    def _discover_pdf_folders(self) -> List[Path]:
        """
        Automatically discover subdirectories containing PDF files.
        
        Returns:
            List of paths to subdirectories containing PDF files
        """
        pdf_folders = []
        
        # Check all subdirectories
        for item in self.pdf_parent_folder.iterdir():
            if item.is_dir():
                # Check if this directory contains any PDF files
                pdf_files = list(item.glob('*.pdf'))
                if pdf_files:
                    pdf_folders.append(item)
        
        return sorted(pdf_folders)  # Sort for consistent ordering
    
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
        This handles the specific format where each row may contain up to 3 parts.
        
        Args:
            row: A row from the ground truth DataFrame
            
        Returns:
            Dictionary in the expected JSON format
        """
        # Create the base structure with quotation-level information
        quotation_data = {
            "supplier_name": str(row.get('supplier_name', '')) if pd.notna(row.get('supplier_name')) else None,
            "quotation_id": str(row.get('quotation_id', '')),
            "valid_until_date": str(row.get('valid_until_date', ''))[:10] if pd.notna(row.get('valid_until_date')) else None,
            "incoterms": str(row.get('incoterms', '')) if pd.notna(row.get('incoterms')) and str(row.get('incoterms', '')).lower() not in ['null', 'nan', 'none', ''] else None,
            "payment_terms": str(row.get('payment_terms', '')) if pd.notna(row.get('payment_terms')) and str(row.get('payment_terms', '')).lower() not in ['null', 'nan', 'none', ''] else None,
            "warranty_terms": str(row.get('warranty_terms', '')) if pd.notna(row.get('warranty_terms')) and str(row.get('warranty_terms', '')).lower() not in ['null', 'nan', 'none', ''] else None,
            "currency": str(row.get('currency', 'USD')),
            "additional_notes": str(row.get('additional_notes', '')) if pd.notna(row.get('additional_notes')) else None,
            "parts": []
        }
        
        # Process up to 3 parts per row (part1, part2, part3)
        # Part 1 has no suffix, Part 2 has "2" suffix, Part 3 has "3" suffix
        part_suffixes = ['', '2', '3']
        
        for suffix in part_suffixes:
            # Check if this part exists (part_number is required)
            part_num_col = f'part_number{suffix}' if suffix else 'part_number'
            
            if part_num_col not in row or not pd.notna(row.get(part_num_col)):
                continue
            
            # Build part information
            part = {
                "part_number": str(row.get(f'part_number{suffix}', '')) if pd.notna(row.get(f'part_number{suffix}')) else '',
                "part_description": str(row.get(f'part_description{suffix}', '')) if pd.notna(row.get(f'part_description{suffix}')) else '',
                "manufacturer": f"{row.get('manufacturer', '')} / {row.get(f'Vendor Part Number{suffix}', '')}" if pd.notna(row.get(f'Vendor Part Number{suffix}')) else str(row.get('manufacturer', '')),
                "unit_of_measure": str(row.get(f'unit_of_measure{suffix}', '')) if pd.notna(row.get(f'unit_of_measure{suffix}')) else '',
                "lead_time_days_weeks": str(row.get(f'lead_time_days_weeks{suffix}', '')) if pd.notna(row.get(f'lead_time_days_weeks{suffix}')) else '',
                "tariff_charge": str(row.get('tariff_charge', 'N/A')) if suffix == '' else 'N/A',  # Only first part has tariff
                "price_breaks": []
            }
            
            # Build price break for this part
            quantity_col = f'quantity{suffix}' if suffix else 'quantity'
            if pd.notna(row.get(quantity_col)):
                price_break = {
                    "quantity": int(row.get(quantity_col, 0)) if pd.notna(row.get(quantity_col)) else 0,
                    "minimum_order_quantity": int(row.get(quantity_col, 0)) if pd.notna(row.get(quantity_col)) else 0,  # Use same as quantity if MOQ not available
                    "price_per_unit": float(row.get(f'price_per_unit{suffix}', 0.0)) if pd.notna(row.get(f'price_per_unit{suffix}')) else 0.0,
                    "total_price": float(row.get(f'total_price{suffix}', 0.0)) if pd.notna(row.get(f'total_price{suffix}')) else 0.0
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
        """Process all PDFs from all folders and create training dataset."""
        logger.info("Starting dataset processing...")
        logger.info(f"Processing {len(self.pdf_folders)} PDF folder(s)")
        
        # Load ground truth
        ground_truth_df = self.load_ground_truth_from_xlsx()
        
        # Collect all PDF files from all folders
        all_pdf_files = []
        for folder in self.pdf_folders:
            pdf_files = list(folder.glob('*.pdf'))
            logger.info(f"Found {len(pdf_files)} PDF files in {folder.name}")
            all_pdf_files.extend([(pdf_path, folder.name) for pdf_path in pdf_files])
        
        if not all_pdf_files:
            logger.error(f"No PDF files found in any of the {len(self.pdf_folders)} folder(s)")
            return
        
        logger.info(f"Total PDF files to process: {len(all_pdf_files)}")
        
        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process each PDF and create training samples
        training_samples = []
        processed_count = 0
        skipped_count = 0
        folder_stats = {}
        
        for pdf_path, folder_name in all_pdf_files:
            if folder_name not in folder_stats:
                folder_stats[folder_name] = {'processed': 0, 'skipped': 0}
            logger.info(f"\nProcessing: {folder_name}/{pdf_path.name}")
            
            # Extract text from PDF
            extraction_result = self.extract_text_from_pdf(pdf_path)
            
            if not extraction_result['success']:
                logger.warning(f"Skipping {folder_name}/{pdf_path.name} due to extraction error")
                skipped_count += 1
                folder_stats[folder_name]['skipped'] += 1
                continue
            
            pdf_text = extraction_result['text_content']
            
            # Find matching ground truth - match by filename or quotation_id
            # Try multiple matching strategies
            pdf_stem = pdf_path.stem  # filename without extension
            
            # Strategy 1: Extract quotation_id from filename (e.g., "quote_106094.pdf" -> "106094")
            quotation_id_match = re.search(r'\d+', pdf_stem)
            
            matching_rows = pd.DataFrame()
            
            if quotation_id_match:
                quotation_id = quotation_id_match.group()
                # Try exact match first
                matching_rows = ground_truth_df[
                    ground_truth_df['quotation_id'].astype(str) == quotation_id
                ]
            
            # Strategy 2: If no exact match, try contains
            if matching_rows.empty:
                matching_rows = ground_truth_df[
                    ground_truth_df['quotation_id'].astype(str).str.contains(pdf_stem, case=False, na=False, regex=False)
                ]
            
            # Strategy 3: Try file_name column if it exists
            if matching_rows.empty and 'file_name' in ground_truth_df.columns:
                matching_rows = ground_truth_df[
                    ground_truth_df['file_name'].astype(str).str.contains(pdf_stem, case=False, na=False, regex=False)
                ]
            
            if matching_rows.empty:
                logger.warning(f"No ground truth found for {folder_name}/{pdf_path.name} (tried quotation_id: {quotation_id_match.group() if quotation_id_match else 'N/A'})")
                skipped_count += 1
                folder_stats[folder_name]['skipped'] += 1
                continue
            
            logger.info(f"Found {len(matching_rows)} matching row(s) for {folder_name}/{pdf_path.name}")
            
            # Group by quotation_id and merge all rows for the same quotation
            # (in case multiple rows exist for same quotation with different parts)
            for quotation_id, group in matching_rows.groupby('quotation_id'):
                # If multiple rows for same quotation, we'll merge all parts
                # For now, just use the first row (since each row can have up to 3 parts)
                row = group.iloc[0]
                
                ground_truth_json = self.format_ground_truth_to_json(row)
                training_sample = self.create_training_sample(pdf_text, ground_truth_json)
                training_samples.append(training_sample)
                processed_count += 1
                folder_stats[folder_name]['processed'] += 1
                
                logger.info(f"✓ Created training sample for {folder_name}/quotation_{quotation_id} with {len(ground_truth_json['parts'])} part(s)")
        
        # Write to JSONL file
        logger.info(f"\n{'='*80}")
        logger.info(f"Writing {len(training_samples)} training samples to {self.output_path}")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"{'='*80}")
        logger.info(f"Dataset processing complete!")
        logger.info(f"✓ Processed: {processed_count} quotations")
        logger.info(f"✗ Skipped: {skipped_count} files")
        logger.info(f"\nBreakdown by folder:")
        for folder_name, stats in folder_stats.items():
            logger.info(f"  {folder_name}: {stats['processed']} processed, {stats['skipped']} skipped")
        logger.info(f"\n📁 Output: {self.output_path}")
        logger.info(f"{'='*80}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Process RFQ quotation PDFs and ground truth XLSX to create training dataset'
    )
    
    parser.add_argument(
        '--pdf-folder',
        type=str,
        required=True,
        help='Path to parent folder containing subdirectories of quotation PDF files (each subdirectory = different template)'
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
        pdf_parent_folder=Path(args.pdf_folder),
        xlsx_path=Path(args.xlsx_path),
        output_path=Path(args.output_path)
    )
    
    processor.process_dataset()


if __name__ == '__main__':
    main()
