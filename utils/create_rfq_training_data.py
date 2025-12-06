"""
Helper script to create RFQ training data from PDFs and ground truth XLSX.

This script demonstrates how to use the RFQDatasetProcessor to create
training data in the ultrachat format.
"""
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import finetune modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune.data.rfq_dataset import RFQDatasetProcessor


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create RFQ training data from PDFs and ground truth XLSX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python create_rfq_training_data.py \\
        --pdf-folder /path/to/quotation_pdfs \\
        --xlsx-path /path/to/ground_truth.xlsx \\
        --output-path /path/to/output/rfq_training_data.jsonl

Expected XLSX structure:
    The XLSX file should contain columns matching the JSON structure:
    - quotation_id: Unique identifier for the quotation
    - supplier_name: Name of the supplier
    - valid_until_date: Validity date of the quotation
    - incoterms: Incoterms (optional)
    - payment_terms: Payment terms (optional)
    - warranty_terms: Warranty terms (optional)
    - currency: Currency code (e.g., USD, EUR)
    - additional_notes: Additional notes about the quotation
    - file_name: PDF filename (for matching)
    
    Part information (one row per part or JSON in 'parts' column):
    - part_number: Part number
    - part_description: Description of the part
    - manufacturer: Manufacturer name
    - unit_of_measure: Unit of measure (e.g., EA, BAG)
    - lead_time_days_weeks: Lead time information
    - tariff_charge: Tariff information
    
    Price break information (one row per price break or JSON in 'price_breaks' column):
    - quantity: Quantity for this price break
    - minimum_order_quantity: Minimum order quantity
    - price_per_unit: Price per unit
    - total_price: Total price for this quantity
    
    Note: You can either have one row per price break, or use JSON strings
    in 'parts' and 'price_breaks' columns for complex structures.
        """
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
        help='Path to output JSONL file (will be created if it does not exist)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    pdf_folder = Path(args.pdf_folder)
    xlsx_path = Path(args.xlsx_path)
    output_path = Path(args.output_path)
    
    if not pdf_folder.exists():
        print(f"Error: PDF folder does not exist: {pdf_folder}")
        sys.exit(1)
    
    if not xlsx_path.exists():
        print(f"Error: XLSX file does not exist: {xlsx_path}")
        sys.exit(1)
    
    # Create processor and run
    print("Initializing RFQ Dataset Processor...")
    processor = RFQDatasetProcessor(
        pdf_folder=pdf_folder,
        xlsx_path=xlsx_path,
        output_path=output_path
    )
    
    print("Processing dataset...")
    processor.process_dataset()
    
    print(f"\n✓ Training data successfully created at: {output_path}")
    print(f"✓ You can now use this file for fine-tuning")


if __name__ == '__main__':
    main()

