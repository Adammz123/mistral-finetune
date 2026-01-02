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
        --pdf-folder /path/to/quotations \\
        --xlsx-path /path/to/ground_truth.xlsx \\
        --output-path /path/to/output/rfq_training_data.jsonl

Directory structure:
    /path/to/quotations/
        ├── D12/                    # Folder name maps to supplier (D12 -> Digikey)
        │   ├── Q format D12-1.pdf
        │   ├── Q format D12-2.pdf
        ├── G12/                    # Folder name maps to supplier (G12 -> Glenair, Inc)
        │   ├── Q format G12-1.pdf
        │   ├── Q format G12-2.pdf
        └── H12a/                   # Folder name maps to supplier (H12a -> Heilind)
            ├── Q format H12a-1.pdf
            └── Q format H12a-2.pdf
    
    The script will automatically discover all subdirectories containing PDFs.
    Each subdirectory name maps to a supplier name (configured in the code).
    Supplier names are determined from folder names, not from the Excel file.

Expected XLSX structure:
    The XLSX file should contain columns matching the JSON structure:
    - Filename reference: Reference number (1, 2, 3, etc.) that matches the number
      in the PDF filename after the hyphen (e.g., "Q format D12-11" matches reference 11)
    - quotation_id: Unique identifier for the quotation
    - supplier_name: Name of the supplier (NOTE: This is overridden by folder-to-supplier mapping.
      The supplier name is determined from the folder name, not from this column)
    - valid_until_date: Validity date of the quotation
    - incoterms: Incoterms (optional)
    - payment_terms: Payment terms (optional)
    - warranty_terms: Warranty terms (optional)
    - currency: Currency code (e.g., USD, EUR)
    - additional_notes: Additional notes about the quotation
    
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
        help='Path to parent folder containing subdirectories of quotation PDFs (automatically discovers all subdirectories)'
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
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of worker threads for parallel processing (default: CPU count + 4, max 32)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    pdf_parent_folder = Path(args.pdf_folder)
    xlsx_path = Path(args.xlsx_path)
    output_path = Path(args.output_path)
    
    if not pdf_parent_folder.exists():
        print(f"Error: PDF parent folder does not exist: {pdf_parent_folder}")
        sys.exit(1)
    
    if not xlsx_path.exists():
        print(f"Error: XLSX file does not exist: {xlsx_path}")
        sys.exit(1)
    
    # Create processor and run (it will auto-discover subdirectories and config file)
    print("Initializing RFQ Dataset Processor...")
    print(f"Parent folder: {pdf_parent_folder}")
    
    processor = RFQDatasetProcessor(
        pdf_parent_folder=pdf_parent_folder,
        xlsx_path=xlsx_path,
        output_path=output_path
    )
    
    print("Processing dataset...")
    processor.process_dataset(max_workers=args.max_workers)
    
    print(f"\n✓ Training data successfully created at: {output_path}")
    print(f"✓ You can now use this file for fine-tuning")


if __name__ == '__main__':
    main()

