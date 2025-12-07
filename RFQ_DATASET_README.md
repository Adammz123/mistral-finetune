# RFQ Dataset Processor

This tool processes quotation PDFs and ground truth Excel data to create training datasets in the ultrachat JSONL format for fine-tuning LLMs.

## Features

- **Automatic Template Discovery**: Automatically finds all subdirectories containing PDFs in different formats
- **Multi-template Support**: Processes multiple PDF templates for the same quotations
- **Structured Output**: Generates training data in ultrachat JSONL format
- **Ground Truth Mapping**: Maps PDFs to ground truth data from Excel files

## Directory Structure

Your data should be organized as follows:

```
project/
├── quotations/                    # Parent folder
│   ├── template1/                 # Different PDF format/template
│   │   ├── quote_105826.pdf
│   │   ├── quote_106044.pdf
│   │   └── quote_106094.pdf
│   ├── template2/                 # Another PDF format/template
│   │   ├── quote_105826.pdf
│   │   ├── quote_106044.pdf
│   │   └── quote_106094.pdf
│   └── template3/                 # Yet another format
│       ├── quote_105826.pdf
│       ├── quote_106044.pdf
│       └── quote_106094.pdf
└── quote_data.xlsx                # Ground truth data
```

**Note**: Each subdirectory contains the same quotations but in different PDF formats/styles. This helps the model learn to extract information regardless of the PDF format.

## Excel File Format

The Excel file (`quote_data.xlsx`) should contain the following columns:

### Required Columns:
- `quotation_id`: Unique identifier for each quotation
- `supplier_name`: Supplier name
- `valid_until_date`: Quotation validity date
- `currency`: Currency code (e.g., USD, EUR)
- `part_number`: Part number
- `part_description`: Description of the part
- `manufacturer`: Manufacturer name
- `unit_of_measure`: Unit (e.g., EA, BAG)
- `quantity`: Quantity
- `price_per_unit`: Price per unit
- `total_price`: Total price
- `lead_time_days_weeks`: Lead time information

### Optional Columns:
- `incoterms`: Incoterms
- `payment_terms`: Payment terms
- `warranty_terms`: Warranty terms
- `additional_notes`: Additional notes
- `tariff_charge`: Tariff information
- `Vendor Part Number`: Vendor's part number

### Multiple Parts per Quotation:
The Excel file supports up to 3 parts per quotation:
- Part 1: `part_number`, `part_description`, `quantity`, `price_per_unit`, etc.
- Part 2: `part_number2`, `part_description2`, `quantity2`, `price_per_unit2`, etc.
- Part 3: `part_number3`, `part_description3`, `quantity3`, `price_per_unit3`, etc.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Or install specific packages:
```bash
pip install pdfplumber openpyxl pandas
```

## Usage

### Method 1: Using the Helper Script

```bash
python utils/create_rfq_training_data.py \
    --pdf-folder /path/to/quotations \
    --xlsx-path /path/to/quote_data.xlsx \
    --output-path /path/to/output/training_data.jsonl
```

### Method 2: Direct Module Usage

```bash
python -m finetune.data.rfq_dataset \
    --pdf-folder /path/to/quotations \
    --xlsx-path /path/to/quote_data.xlsx \
    --output-path /path/to/output/training_data.jsonl
```

### Method 3: Python Script

```python
from pathlib import Path
from finetune.data.rfq_dataset import RFQDatasetProcessor

processor = RFQDatasetProcessor(
    pdf_parent_folder=Path('data/quotations'),
    xlsx_path=Path('data/quote_data.xlsx'),
    output_path=Path('output/training_data.jsonl')
)

processor.process_dataset()
```

## Output Format

The tool generates a JSONL file where each line is a training sample in ultrachat format:

```json
{
  "prompt": "Extract the following information from the quotation below...",
  "prompt_id": "abc123...",
  "messages": [
    {
      "content": "Extract the following information...",
      "role": "user"
    },
    {
      "content": "{\"supplier_name\": \"Glenair Inc\", \"quotation_id\": \"105826\", ...}",
      "role": "assistant"
    }
  ]
}
```

The assistant's response contains the structured JSON with:
- Quotation metadata (supplier, ID, dates, terms, etc.)
- Array of parts with pricing information
- Price breaks for each part

## Example

```bash
# Process quotations from data/example/quotations folder
python utils/create_rfq_training_data.py \
    --pdf-folder data/example/quotations \
    --xlsx-path "data/example/quote data.xlsx" \
    --output-path data/example/rfq_training_data.jsonl
```

## Troubleshooting

### No PDFs found
- Ensure your PDFs are in subdirectories, not directly in the parent folder
- Check that PDF files have `.pdf` extension

### No ground truth found
- Ensure the PDF filename contains the `quotation_id` from the Excel file
- Example: `quote_105826.pdf` matches `quotation_id: 105826` in Excel

### Missing dependencies
```bash
pip install pdfplumber openpyxl pandas
```

## Output Statistics

The tool provides detailed statistics:
- Total PDFs processed
- PDFs skipped (with reasons)
- Breakdown by template folder
- Total training samples generated

Example output:
```
================================================================================
Dataset processing complete!
✓ Processed: 150 quotations
✗ Skipped: 5 files

Breakdown by folder:
  template1: 50 processed, 2 skipped
  template2: 50 processed, 1 skipped
  template3: 50 processed, 2 skipped

📁 Output: data/example/rfq_training_data.jsonl
================================================================================
```

