# PDF Data Extraction Pipeline

A Python-based pipeline for extracting data from PDF files containing agricultural trial data, specifically designed for Landessortenversuch (LSV) reports.

## Features

- Extracts data from PDF files containing agricultural trial data
- Processes multiple PDF files in batch
- Handles different table formats and structures
- Transforms data into a standardized format
- Saves extracted data to Excel files
- Provides detailed logging with colorful output

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-to-data
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your PDF files in the `input` directory

2. Run the pipeline:
```bash
python main.py lsv
```

Optional arguments:
- `--input-dir`: Directory containing input PDF files (default: "input")
- `--output-dir`: Directory for output Excel files (default: "output")

Example:
```bash
python main.py lsv --input-dir my_pdfs --output-dir results
```

## Output Format

The pipeline generates Excel files with the following columns:
- Year: The year of the trial
- Trial: Trial identifier (e.g., "2023_whw_de_prt_lsv")
- Trait: The measured trait (e.g., "Kornertrag absolut")
- Variety: The variety name
- Location: The trial location
- Treatment: The treatment applied (e.g., "St 1", "St 2")
- Value: The measured value

## Project Structure

- `main.py`: Main script for running the pipeline
- `pdf_processor.py`: Module for processing PDF files
- `data_transformer.py`: Module for transforming extracted data
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Logging

The pipeline uses the Rich library to provide colorful and informative logging output. Log messages include:
- Information about the processing status
- Warnings about skipped files or tables
- Errors that occur during processing