"""
Module for processing PDF files and extracting tables.
"""
import pdfplumber
from pdfplumber.page import Page
from collections import defaultdict
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from logging_config import get_logger

# Suppress all pdfminer logging
logging.getLogger('pdfminer').setLevel(logging.ERROR)

# Suppress pdfminer warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pdfminer')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pdfminer')

logger = get_logger("pdf_pipeline")

class PDFProcessor:
    def __init__(self):
        """Initialize PDF processor."""
        logger.debug("[cyan]Initializing PDFProcessor[/cyan]")

    def _clean_cell_text(self, cell_text: Any) -> Any:
        """Clean up cell text by removing unnecessary newlines while preserving the content."""
        if cell_text is None:
            return None
        
        text = str(cell_text)
        
        # If the text contains newlines, it might be vertical text
        if '\n' in text:
            # Check if this looks like vertical text (single characters per line)
            lines = text.split('\n')
            if all(len(line.strip()) <= 3 for line in lines if line.strip()):
                # This looks like vertical text - keep it as is for now
                # It will be handled by the transformer
                return text
            else:
                # Regular multi-line text - join with space
                return ' '.join(line.strip() for line in lines if line.strip())
        
        return text

    def _clean_table_data(self, raw_table_data: List[List[Any]]) -> List[List[Any]]:
        """Clean up the raw table data by processing each cell."""
        cleaned_table = []
        for row in raw_table_data:
            if row:
                cleaned_row = [self._clean_cell_text(cell) for cell in row]
                cleaned_table.append(cleaned_row)
            else:
                cleaned_table.append(row)
        return cleaned_table

    def extract_tables(self, pdf_path: Path, config_manager: Any = None) -> List[Dict[str, Any]]:
        """Extract tables from a PDF file."""
        logger.info(f"[cyan]Processing PDF: {pdf_path}[/cyan]")
        tables_for_transformer = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process_configs = []
                if config_manager:
                    pages_to_process_configs = config_manager.get_pages_to_process(pdf_path.name)
                    logger.info(f"[cyan]Found {len(pages_to_process_configs)} configured pages to process[/cyan]")
                
                if not pages_to_process_configs:
                    logger.info("[cyan]No pages configured, processing all pages[/cyan]")
                    # Create a default config for each page if none provided
                    pages_to_process_configs = [{'number': i + 1, 'table_type': None, 'treatment': None, 'trait': None} for i in range(len(pdf.pages))]
                
                for page_config in pages_to_process_configs:
                    page_num = page_config['number']
                    table_type_from_config = page_config.get('table_type') 
                    treatment_from_config = page_config.get('treatment')
                    trait_from_config = page_config.get('trait')
                    
                    if not (1 <= page_num <= len(pdf.pages)):
                        logger.warning(f"[yellow]Page number {page_num} is out of range for PDF {pdf_path.name} with {len(pdf.pages)} pages. Skipping.[/yellow]")
                        continue
                        
                    page = pdf.pages[page_num - 1]
                    text_for_year_extraction = page.extract_text() or ""
                    
                    # Year extraction logic
                    current_year = None
                    if config_manager:
                        current_year = config_manager.get_pdf_year(pdf_path.name)
                    if not current_year:
                        current_year = self._extract_year(text_for_year_extraction, pdf_path, config_manager)
                    
                    if current_year is None:
                        logger.warning(f"[yellow]Could not determine year for page {page_num} of {pdf_path.name}. Skipping page tables.[/yellow]")

                    try:
                        extracted_page_tables = page.extract_tables()
                        logger.debug(f"[green]Page {page_num}: Extracted {len(extracted_page_tables)} raw tables structures[/green]")
                    except Exception as e:
                        logger.warning(f"[yellow]Page {page_num}: Error extracting raw tables structures: {str(e)}[/yellow]")
                        continue
                    
                    for table_idx, raw_table_data in enumerate(extracted_page_tables):
                        if not raw_table_data:
                            logger.debug(f"Page {page_num}, Raw Table {table_idx}: Empty table data, skipping.")
                            continue
                        
                        # Clean the table data
                        raw_table_data = self._clean_table_data(raw_table_data)
                        
                        current_table_type = table_type_from_config

                        if not current_table_type:
                            logger.warning(f"Page {page_num}, Raw Table {table_idx}: table_type not specified in config. Cannot determine processing method.")
                            continue

                        table_dict_for_transformer = {
                            'page': page_num,
                            'table_num_on_page': table_idx,
                            'year': current_year,
                            'table_type': current_table_type,
                            'treatment': treatment_from_config,
                            'source': pdf_path.name,
                            'trait': trait_from_config,
                            'raw_table_data': raw_table_data
                        }

                        if current_table_type == "relative":
                            reference_row_content = None
                            reference_row_original_idx = -1
                            for row_content_idx, row_list in enumerate(raw_table_data):
                                if row_list and any(cell_content and "rel. 100" in str(cell_content) for cell_content in row_list):
                                    reference_row_content = row_list
                                    reference_row_original_idx = row_content_idx
                                    break
                            
                            if reference_row_content is not None:
                                table_dict_for_transformer['reference_row_content'] = reference_row_content
                                table_dict_for_transformer['reference_row_original_idx_in_table'] = reference_row_original_idx
                                tables_for_transformer.append(table_dict_for_transformer)
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Identified as relevant 'relative' table. Added for transformation.")
                            else:
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Configured as 'relative' but no 'rel. 100' row found. Skipping.")
                        
                        elif current_table_type == "absolute":
                            headers = raw_table_data[0] if raw_table_data else []
                            rows = raw_table_data[1:] if len(raw_table_data) > 1 else []
                            if self._is_relevant_table(headers, rows, current_table_type):
                                table_dict_for_transformer['headers'] = headers
                                table_dict_for_transformer['rows'] = rows
                                tables_for_transformer.append(table_dict_for_transformer)
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Identified as relevant 'absolute' table. Added for transformation.")
                            else:
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Configured as 'absolute' but deemed not relevant. Skipping.")
                        else:
                            logger.warning(f"Page {page_num}, Raw Table {table_idx}: Unknown or unhandled table_type '{current_table_type}'. Skipping.")
                            
            logger.info(f"[green]Extracted {len(tables_for_transformer)} tables for further transformation from {pdf_path.name}[/green]")
            return tables_for_transformer
            
        except Exception as e:
            logger.error(f"[red]Error processing PDF {pdf_path}: {str(e)}[/red]")
            return []

    def _extract_year(self, text: str, pdf_path: Path, config_manager: Any = None) -> Optional[int]:
        """
        Extract year from text.
        
        Args:
            text: Text to extract year from
            pdf_path: Path to the PDF file
            config_manager: Configuration manager instance
            
        Returns:
            Extracted year or None if not found
        """
        # First try to get year from config
        if config_manager:
            config_year = config_manager.get_pdf_year(pdf_path.name)
            if config_year:
                return config_year
                
        # Fallback to extracting from text
        year_pattern = r'\b(19|20)\d{2}\b'
        match = re.search(year_pattern, text)
        if match:
            return int(match.group())
        return None

    def _is_relevant_table(self, headers: List[Any], rows: List[List[Any]], table_type: Optional[str]) -> bool:
        """Check if table contains relevant data."""
        if not headers or not rows:
            return False
            
        if table_type == 'absolute':
            return len(headers) > 2 and any(str(h).strip() for h in headers[1:])
        
        if table_type is None or table_type not in ['relative', 'absolute']:
             return len(headers) > 1 and len(rows) > 0
        
        return False