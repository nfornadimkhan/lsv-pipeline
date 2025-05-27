"""
Module for processing PDF files and extracting tables.
"""
import pdfplumber
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from logging_config import get_logger
import warnings

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
        
    def extract_tables(self, pdf_path: Path, config_manager: Any = None) -> List[Dict[str, Any]]:
        """
        Extract tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            config_manager: Configuration manager instance
            
        Returns:
            List of extracted tables with metadata
        """
        logger.info(f"[cyan]Processing PDF: {pdf_path}[/cyan]")
        tables = []
        year = None
        
        try:
            # Open PDF with error handling
            try:
                with pdfplumber.open(pdf_path, pages=None) as pdf:
                    # Get pages to process from config
                    pages_to_process = []
                    if config_manager:
                        pages_to_process = config_manager.get_pages_to_process(pdf_path.name)
                        logger.info(f"[cyan]Found {len(pages_to_process)} configured pages to process[/cyan]")
                    
                    # If no pages configured, process all pages
                    if not pages_to_process:
                        logger.info("[cyan]No pages configured, processing all pages[/cyan]")
                        pages_to_process = [{'number': i} for i in range(1, len(pdf.pages) + 1)]
                    
                    for page_config in pages_to_process:
                        page_num = page_config['number']
                        table_type = page_config.get('table_type')
                        treatment = page_config.get('treatment')
                        
                        if page_num > len(pdf.pages):
                            logger.warning(f"[yellow]Page {page_num} not found in PDF[/yellow]")
                            continue
                            
                        try:
                            page = pdf.pages[page_num - 1]
                            text = page.extract_text() or ""
                            
                            # Extract year from first page
                            if page_num == 1:
                                year = self._extract_year(text, pdf_path, config_manager)
                                logger.info(f"[cyan]Extracted year: {year}[/cyan]")
                            
                            # Always get year from config manager for this PDF
                            year_from_config = config_manager.get_pdf_year(pdf_path.name) if config_manager else year
                            source_filename = pdf_path.name
                            
                            # Extract tables with error handling
                            try:
                                extracted_tables = page.extract_tables()
                                logger.debug(f"[green]Extracted {len(extracted_tables)} tables on page {page_num}[/green]")
                            except Exception as e:
                                logger.warning(f"[yellow]Error extracting tables from page {page_num}: {str(e)}[/yellow]")
                                continue
                            
                            for table_num, table in enumerate(extracted_tables, 1):
                                if not table:
                                    continue
                                    
                                try:
                                    headers = table[0]
                                    rows = table[1:]
                                    if not headers or not rows:
                                        continue
                                        
                                    # Special handling for kornertrag_relativ
                                    if table_type == "kornertrag_relativ":
                                        reference_row = None
                                        reference_row_idx = None
                                        for idx, row in enumerate(rows):
                                            if any(cell and "rel. 100" in str(cell) for cell in row):
                                                reference_row = row
                                                reference_row_idx = idx
                                                break
                                        if reference_row:
                                            tables.append({
                                                'page': page_num,
                                                'table_num': table_num,
                                                'full_table': [headers] + rows,
                                                'year': year_from_config,
                                                'table_type': 'kornertrag_relativ',
                                                'reference_row': reference_row,
                                                'reference_row_idx': reference_row_idx,
                                                'treatment': treatment,
                                                'source': source_filename
                                            })
                                        continue
                                        
                                    # Check if table contains relevant data
                                    if self._is_relevant_table(headers, rows, table_type):
                                        tables.append({
                                            'page': page_num,
                                            'table_num': table_num,
                                            'headers': headers,
                                            'rows': rows,
                                            'year': year_from_config,
                                            'treatment': treatment,
                                            'source': source_filename
                                        })
                                except Exception as e:
                                    logger.warning(f"[yellow]Error processing table {table_num} on page {page_num}: {str(e)}[/yellow]")
                                    continue
                                    
                        except Exception as e:
                            logger.warning(f"[yellow]Error processing page {page_num}: {str(e)}[/yellow]")
                            continue
                            
            finally:
                pdf.close()
                
            logger.info(f"[green]Extracted {len(tables)} relevant tables[/green]")
            return tables
            
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
            
    def _is_relevant_table(self, headers: List[str], rows: List[List[str]], table_type: Optional[str] = None) -> bool:
        """
        Check if table contains relevant data.
        
        Args:
            headers: Table headers
            rows: Table rows
            table_type: Expected table type from config
            
        Returns:
            True if table is relevant, False otherwise
        """
        # Skip tables with "Wertprüfung" in headers
        if any("Wertprüfung" in str(h) for h in headers):
            return False
            
        # If table type is specified in config, check for that type
        if table_type:
            if table_type == "kornertrag":
                return any("Kornertrag absolut" in str(h) for h in headers)
            elif table_type == "ertrage":
                return any("Erträge" in str(h) for h in headers) and any("Absoluter Ertrag" in str(h) for h in headers)
            return False
            
        # If no table type specified, check for both types
        return (
            any("Kornertrag absolut" in str(h) for h in headers) or
            (any("Erträge" in str(h) for h in headers) and any("Absoluter Ertrag" in str(h) for h in headers))
        ) 