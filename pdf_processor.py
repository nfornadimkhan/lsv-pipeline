"""
Module for processing PDF files and extracting tables.
"""
import pdfplumber
from pdfplumber.page import Page # Ensure Page is imported
from collections import defaultdict
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from logging_config import get_logger
import re

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

    def _reconstruct_vertical_text(self, chars: List[Dict]) -> str:
        """
        Reconstruct vertical text from characters.
        
        Args:
            chars: List of character dictionaries with position information
        
        Returns:
            Reconstructed text string
        """
        if not chars:
            return ""

        # Sort characters by vertical position (top to bottom)
        chars = sorted(chars, key=lambda x: (-x['top'], x['x0']))
        
        # Group characters that are approximately at the same x-position
        x_groups = {}
        current_x = None
        x_tolerance = 1  # Tolerance for x-position grouping
        
        for char in chars:
            x_pos = char['x0']
            if current_x is None or abs(x_pos - current_x) > x_tolerance:
                current_x = x_pos
            
            if current_x not in x_groups:
                x_groups[current_x] = []
            x_groups[current_x].append(char)
        
        # Sort groups by x-position
        sorted_groups = sorted(x_groups.items(), key=lambda x: x[0])
        
        # Reconstruct text from each group
        text_parts = []
        for _, group in sorted_groups:
            # Sort characters in group by vertical position (top to bottom)
            sorted_chars = sorted(group, key=lambda x: x['top'])
            text_parts.append(''.join(char['text'] for char in sorted_chars))
        
        # Join all parts with proper handling of special characters
        result = ''
        for i, part in enumerate(text_parts):
            if i > 0:
                # Add space before opening bracket or after closing bracket
                if part.startswith('(') or text_parts[i-1].endswith(')'):
                    result += ' '
                # Add hyphen between parts that form a compound word
                elif part.startswith('-') or text_parts[i-1].endswith('-'):
                    result += ''
                else:
                    result += ' '
            result += part
        
        return result

    def _get_vertical_labels_from_page(self, page: Page) -> List[str]:
        """Extract vertical text labels from a page using both word and character-based methods."""
        logger.debug(f"Starting vertical label extraction for page {page.page_number}")
        
        # Get all words from the page
        words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
        
        # Filter for likely vertical labels (height > width by a significant margin)
        vertical_words = [w for w in words if w['height'] > w['width'] * 2]
        
        # Dictionary to store reconstructed vertical labels
        location_dict = {}
        
        # Get all characters from the page 
        chars = page.chars
        
        # Filter potential vertical label characters (ones in vertical words)
        for vword in vertical_words:
            x_center = (vword['x0'] + vword['x1']) / 2
            y_center = (vword['top'] + vword['bottom']) / 2
            
            # Find all chars that fall within this vertical word's bounds
            word_chars = [c for c in chars if 
                         (vword['x0'] <= c['x0'] <= vword['x1'] or
                          vword['x0'] <= c['x1'] <= vword['x1']) and
                         vword['top'] <= c['top'] <= vword['bottom']]
            
            # Skip if no characters found
            if not word_chars:
                continue
                
            # Sort chars by vertical position (top to bottom)
            word_chars.sort(key=lambda c: c['top'])
            
            # Reconstruct the vertical text
            text = ''.join([c['text'] for c in word_chars])
            
            # Clean up the text and store it
            if text and len(text) > 2:  # Min 3 chars for valid location
                # Clean up the text: remove excess spaces, normalize whitespace
                clean_text = re.sub(r'\s+', ' ', text).strip()
                
                # Store in dictionary with x-position as key to handle multiple columns
                x_key = int(x_center)
                if x_key not in location_dict:
                    location_dict[x_key] = []
                location_dict[x_key].append(clean_text)

        # Additional processing for known location patterns
        known_locations = {
            'kerpen': 'Kerpen-Buir',
            'erkelenz': 'Erkelenz-Venrath',
            'düsse': 'Haus Düsse (Ostingh.)',
            'haus': 'Haus Düsse (Ostingh.)', 
            'lage': 'Lage-Heiden',
            'heiden': 'Lage-Heiden',
            'holstein': 'Blomberg-Holstenhöfen',
            'blomberg': 'Blomberg-Holstenhöfen',
            'warstein': 'Warstein-Allagen',
            'allagen': 'Warstein-Allagen',
            'greve': 'Greven',
            'mittelwert': 'Mittelwert'
        }
        
        # Final list of reconstructed vertical labels
        vertical_labels = []
        
        # Process each column of vertical text
        for x_pos in sorted(location_dict.keys()):
            column_texts = location_dict[x_pos]
            
            for text in column_texts:
                # Check if the text matches a known location pattern
                text_lower = text.lower()
                
                matched = False
                for pattern, replacement in known_locations.items():
                    if pattern in text_lower:
                        vertical_labels.append(replacement)
                        matched = True
                        break
                        
                if not matched:
                    vertical_labels.append(text)
        
        if vertical_labels:
            logger.debug(f"Found {len(vertical_labels)} vertical labels on page {page.page_number}")
            logger.debug(f"Vertical labels: {vertical_labels}")
        
        return vertical_labels

    def extract_tables(self, pdf_path: Path, config_manager: Any = None) -> List[Dict[str, Any]]:
        """Extract tables from a PDF file."""
        logger.info(f"[cyan]Processing PDF: {pdf_path}[/cyan]")
        tables_for_transformer = []
        
        try:
            # Create output directory if it doesn't exist
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            with pdfplumber.open(pdf_path) as pdf:
                # Get configured pages
                pages_to_process_configs = []
                if config_manager:
                    pages_to_process_configs = config_manager.get_pages_to_process(pdf_path.name)
                    if not pages_to_process_configs:
                        logger.info(f"[yellow]No pages configured for {pdf_path.name}, skipping file[/yellow]")
                        return []
                    configured_page_numbers = [p['number'] for p in pages_to_process_configs]
                    logger.info(f"[cyan]Found {len(pages_to_process_configs)} configured pages to process: {configured_page_numbers}[/cyan]")
                else:
                    logger.warning(f"[yellow]No config_manager provided for {pdf_path.name}, skipping file[/yellow]")
                    return []
            
                # Process only configured pages
                for page_config in pages_to_process_configs:
                    page_num = page_config['number']
                    logger.info(f"[cyan]Processing configured page {page_num}[/cyan]")
                    
                    # Skip if page number is out of range
                    if not (1 <= page_num <= len(pdf.pages)):
                        logger.warning(f"[yellow]Page number {page_num} is out of range for PDF {pdf_path.name} with {len(pdf.pages)} pages. Skipping.[/yellow]")
                        continue
                        
                    # Get page configuration
                    table_type_from_config = page_config.get('table_type')
                    treatment_from_config = page_config.get('treatment')
                    trait_from_config = page_config.get('trait')
                    
                    # Only process this specific page
                    page = pdf.pages[page_num - 1]  # Convert to 0-based index
                    
                    # Extract text and tables only for this page
                    page_vertical_labels = self._get_vertical_labels_from_page(page)
                    if page_vertical_labels:
                        logger.info(f"[green]Found vertical labels on page {page_num}: {page_vertical_labels}[/green]")
                
                    # Process tables on this page
                    text_for_year_extraction = page.extract_text() or ""
                    
                    # Year extraction logic
                    current_year = None
                    if config_manager:
                        current_year = config_manager.get_pdf_year(pdf_path.name)
                    if not current_year: # Fallback if not in PDF-level config or no config_manager
                        current_year = self._extract_year(text_for_year_extraction, pdf_path, config_manager) # _extract_year can also use page-specific config if adapted
                    
                    if current_year is None:
                        logger.warning(f"[yellow]Could not determine year for page {page_num} of {pdf_path.name}. Skipping page tables.[/yellow]")
                        # continue # Or decide if tables can be processed without a year

                    # Extract vertical labels for the current page
                    page_vertical_labels = self._get_vertical_labels_from_page(page)
                    
                    try:
                        # pdfplumber.page.extract_tables() returns a list of tables,
                        # where each table is a list of rows, and each row is a list of cells.
                        extracted_page_tables = page.extract_tables()
                        logger.debug(f"[green]Page {page_num}: Extracted {len(extracted_page_tables)} raw tables structures[/green]")
                    except Exception as e:
                        logger.warning(f"[yellow]Page {page_num}: Error extracting raw tables structures: {str(e)}[/yellow]")
                        continue # Skip this page if table extraction fails
                    
                    for table_idx, raw_table_data in enumerate(extracted_page_tables):
                        if not raw_table_data:
                            logger.debug(f"Page {page_num}, Raw Table {table_idx}: Empty table data, skipping.")
                            continue
                        
                        # Determine table_type for this specific table if not globally set for page
                        # This might involve inspecting raw_table_data or relying on config
                        current_table_type = table_type_from_config # Use page-level config for now

                        if not current_table_type:
                            # Attempt to infer table type if not specified in config
                            # For now, if not specified, we might not know how to process it.
                            logger.warning(f"Page {page_num}, Raw Table {table_idx}: table_type not specified in config. Cannot determine processing method.")
                            continue # Or apply a default relevance check

                        table_dict_for_transformer = {
                            'page': page_num,
                            'table_num_on_page': table_idx,
                            'year': current_year,
                            'table_type': current_table_type,
                            'treatment': treatment_from_config,  # This needs to be handled differently
                            'source': pdf_path.name,
                            'trait': trait_from_config,
                            'raw_table_data': raw_table_data
                        }

                        if current_table_type == "relative":
                            # For relative tables, find the "rel. 100" row to confirm relevance and get its data
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
                                table_dict_for_transformer['page_vertical_labels'] = page_vertical_labels
                                tables_for_transformer.append(table_dict_for_transformer)
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Identified as relevant 'relative' table. Added for transformation.")
                            else:
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Configured as 'relative' but no 'rel. 100' row found. Skipping.")
                        
                        elif current_table_type == "absolute":
                            # For absolute tables, use _is_relevant_table or a similar check
                            headers = raw_table_data[0] if raw_table_data else []
                            rows = raw_table_data[1:] if len(raw_table_data) > 1 else []
                            if self._is_relevant_table(headers, rows, current_table_type):
                                # Handle "both" treatment type by creating two table dictionaries
                                if treatment_from_config and treatment_from_config.lower() == "both":
                                    # Create intensive treatment table
                                    intensive_dict = table_dict_for_transformer.copy()
                                    intensive_dict['treatment'] = 'intensive'
                                    intensive_dict['headers'] = headers
                                    intensive_dict['rows'] = rows
                                    tables_for_transformer.append(intensive_dict)
                                    logger.debug(f"Page {page_num}, Raw Table {table_idx}: Added 'intensive' treatment table.")

                                    # Create extensive treatment table
                                    extensive_dict = table_dict_for_transformer.copy()
                                    extensive_dict['treatment'] = 'extensive'
                                    extensive_dict['headers'] = headers
                                    extensive_dict['rows'] = rows
                                    tables_for_transformer.append(extensive_dict)
                                    logger.debug(f"Page {page_num}, Raw Table {table_idx}: Added 'extensive' treatment table.")
                                else:
                                    # Original behavior for non-"both" treatment
                                    table_dict_for_transformer['headers'] = headers
                                    table_dict_for_transformer['rows'] = rows
                                    tables_for_transformer.append(table_dict_for_transformer)
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Identified as relevant 'absolute' table. Added for transformation.")
                            else:
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Configured as 'absolute' but deemed not relevant. Skipping.")
                        else:
                            logger.warning(f"Page {page_num}, Raw Table {table_idx}: Unknown or unhandled table_type '{current_table_type}'. Skipping.")
                            
            logger.info(f"[green]Extracted {len(tables_for_transformer)} tables for further transformation from {pdf_path.name}[/green]")
            
            # Return the tables without trying to save them here
            return tables_for_transformer
            
        except Exception as e:
            logger.error(f"[red]Error processing PDF {pdf_path}: {str(e)}[/red]")
            return []
# ... rest of PDFProcessor, including _extract_year and _is_relevant_table ...
# Ensure _is_relevant_table is defined correctly
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
            
        # This function is now primarily for 'absolute' tables or as a generic check if type is unknown
        if table_type == 'absolute':
            # Example: an absolute table needs more than 2 columns and some content beyond headers
            return len(headers) > 2 and any(str(h).strip() for h in headers[1:])
        
        # For 'relative' tables, relevance is now checked by finding "rel. 100" row directly in extract_tables.
        # If table_type is None or other, apply a generic fallback.
        if table_type is None or table_type not in ['relative', 'absolute']:
             return len(headers) > 1 and len(rows) > 0 # Basic check: has headers and at least one data row
        
        return False # Default if type is known but not 'absolute' (e.g. 'relative' handled elsewhere)

    def reconstruct_vertical_text(self, text_from_cell: str) -> Optional[str]:
        """
        Reconstruct vertical text from cell content.
        
        Args:
            text_from_cell: String containing vertically stacked characters
            
        Returns:
            Properly reconstructed location name or None if invalid
        """
        if not text_from_cell:
            return None
        
        # Clean and normalize the input text
        text = str(text_from_cell).strip()
        
        # Split text by newlines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return None
            
        # Handle common patterns based on content analysis
        if "Kerpen" in text or "Buir" in text:
            return "Kerpen-Buir"
        elif "Erkelenz" in text or "Venrath" in text:
            return "Erkelenz-Venrath"
        elif "Haus" in text or "Düsse" in text or "Ostingh" in text:
            return "Haus Düsse (Ostingh.)"
        elif "Lage" in text or "Heiden" in text:
            return "Lage-Heiden"
        elif "Blomberg" in text or "Holstein" in text:
            return "Blomberg-Holstenhöfen"
        elif "Warstein" in text or "Allagen" in text:
            return "Warstein-Allagen"
        elif "Greven" in text:
            return "Greven"
        elif "Mittelwert" in text:
            return "Mittelwert"
        elif any("elttiM" in line for line in lines):
            return "Mittelwert"
            
        # For general case, join the lines with proper spacing
        result = ' '.join(lines)
        
        # Replace common pattern artifacts
        result = result.replace("t r e w l e t t i M", "Mittelwert")
        
        return result