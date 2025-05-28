"""
Module for processing PDF files and extracting tables.
"""
import pdfplumber # Ensure pdfplumber is imported if not already at the top level of imports
from pdfplumber.page import Page
from collections import defaultdict
import logging # Ensure logging is imported
import warnings # Ensure warnings is imported
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
    def _get_vertical_labels_from_page(self, page: Page) -> List[str]:
        """
        Extracts vertical text labels from a given PDF page.
        Tries two methods:
        1. Character-level analysis for non-upright text.
        2. Word-level analysis with vertical grouping as a fallback.
        Filters out common non-location labels like "Mittelwert".
        """
        logger.debug(f"Page {page.page_number}: Starting vertical label extraction.")
        
        extracted_labels = []
        min_chars_for_label = 3  # Heuristic for character-based labels
        min_len_for_word_label = 3 # Heuristic for word-based labels

        # --- Attempt 1: Using non-upright characters (existing method) ---
        logger.debug(f"Page {page.page_number}: Attempt 1 - Using non-upright character analysis.")
        chars = page.chars
        # Default to True for 'upright' if key is missing, so non-upright means 'upright' is False.
        vchars = [c for c in chars if not c.get("upright", True)] 

        if vchars:
            char_buckets = defaultdict(list)
            for ch in vchars:
                char_buckets[round(ch["x0"])].append(ch)
            
            sorted_x_coords_chars = sorted(char_buckets.keys())
            for x_coord in sorted_x_coords_chars:
                sorted_chars_in_bucket = sorted(char_buckets[x_coord], key=lambda c: c["top"])
                word = "".join(ch["text"] for ch in sorted_chars_in_bucket)
                
                if len(word) >= min_chars_for_label and not word.isnumeric():
                    # Explicitly filter out "Mittelwert"
                    if "mittelwert" not in word.lower().replace(" ", ""):
                        extracted_labels.append(word)
                    else:
                        logger.debug(f"Page {page.page_number} (Attempt 1): Filtered out 'Mittelwert' (or similar): '{word}'")
                elif len(word) < min_chars_for_label and word.strip(): # Log skipped short non-empty text
                    logger.debug(f"Page {page.page_number} (Attempt 1): Skipped short vertical text: '{word}' at x0={x_coord}")
            
            if extracted_labels:
                logger.info(f"Page {page.page_number} (Attempt 1): Extracted {len(extracted_labels)} labels using non-upright chars: {extracted_labels}")
                return extracted_labels
            else:
                logger.info(f"Page {page.page_number} (Attempt 1): No labels found using non-upright character analysis, though non-upright chars were present.")
        else:
            logger.debug(f"Page {page.page_number} (Attempt 1): No non-upright characters found.")

        # --- Attempt 2: Using word extraction and vertical grouping (if Attempt 1 failed or found nothing) ---
        logger.info(f"Page {page.page_number}: Attempt 2 - Using word extraction and vertical grouping.")
        
        # X_TOLERANCE: How close words need to be horizontally to be in the same bucket.
        # Smaller values mean stricter alignment.
        X_TOLERANCE = 2  # points
        # MAX_VERTICAL_WORD_GAP: Max vertical distance (bottom of word A to top of word B)
        # for words in the same x-bucket to be considered part of the same label.
        MAX_VERTICAL_WORD_GAP = 5 # points (heuristic, might need tuning)

        words_on_page = page.extract_words(
            keep_blank_chars=False, 
            use_text_flow=True, 
            extra_attrs=["x0", "x1", "top", "bottom"] # 'text' is included by default
        )

        if not words_on_page:
            logger.debug(f"Page {page.page_number} (Attempt 2): No words extracted from page.")
            return [] # Return empty list if both attempts fail

        word_buckets = defaultdict(list)
        for word_obj in words_on_page:
            # Group words by their horizontal position, rounded to the nearest X_TOLERANCE band
            bucket_key = round(word_obj["x0"] / X_TOLERANCE) * X_TOLERANCE
            word_buckets[bucket_key].append(word_obj)
            
        alternative_labels = []
        sorted_x_bucket_keys = sorted(word_buckets.keys())

        for x_key in sorted_x_bucket_keys:
            words_in_bucket = sorted(word_buckets[x_key], key=lambda w: w["top"])
            
            if not words_in_bucket:
                continue

            current_label_parts = []
            last_word_bottom = 0

            for i, word_obj in enumerate(words_in_bucket):
                word_text = word_obj["text"]
                if not word_text.strip(): # Should be rare due to keep_blank_chars=False
                    continue

                # If there are parts for the current label and a significant vertical gap to the current word
                if current_label_parts and word_obj["top"] > last_word_bottom + MAX_VERTICAL_WORD_GAP:
                    # Finalize the previous label
                    formed_label = "".join(current_label_parts)
                    if len(formed_label) >= min_len_for_word_label and not formed_label.isnumeric():
                        if "mittelwert" not in formed_label.lower().replace(" ", ""):
                            alternative_labels.append(formed_label)
                        else:
                            logger.debug(f"Page {page.page_number} (Attempt 2): Filtered 'Mittelwert': '{formed_label}' (gap logic)")
                    elif len(formed_label) < min_len_for_word_label and formed_label.strip():
                         logger.debug(f"Page {page.page_number} (Attempt 2): Skipped short text (gap logic): '{formed_label}'")
                    current_label_parts = [word_text] # Start new label
                else:
                    # Append to current label
                    current_label_parts.append(word_text)
                
                last_word_bottom = word_obj["bottom"]

            # Add the last accumulated label from the bucket
            if current_label_parts:
                formed_label = "".join(current_label_parts)
                if len(formed_label) >= min_len_for_word_label and not formed_label.isnumeric():
                    if "mittelwert" not in formed_label.lower().replace(" ", ""):
                        alternative_labels.append(formed_label)
                    else:
                        logger.debug(f"Page {page.page_number} (Attempt 2): Filtered 'Mittelwert': '{formed_label}' (end of bucket)")
                elif len(formed_label) < min_len_for_word_label and formed_label.strip():
                    logger.debug(f"Page {page.page_number} (Attempt 2): Skipped short text (end of bucket): '{formed_label}'")
        
        if alternative_labels:
            logger.info(f"Page {page.page_number} (Attempt 2): Extracted {len(alternative_labels)} labels using word grouping: {alternative_labels}")
            return alternative_labels
        else:
            logger.info(f"Page {page.page_number} (Attempt 2): No labels found using word grouping.")
            return [] # Return empty list if both attempts fail
        
    def extract_tables(self, pdf_path: Path, config_manager: Any = None) -> List[Dict[str, Any]]:
        """Extract tables from a PDF file."""
        logger.info(f"[cyan]Processing PDF: {pdf_path}[/cyan]")
        tables_for_transformer = [] # Renamed to avoid confusion
        # year = None # year is extracted per page or from config, not a single value for the whole PDF initially
        
        try:
            with pdfplumber.open(pdf_path) as pdf: # Removed pages=None, it's default
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
                    # Ensure table_type is fetched, default if necessary, or handle if None later
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
                            'treatment': treatment_from_config,
                            'source': pdf_path.name,
                            'trait': trait_from_config,
                            'raw_table_data': raw_table_data # Pass the entire extracted table
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
                                table_dict_for_transformer['page_vertical_labels'] = page_vertical_labels # Add extracted vertical labels
                                tables_for_transformer.append(table_dict_for_transformer)
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Identified as relevant 'relative' table. Added for transformation.")
                            else:
                                logger.debug(f"Page {page_num}, Raw Table {table_idx}: Configured as 'relative' but no 'rel. 100' row found. Skipping.")
                        
                        elif current_table_type == "absolute":
                            # For absolute tables, use _is_relevant_table or a similar check
                            headers = raw_table_data[0] if raw_table_data else []
                            rows = raw_table_data[1:] if len(raw_table_data) > 1 else []
                            if self._is_relevant_table(headers, rows, current_table_type): # Pass 'absolute'
                                # For absolute tables, 'headers' and 'rows' are more direct
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