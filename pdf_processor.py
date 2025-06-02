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

    def _get_vertical_labels_from_page(self, page: Page) -> List[str]:
        """Extract vertical text labels from a page using both non-upright character and word-based methods."""
        # Only process if this page is in the configured pages
        logger.debug(f"[cyan]Starting vertical label extraction for page {page.page_number}[/cyan]")
        
        extracted_labels = []
        
        # --- Attempt 1: Refined Character-level analysis for non-upright text ---
        logger.debug(f"Page {page.page_number}: Attempt 1 - Using refined non-upright character analysis.")
        
        min_len_chars = 3    # Minimum length for a character-based label
        tolerance_chars = 2  # Tolerance for grouping x-coordinates

        vchars = [c for c in page.chars if not c.get("upright", True)]
        logger.debug(f"Page {page.page_number} (Attempt 1): Total characters on page: {len(page.chars)}")
        logger.debug(f"Page {page.page_number} (Attempt 1): Found {len(vchars)} non-upright characters.")

        if vchars:
            char_buckets = defaultdict(list)
            logger.debug(f"Page {page.page_number} (Attempt 1): Bucketing non-upright characters with tolerance {tolerance_chars}...")
            for ch in vchars:
                # Round to nearest multiple of "tolerance_chars" for grouping
                key = round(ch["x0"] / tolerance_chars) * tolerance_chars
                char_buckets[key].append(ch)

            sorted_x_coords_chars = sorted(char_buckets.keys())
            if not sorted_x_coords_chars:
                logger.debug(f"Page {page.page_number} (Attempt 1): No character buckets formed.")

            for x_coord in sorted_x_coords_chars:
                chars_in_bucket = char_buckets[x_coord]
                logger.debug(f"Page {page.page_number} (Attempt 1): Bucket x_key={x_coord} has {len(chars_in_bucket)} chars.")
                
                sorted_chars_in_bucket = sorted(chars_in_bucket, key=lambda c: c["top"])
                word_raw = "".join(ch["text"] for ch in sorted_chars_in_bucket)
                word = word_raw.strip()
                logger.debug(f"Page {page.page_number} (Attempt 1): Formed raw word: '{word_raw}' (stripped: '{word}') from bucket {x_coord}.")
                
                if len(word) >= min_len_chars:
                    if not word.isnumeric(): # Added check to filter out purely numeric labels
                        # Explicitly filter out "Mittelwert" or similar common non-location terms
                        # Making this check case-insensitive and space-insensitive
                        if "mittelwert" not in word.lower().replace(" ", ""):
                            extracted_labels.append(word)
                            logger.debug(f"Page {page.page_number} (Attempt 1): Kept label: '{word}'.")
                        else:
                            logger.debug(f"Page {page.page_number} (Attempt 1): Filtered out 'Mittelwert' (or similar): '{word}'")
                    else:
                        logger.debug(f"Page {page.page_number} (Attempt 1): Word '{word}' is numeric, filtered out.")
                elif word: # Log if it's too short but not empty
                    logger.debug(f"Page {page.page_number} (Attempt 1): Word '{word}' (len {len(word)}) too short, min_len is {min_len_chars}.")
            
            if extracted_labels:
                logger.info(f"Page {page.page_number} (Attempt 1): Successfully extracted {len(extracted_labels)} labels using non-upright chars: {extracted_labels}")
                return extracted_labels
            else:
                logger.info(f"Page {page.page_number} (Attempt 1): No labels extracted after processing non-upright characters (e.g., all too short, numeric, or filtered).")
        else:
            logger.info(f"Page {page.page_number} (Attempt 1): No non-upright characters found. Skipping character-based vertical label extraction.")

        # --- Attempt 2: Using word extraction and vertical grouping (if Attempt 1 failed or found nothing) ---
        logger.info(f"Page {page.page_number}: Attempt 2 - Using word extraction and vertical grouping as fallback.")
        
        min_len_for_word_label = 3 # Heuristic for word-based labels
        X_TOLERANCE_WORDS = 2  # points, for word bucketing
        MAX_VERTICAL_WORD_GAP = 5 # points

        words_on_page = page.extract_words(
            keep_blank_chars=False, 
            use_text_flow=True, 
            extra_attrs=["x0", "x1", "top", "bottom"]
        )
        logger.debug(f"Page {page.page_number} (Attempt 2): Extracted {len(words_on_page)} words for grouping.")

        if not words_on_page:
            logger.info(f"Page {page.page_number} (Attempt 2): No words extracted from page. No labels found.")
            return [] 

        word_buckets = defaultdict(list)
        for word_obj in words_on_page:
            bucket_key = round(word_obj["x0"] / X_TOLERANCE_WORDS) * X_TOLERANCE_WORDS
            word_buckets[bucket_key].append(word_obj)
            
        alternative_labels = []
        sorted_x_bucket_keys = sorted(word_buckets.keys())
        if not sorted_x_bucket_keys:
            logger.debug(f"Page {page.page_number} (Attempt 2): No word buckets formed.")

        for x_key in sorted_x_bucket_keys:
            words_in_bucket = sorted(word_buckets[x_key], key=lambda w: w["top"])
            logger.debug(f"Page {page.page_number} (Attempt 2): Word bucket x_key={x_key} has {len(words_in_bucket)} words.")
            
            if not words_in_bucket:
                continue

            current_label_parts = []
            last_word_bottom = 0

            for i, word_obj in enumerate(words_in_bucket):
                word_text_raw = word_obj["text"]
                word_text = word_text_raw.strip()
                if not word_text:
                    logger.debug(f"Page {page.page_number} (Attempt 2): Skipped blank word in bucket {x_key}.")
                    continue
                
                logger.debug(f"Page {page.page_number} (Attempt 2): Processing word '{word_text}' (top: {word_obj['top']}, bottom: {last_word_bottom}, gap: {word_obj['top'] - last_word_bottom})")

                if current_label_parts and word_obj["top"] > last_word_bottom + MAX_VERTICAL_WORD_GAP:
                    formed_label_raw = "".join(current_label_parts)
                    formed_label = formed_label_raw.strip()
                    logger.debug(f"Page {page.page_number} (Attempt 2): Gap detected. Potential label from parts: '{formed_label}'")
                    if len(formed_label) >= min_len_for_word_label and not formed_label.isnumeric():
                        if "mittelwert" not in formed_label.lower().replace(" ", ""):
                            alternative_labels.append(formed_label)
                            logger.debug(f"Page {page.page_number} (Attempt 2): Kept label (gap logic): '{formed_label}'.")
                        else:
                            logger.debug(f"Page {page.page_number} (Attempt 2): Filtered 'Mittelwert' (gap logic): '{formed_label}'")
                    elif formed_label:
                         logger.debug(f"Page {page.page_number} (Attempt 2): Word '{formed_label}' (len {len(formed_label)}) too short or numeric (gap logic).")
                    current_label_parts = [word_text] 
                else:
                    current_label_parts.append(word_text)
                
                last_word_bottom = word_obj["bottom"]

            if current_label_parts:
                formed_label_raw = "".join(current_label_parts)
                formed_label = formed_label_raw.strip()
                logger.debug(f"Page {page.page_number} (Attempt 2): End of bucket. Potential label from remaining parts: '{formed_label}'")
                if len(formed_label) >= min_len_for_word_label and not formed_label.isnumeric():
                    if "mittelwert" not in formed_label.lower().replace(" ", ""):
                        alternative_labels.append(formed_label)
                        logger.debug(f"Page {page.page_number} (Attempt 2): Kept label (end of bucket): '{formed_label}'.")
                    else:
                        logger.debug(f"Page {page.page_number} (Attempt 2): Filtered 'Mittelwert' (end of bucket): '{formed_label}'")
                elif formed_label:
                    logger.debug(f"Page {page.page_number} (Attempt 2): Word '{formed_label}' (len {len(formed_label)}) too short or numeric (end of bucket).")
        
        if alternative_labels:
            logger.info(f"Page {page.page_number} (Attempt 2): Successfully extracted {len(alternative_labels)} labels using word grouping: {alternative_labels}")
            return alternative_labels
        else:
            logger.info(f"Page {page.page_number} (Attempt 2): No labels found using word grouping. No labels found overall for page.")
            return []
        
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