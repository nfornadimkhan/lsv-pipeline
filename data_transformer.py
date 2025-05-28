"""
Module for transforming extracted table data into standardized format.
"""
import pandas as pd
import logging
from typing import List, Dict, Any, Optional # Import Optional
import re
from pathlib import Path
from config_manager import ConfigManager
from logging_config import get_logger

# Get logger instance
logger = get_logger("pdf_pipeline")

class DataTransformer:
    def __init__(self, config_manager=None):
        """Initialize data transformer."""
        # Standard columns for all tables
        self.standard_columns = [
            'Year', 'Trial', 'Trait', 'Variety', 
            'Location', 'Treatment', 'RelativeValue', 'AbsoluteValue', 'Source'
        ]
        self.config_manager = config_manager
        self.exclude_columns = config_manager.get_exclude_columns() if config_manager else []
        self.exclude_rows = config_manager.get_exclude_rows() if config_manager else []
        logger.info("[cyan]Initializing DataTransformer[/cyan]")
        logger.debug(f"[green]Loaded exclude_rows patterns:[/green] {self.exclude_rows}")
        
    def _transform_relative_table(self, year: int, table_dict: Dict[str, Any]) -> list: # Modified signature
        transformed_rows = []
        
        full_table_data = table_dict.get('raw_table_data')
        reference_row_content = table_dict.get('reference_row_content')
        # reference_row_idx_in_full_table = table_dict.get('reference_row_original_idx_in_table') # Original index
        page_vertical_labels = table_dict.get('page_vertical_labels')
        treatment = table_dict.get('treatment') # Get from table_dict
        source = table_dict.get('source')       # Get from table_dict
        trait = table_dict.get('trait')         # Get from table_dict

        if not full_table_data or reference_row_content is None:
            logger.warning("Relative table transformation missing essential data (full_table_data or reference_row_content).")
            return []

        # Find the actual index of reference_row_content within full_table_data
        # This is important because full_table_data is the complete raw table.
        actual_reference_row_idx = -1
        for i, r_data in enumerate(full_table_data):
            if r_data == reference_row_content: # Compare list objects
                actual_reference_row_idx = i
                break
        
        if actual_reference_row_idx == -1:
            logger.warning("Could not re-locate reference_row_content in full_table_data.")
            return []

        logger.debug(f"Processing relative table. Year: {year}, Trait: {trait}")
        logger.debug(f"Reference row (idx {actual_reference_row_idx}): {reference_row_content[:7]}")
        if page_vertical_labels:
            logger.debug(f"Using page_vertical_labels: {page_vertical_labels}")

        locations = []
        location_cols = [] # These are column indices in full_table_data

        # Try to use page_vertical_labels first
        if page_vertical_labels:
            locations = page_vertical_labels
            # Now, map these locations to column indices.
            # We need to find the "Versuch" row in full_table_data to know where data columns start.
            versuch_row_found = False
            for r_data in full_table_data: # Iterate through rows of the raw table
                if r_data and len(r_data) > 1 and r_data[1] and "Versuch" in str(r_data[1]):
                    # Assuming "Versuch" is in the second cell (index 1) of its row.
                    # The actual location data columns start from the third cell (index 2).
                    if len(r_data) >= 2 + len(locations):
                        location_cols = list(range(2, 2 + len(locations)))
                        logger.info(f"Successfully mapped {len(locations)} page_vertical_labels to data columns: {location_cols}")
                        versuch_row_found = True
                    else:
                        logger.warning(f"The 'Versuch' row (or equivalent header) has {len(r_data)} cells, not enough for {len(locations)} vertical labels starting at index 2. Vertical labels: {locations}")
                    break # Found the 'Versuch' row
            
            if not versuch_row_found:
                logger.warning("Could not find a 'Versuch' row to align page_vertical_labels. These labels will be used but column indices might be default.")
                # Fallback: assume locations start at col 2 if no Versuch row helps. This is risky.
                # A better fallback might be to revert to the old parsing logic if this fails.
                # For now, if we have page_vertical_labels, we try to use them.
                # If alignment fails, the `location_cols` might remain empty, which will skip data processing.
                if not location_cols and locations: # If we have labels but no cols
                    logger.warning(f"Attempting to use default column indices for {len(locations)} vertical labels (2 to {2+len(locations)-1}) due to no 'Versuch' row alignment.")
                    # Check if the first data row (e.g. reference_row_content) has enough columns
                    if len(reference_row_content) >= 2 + len(locations):
                        location_cols = list(range(2, 2 + len(locations)))
                    else:
                        logger.error(f"Cannot use default column indices for vertical labels; reference row too short. Labels: {locations}")
                        locations = [] # Clear locations to prevent processing with bad cols
                        location_cols = []


        # Fallback or if page_vertical_labels were not available/usable
        if not locations: # if locations list is still empty
            logger.info("No page_vertical_labels used or mapping failed. Falling back to reconstructing locations from table content.")
            location_row_from_table = None
            for r_data in full_table_data:
                 if r_data and len(r_data) > 1 and r_data[1] and "Versuch" in str(r_data[1]):
                     location_row_from_table = r_data
                     break
            if location_row_from_table:
                logger.debug(f"Fallback: Found location row in table: {location_row_from_table[:7]}")
                # Iterate its cells (from index 2), use reconstruct_vertical_text
                for col_idx in range(2, len(location_row_from_table)):
                    raw_header_text = location_row_from_table[col_idx]
                    if raw_header_text: 
                        reconstructed_location = self.reconstruct_vertical_text(str(raw_header_text))
                        if reconstructed_location and reconstructed_location.strip():
                            # Check if this column index is within bounds of the reference_row_content,
                            # as that indicates a valid data column.
                            if col_idx < len(reference_row_content):
                                locations.append(reconstructed_location.strip())
                                location_cols.append(col_idx)
                            else:
                                logger.debug(f"Fallback: Reconstructed location '{reconstructed_location.strip()}' from column {col_idx} ignored as it's out of bounds for reference_row_content (len: {len(reference_row_content)}).")
                        else:
                            logger.debug(f"Fallback: Reconstructed text for column {col_idx} is empty or None. Original: '{raw_header_text}'")
                    # else: # Optional: if cell is empty, decide whether to stop or continue
                        # logger.debug(f"Fallback: Cell at column {col_idx} in location_row_from_table is empty.")

                if locations:
                    logger.info(f"Fallback: Successfully reconstructed {len(locations)} locations from table content: {locations}")
                    logger.info(f"Fallback: Corresponding column indices: {location_cols}")
                else:
                    logger.warning("Fallback: Could not reconstruct any valid locations from the identified table row.")
            else:
                logger.warning("Fallback failed: No 'Versuch' row found in table content either.")


        if not locations or not location_cols:
            logger.warning("No valid locations or location_cols determined. Cannot process relative table data.")
            return []

        # Build reference map for calculating absolute values
        reference_map = {}
        for loc_col_idx in location_cols:
            if loc_col_idx < len(reference_row_content):
                ref_val = self._clean_value(reference_row_content[loc_col_idx])
                if ref_val is not None:
                    reference_map[loc_col_idx] = ref_val
            else:
                logger.warning(f"Location column index {loc_col_idx} is out of bounds for reference_row_content (len: {len(reference_row_content)})")
        
        if not reference_map:
            logger.warning("Reference map is empty. Absolute values cannot be calculated.")
            # Decide if to proceed without absolute values or return []

        trait_str = trait if trait else "Default Relative Trait"
        logger.debug(f"Final locations to process: {locations}")
        logger.debug(f"Corresponding column indices: {location_cols}")
        logger.debug(f"Reference map: {reference_map}")

        # Iterate through data rows in full_table_data, starting after the reference row
        for data_row_idx in range(actual_reference_row_idx + 1, len(full_table_data)):
            row_data = full_table_data[data_row_idx]
            if not row_data or len(row_data) <= 1: # Skip empty or too short rows
                continue
                
            variety = str(row_data[1]).strip() if len(row_data) > 1 and row_data[1] else ""
            if not variety: # Skip if no variety name
                continue
            
            # Variety exclusion logic
            if self._is_excluded(variety, self.exclude_rows) or \
               any(x in variety.lower() for x in ["anbaugebiet", "boden", "vgl. reduziert", "rel. grenzdifferenz", 
                                                "e-sorten", "a-sorten", "b-sorten", "c-sorten", "anhang", 
                                                "durum (hartweizen)", "versuch", "mittelwert", "alle"]): # Added mittelwert/alle
                logger.debug(f"[red]Excluded variety/row: {repr(variety)}[/red]")
                continue

            for loc_col_idx, location_name in zip(location_cols, locations):
                if loc_col_idx >= len(row_data):
                    logger.debug(f"Variety {variety}: loc_col_idx {loc_col_idx} out of bounds for row_data (len {len(row_data)})")
                    continue
                    
                rel_val_str = row_data[loc_col_idx]
                rel_val = self._clean_value(rel_val_str)
                ref_val = reference_map.get(loc_col_idx)
                
                if rel_val is not None and ref_val is not None and ref_val != 0:
                    abs_val = round(rel_val * ref_val / 100.0, 2) # Added rounding
                    transformed_rows.append({
                        'Year': year,
                        'Trial': f"{year}_whw_de_prt_lsv" if year else "whw_de_prt_lsv", # Ensure year is int
                        'Trait': trait_str,
                        'Variety': variety,
                        'Location': location_name,
                        'Treatment': treatment if treatment and treatment != 'Both' else 'Standard',
                        'RelativeValue': rel_val,
                        'AbsoluteValue': abs_val,
                        'Source': source
                    })
                # Optional: Add logging for skipped data points
                elif rel_val is None:
                    logger.debug(f"Skipping {variety} at {location_name}: rel_val is None (original: '{rel_val_str}')")
                elif ref_val is None:
                    logger.debug(f"Skipping {variety} at {location_name}: ref_val is None for loc_idx {loc_col_idx}")
                elif ref_val == 0:
                    logger.debug(f"Skipping {variety} at {location_name}: ref_val is 0 for loc_idx {loc_col_idx}")


        logger.info(f"[green]Transformed {len(transformed_rows)} rows from relative table for trait '{trait_str}'[/green]")
        return transformed_rows

    def transform_tables(self, tables_from_processor: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform extracted tables into standardized format."""
        all_transformed_data = []
        
        for table_data_dict in tables_from_processor:
            try:
                year_val = table_data_dict.get('year')
                table_type = table_data_dict.get('table_type')
                
                if year_val is None: # Ensure year is present
                    logger.warning(f"Skipping table transformation, year is None. Source: {table_data_dict.get('source')}")
                    continue
                if not isinstance(year_val, int): # Ensure year is int
                    try:
                        year_val = int(year_val)
                    except ValueError:
                        logger.warning(f"Skipping table transformation, year '{year_val}' is not a valid integer. Source: {table_data_dict.get('source')}")
                        continue
                
                if table_type == 'relative':
                    # Pass the whole table_data_dict to _transform_relative_table
                    all_transformed_data.extend(
                        self._transform_relative_table(year_val, table_data_dict)
                    )
                elif table_type == 'absolute':
                    # Ensure _transform_absolute_table also gets what it needs from table_data_dict
                    headers = table_data_dict.get('headers')
                    rows = table_data_dict.get('rows')
                    treatment = table_data_dict.get('treatment')
                    source = table_data_dict.get('source')
                    trait = table_data_dict.get('trait')
                    if headers and rows:
                         all_transformed_data.extend(
                            self._transform_absolute_table(year_val, headers, rows, treatment, source, trait)
                        )
                    else:
                        logger.warning(f"Missing headers/rows for absolute table. Source: {source}")
                # ... other table types
            except Exception as e:
                logger.error(f"Error during table transformation dispatch: {str(e)} for table from {table_data_dict.get('source')}", exc_info=True)
                continue
        
        df = pd.DataFrame(all_transformed_data, columns=self.standard_columns)
        df = self._clean_data(df) # Assuming _clean_data is defined
        return df
    
    # Ensure reconstruct_vertical_text is available if used in fallback
    def reconstruct_vertical_text(self, text_from_cell: str) -> Optional[str]:
        # ... (your existing implementation of this function)
        if not text_from_cell: return None
        parts_raw = text_from_cell.split('\n')
        parts_stripped = [p.strip() for p in parts_raw if p.strip()]
        if not parts_stripped: return None
        reconstructed_name = "".join(parts_stripped)
        reconstructed_name = " ".join(reconstructed_name.split())
        return reconstructed_name if reconstructed_name else None
        
    def _transform_absolute_table(self, year: int, headers: list, rows: list, treatment: Optional[str] = None, # Use Optional for clarity
                                  source: Optional[str] = None, trait: Optional[str] = None) -> list:
        transformed_rows = []
        location_cols = [i for i, h in enumerate(headers) if h and not any(x in str(h).lower() for x in ['sorte', 'qualität']) and str(h).strip() not in self.exclude_columns]
        
        # Use trait from config or default
        trait_str = trait if trait else "Default Trait"
        
        for row in rows:
            if not row or any("Mittel" in str(cell) for cell in row):
                continue
            variety = row[0]
            if not variety or "Mittel" in str(variety) or variety.strip() in self.exclude_rows:
                continue
            if treatment and treatment != 'Both':
                for loc_idx in location_cols:
                    location = headers[loc_idx]
                    if not location:
                        continue
                    value = self._clean_value(row[loc_idx])
                    if value is not None:
                        transformed_rows.append({
                            'Year': year,
                            'Trial': f"{year}_whw_de_prt_lsv",
                            'Trait': trait_str,
                            'Variety': variety,
                            'Location': location,
                            'Treatment': treatment,
                            'RelativeValue': None,
                            'AbsoluteValue': value,
                            'Source': source
                        })
            else:
                for loc_idx in location_cols:
                    location = headers[loc_idx]
                    if not location:
                        continue
                    st1_idx = loc_idx + 1
                    st2_idx = loc_idx + 2
                    if st1_idx < len(row) and row[st1_idx]:
                        transformed_rows.append({
                            'Year': year,
                            'Trial': f"{year}_whw_de_prt_lsv",
                            'Trait': trait_str,
                            'Variety': variety,
                            'Location': location,
                            'Treatment': 'St 1',
                            'RelativeValue': None,
                            'AbsoluteValue': self._clean_value(row[st1_idx]),
                            'Source': source
                        })
                    if st2_idx < len(row) and row[st2_idx]:
                        transformed_rows.append({
                            'Year': year,
                            'Trial': f"{year}_whw_de_prt_lsv",
                            'Trait': trait_str,
                            'Variety': variety,
                            'Location': location,
                            'Treatment': 'St 2',
                            'RelativeValue': None,
                            'AbsoluteValue': self._clean_value(row[st2_idx]),
                            'Source': source
                        })
        return transformed_rows
        
    def _clean_value(self, value: Any) -> Optional[float]: # Allow Any type for input value
        """Clean and convert value to float."""
        if value is None: # Explicitly check for None
            return None
            
        # Remove any non-numeric characters except decimal point
        cleaned = re.sub(r'[^\d.,]', '', str(value))
        # Replace comma with decimal point
        cleaned = cleaned.replace(',', '.')
        
        try:
            return float(cleaned)
        except ValueError:
            return None
            
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate transformed data."""
        if 'AbsoluteValue' in df.columns:
            df = df.dropna(subset=['AbsoluteValue'])
            df = df[df['AbsoluteValue'] > 0]
        
        # Sort by Year, Variety, Location, Treatment
        df = df.sort_values(['Year', 'Variety', 'Location', 'Treatment'])
        
        return df 

    def _is_excluded(self, value: Any, exclusion_patterns: List[Dict[str, str]]): # Allow Any for value
        """Check if a value matches any exclusion pattern."""
        if value is None: # Explicitly check for None
            return False
            
        value_lower = str(value).strip().lower()
        logger.debug(f"[cyan]Checking exclusion for value:[/cyan] {repr(value_lower)}")
        
        for pattern in exclusion_patterns:
            ptype = pattern.get('type', 'contains')
            pval = str(pattern.get('value', '')).strip().lower()
            
            logger.debug(f"[yellow]Testing pattern:[/yellow] type={ptype}, value={repr(pval)}")
            
            if ptype == 'contains' and pval in value_lower:
                logger.debug(f"[green]Matched 'contains' pattern:[/green] {repr(pval)} in {repr(value_lower)}")
                return True
            if ptype == 'startswith' and value_lower.startswith(pval):
                logger.debug(f"[green]Matched 'startswith' pattern:[/green] {repr(value_lower)} starts with {repr(pval)}")
                return True
            if ptype == 'endswith' and value_lower.endswith(pval):
                logger.debug(f"[green]Matched 'endswith' pattern:[/green] {repr(value_lower)} ends with {repr(pval)}")
                return True
                
        return False