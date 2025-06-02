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

    def transform_tables(self, tables_from_processor: List[Dict[str, Any]], year: int) -> pd.DataFrame:
        """
        Transform extracted tables into a standardized format.
        
        Args:
            tables_from_processor: List of dictionaries containing raw table data
            year: Year of the trials
            
        Returns:
            DataFrame with standardized columns:
            - Year
            - Trial
            - Trait
            - Variety
            - Location
            - Treatment
            - RelativeValue
            - AbsoluteValue
            - Source
        """
        all_transformed_rows = []
        
        for table in tables_from_processor:
            table_type = table.get('table_type')
            treatment = table.get('treatment')
            source = table.get('source')
            trait = table.get('trait')
            
            if table_type == 'relative':
                transformed_rows = self._transform_relative_table(
                    year=year,
                    table_dict=table
                )
                all_transformed_rows.extend(transformed_rows)
                
            elif table_type == 'absolute':
                transformed_rows = self._transform_absolute_table(
                    year=year,
                    headers=table.get('headers', []),
                    rows=table.get('rows', []),
                    treatment=treatment,
                    source=source,
                    trait=trait
                )
                all_transformed_rows.extend(transformed_rows)
        
        if not all_transformed_rows:
            return pd.DataFrame(columns=self.standard_columns)
        
        # Convert to DataFrame and ensure all standard columns are present
        df = pd.DataFrame(all_transformed_rows)
        for col in self.standard_columns:
            if col not in df.columns:
                df[col] = None
                
        # Reorder columns to match standard format
        df = df[self.standard_columns]
        
        # Sort by Treatment, Location, and Variety
        df.sort_values(['Treatment', 'Location', 'Variety'], inplace=True)
        
        return df
    
    # Ensure reconstruct_vertical_text is available if used in fallback
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
            
        # Split into lines and filter out empty/whitespace-only lines
        parts = [p.strip() for p in text_from_cell.split('\n') if p.strip()]
        
        if not parts:
            return None
            
        # Special handling for location names with brackets
        if any('(' in p or ')' in p for p in parts):
            # Find opening and closing bracket indices
            start_idx = next((i for i, p in enumerate(parts) if '(' in p), -1)
            end_idx = next((i for i, p in enumerate(parts) if ')' in p), -1)
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                # Handle text before brackets
                before_brackets = ''.join(parts[:start_idx])
                
                # Handle text inside brackets
                in_brackets = ''.join(parts[start_idx:end_idx+1])
                in_brackets = in_brackets.replace('(', ' (').replace(')', ') ')
                
                # Handle text after brackets
                after_brackets = ''.join(parts[end_idx+1:])
                
                # Combine all parts
                return (before_brackets + in_brackets + after_brackets).strip()
        
        # Handle hyphenated location names
        if any('-' in p for p in parts):
            result = []
            current_word = []
            
            for part in parts:
                if '-' in part:
                    if current_word:
                        result.append(''.join(current_word))
                        current_word = []
                    result.append(part)
                else:
                    current_word.append(part)
            
            if current_word:
                result.append(''.join(current_word))
                
            return ' '.join(result)
        
        # Standard case: join all characters
        return ''.join(parts)
    
    def _clean_location_name(self, location: str) -> str:
        """
        Clean and format location names.
        
        Args:
            location: Raw location name string
            
        Returns:
            Cleaned location name
        """
        if not location:
            return ""
            
        # If location appears to be vertical text
        if '\n' in location:
            reconstructed = self.reconstruct_vertical_text(location)
            if reconstructed:
                return reconstructed
                
        return location.strip()
        
    def _transform_absolute_table(self, year: int, headers: list, rows: list, 
                            treatment: Optional[str] = None,
                            source: Optional[str] = None, 
                            trait: Optional[str] = None) -> list:
        transformed_rows = []
        location_cols = [i for i, h in enumerate(headers) if h and not any(x in str(h).lower() 
                    for x in ['sorte', 'qualität']) and str(h).strip() not in self.exclude_columns]
        
        trait_str = trait if trait else "Default Trait"
        
        for row in rows:
            if not row or any("Mittel" in str(cell) for cell in row):
                continue
                
            variety = row[0]
            if not variety or "Mittel" in str(variety) or variety.strip() in self.exclude_rows:
                continue

            # Handle both single treatment and 'Both' cases consistently
            treatments_to_process = []
            if treatment == 'Both':
                treatments_to_process = ['intensive', 'extensive']  # Changed from 'St 1'/'St 2' to match relative table
            else:
                treatments_to_process = [treatment if treatment else 'Standard']  # Use 'Standard' as fallback like relative table

            for loc_idx in location_cols:
                location = headers[loc_idx]
                if not location:
                    continue
                
                # Clean location name
                location = self._clean_location_name(location)
                
                # Skip if location name cleaning failed
                if not location:
                    continue

                if len(treatments_to_process) == 1:
                    # Single treatment case
                    value = self._clean_value(row[loc_idx])
                    if value is not None:
                        transformed_rows.append({
                            'Year': year,
                            'Trial': f"{year}_whw_de_prt_lsv",
                            'Trait': trait_str,
                            'Variety': variety,
                            'Location': location,
                            'Treatment': treatments_to_process[0],
                            'RelativeValue': None,  # Could calculate if needed
                            'AbsoluteValue': value,
                            'Source': source
                        })
                else:
                    # 'Both' treatment case - handle intensive and extensive
                    st1_idx = loc_idx + 1
                    st2_idx = loc_idx + 2
                    
                    # Process intensive treatment (St 1)
                    if st1_idx < len(row) and row[st1_idx]:
                        value = self._clean_value(row[st1_idx])
                        if value is not None:
                            transformed_rows.append({
                                'Year': year,
                                'Trial': f"{year}_whw_de_prt_lsv",
                                'Trait': trait_str,
                                'Variety': variety,
                                'Location': location,
                                'Treatment': 'intensive',
                                'RelativeValue': None,  # Could calculate if needed
                                'AbsoluteValue': value,
                                'Source': source
                            })
                    
                    # Process extensive treatment (St 2)
                    if st2_idx < len(row) and row[st2_idx]:
                        value = self._clean_value(row[st2_idx])
                        if value is not None:
                            transformed_rows.append({
                                'Year': year,
                                'Trial': f"{year}_whw_de_prt_lsv",
                                'Trait': trait_str,
                                'Variety': variety,
                                'Location': location,
                                'Treatment': 'extensive',
                                'RelativeValue': None,  # Could calculate if needed
                                'AbsoluteValue': value,
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

    def save_to_excel(self, transformed_data: pd.DataFrame, pdf_path: Path) -> None:
        """
        Save transformed data to Excel file with name matching the source PDF.
        
        Args:
            transformed_data: DataFrame containing the transformed table data
            pdf_path: Original PDF file path
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            # Get output filename - replace .pdf with .xlsx
            output_filename = pdf_path.stem + '.xlsx'
            output_path = output_dir / output_filename
            
            # Save to Excel
            transformed_data.to_excel(output_path, index=False)
            logger.info(f"[green]Successfully saved transformed data to {output_path}[/green]")
            
        except Exception as e:
            logger.error(f"[red]Error saving transformed data to Excel: {str(e)}[/red]")