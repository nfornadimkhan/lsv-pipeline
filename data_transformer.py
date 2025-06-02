"""
Module for transforming extracted table data into standardized format.
"""
import re
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
<<<<<<< HEAD
=======
import re
>>>>>>> aad3dccbd548be587baef5c04477742559373fa6
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
        logger.debug(f"[green]Loaded exclude_columns patterns:[/green] {self.exclude_columns}")
        logger.debug(f"[green]Loaded exclude_rows patterns:[/green] {self.exclude_rows}")
        
    def _matches_exclusion_pattern(self, text: str, patterns: List[Dict[str, str]]) -> bool:
        """
        Check if text matches any exclusion pattern.
        
        Args:
            text: Text to check
            patterns: List of pattern dictionaries with 'type' and 'value' keys
            
        Returns:
            bool: True if text matches any pattern, False otherwise
        """
        if not text or not patterns:
            return False
            
        text = str(text).strip()
        for pattern in patterns:
            pattern_type = pattern.get('type')
            pattern_value = pattern.get('value')
            
            if pattern_type == 'startswith' and text.lower().startswith(pattern_value.lower()):
                return True
            elif pattern_type == 'endswith' and text.lower().endswith(pattern_value.lower()):
                return True
            elif pattern_type == 'contains' and pattern_value.lower() in text.lower():
                return True
                
        return False

    def _normalize_german_text(self, text: str) -> str:
        """
        Normalize German text by handling special characters and common replacements.
        """
        if not text:
            return ""
            
        # Common German text replacements
        replacements = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
            'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
            'ß': 'ss',
            'é': 'e', 'è': 'e', 'ê': 'e',
            'á': 'a', 'à': 'a', 'â': 'a',
            'í': 'i', 'ì': 'i', 'î': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u'
        }
        
        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text

    def _reconstruct_vertical_text(self, text: str) -> str:
        """
        Reconstruct vertical text into proper location name.
        Example: 'r )\nui h\n-B nic\nn e\ne v\np r\nr ö\ne N\nK (' -> 'Kerpen-Buir (Nörvenich)'
        """
        if not text:
            return ""
            
        # Special handling for vertical text patterns
        # Map of scrambled patterns to actual location names
        vertical_patterns = {
            # Original patterns with newlines
            'r )\nui h\n-B nic\nn e\ne v\np r\nr ö\ne N\nK (': 'Kerpen-Buir (Nörvenich)',
            'z- h\nn t\ne a\nel nr\nk e\nr V\nE': 'Erkelenz-Venrath', 
            't\nr\ne\nw\nel\nt\nt\nMi': 'Mittelwert',
            'e\ns .)\ns h\nü g\nD n\ns ti\nu s\na O\nH (': 'Haus Düsse (Oberkassel)',
            'n\ne- e\ng d\na ei\nL H': 'Hagen-Leidenhausen',
            'n\ne\nv\ne\nr\nG': 'Greven',
            '-\nn n\nei e\nt g\ns a\nWar All': 'Warstein-Allagen',
            'n\n- e\ng f\nr ö\ne h\nb n\nm e\no st\nBl ol\nH': 'Blomberg-Hohenfels',
            'm *\nz\ne t\nt a\nr s\ne n\nzi ei\nu z\nd t\ne u\nr h\nei sc\nb n\nze\ng\na n\ntr a\nEr Pfl': 'Ertrag bei angepasstem Pflanzenschutz',
            # Patterns after PDF processor cleaning (spaces between characters)
            'r ) ui h -B nic n e e v p r r ö e N K (': 'Kerpen-Buir (Nörvenich)',
            'r ) uih -Bnicn ee vp rr öe NK (': 'Kerpen-Buir (Nörvenich)',
            'z- h n t e a el nr k e r V E': 'Erkelenz-Venrath',
            'z- hn te aelnrk er VE': 'Erkelenz-Venrath',
            't r e w el t t Mi': 'Mittelwert',
            'e s .) s h ü g D n s ti u s a O H (': 'Haus Düsse (Oberkassel)',
            'es .) sh üg Dn stiu sa OH (': 'Haus Düsse (Oberkassel)',
            'n e- e g d a ei L H': 'Hagen-Leidenhausen',
            'ne- eg da eiL H': 'Hagen-Leidenhausen',
            'n e v e r G': 'Greven',
            '- n n ei e t g s a War All': 'Warstein-Allagen',
            '- nn eie tg sa WarAll': 'Warstein-Allagen',
            'n - e g f r ö e h b n m e o st Bl ol H': 'Blomberg-Hohenfels',
            'n - eg fr öe hb nm eo stBlolH': 'Blomberg-Hohenfels',
            'm * z e t t a r s e n zi ei u z d t e u r h ei sc b n ze g a n tr a Er Pfl': 'Ertrag bei angepasstem Pflanzenschutz',
            'm * ze tt ar se nzieiu zd te ur heiscb nzeg an tra ErPfl': 'Ertrag bei angepasstem Pflanzenschutz',
            # Add pattern for Differenz
            'm . h e c t s s n s a e p z n e g a n fl a P u m z e z v n si e n er e ff nt Di /i': 'Differenz',
            'm . h e c t s s n s a e p z n e g a n fl a P u m z e z v n si e n er e ff nt Di': 'Differenz',
            'm . h e c t s s n s a e p z n e g a n fl a P u m z e z v n si e n er e ff nt Di /i': 'Differenz',
            
        }
        
        # Check if text matches any known vertical pattern exactly
        if text in vertical_patterns:
            return vertical_patterns[text]
        
        # Try to reconstruct from vertical text
        # Remove newlines and extra spaces
        clean_text = re.sub(r'\s+', '', text)
        
        # Known location mappings based on character sequences
        # Fix: Escape special characters in regex patterns
        location_mappings = {
            r'r\)uih-Bnicneevprr[öo]eNK\(\)': 'Kerpen-Buir (Nörvenich)',
            r'z-hnte[ae]nrelkr[ne]rVE': 'Erkelenz-Venrath',
            r'trewelttMi': 'Mittelwert',
            r'es[.)]sh[üu]gDnsti[ou]saOH\(\)': 'Haus Düsse (Oberkassel)',
            r'ne-egdaeiLH': 'Hagen-Leidenhausen',
            r'neverG': 'Greven',
            r'-nneietgsaWarAll': 'Warstein-Allagen',
            r'n-egfr[öo]ehbnmeostBlolH': 'Blomberg-Hohenfels',
            r'mzettarsenzieiuzdteurheiscbnzegantraErPfl': 'Ertrag bei angepasstem Pflanzenschutz',
            # Add pattern for Differenz
            r'm\.hectssnsaepznenganflaPumzezvnsienereffntDi': 'Differenz',
            r'm\.hectssnsaepznenganflaPumzezvnsienereffntDi/i': 'Differenz',
        }
        
        # Check if the cleaned text matches any known mapping (using regex)
        for pattern, location in location_mappings.items():
            try:
                if re.match(pattern, clean_text, re.IGNORECASE):
                    return location
            except re.error as e:
                logger.warning(f"Regex error with pattern '{pattern}': {str(e)}")
                continue
        
        # If no exact match, try to identify patterns
        # Handle Mittelwert pattern
        if 'mittel' in clean_text.lower() or ('mi' in clean_text.lower() and 'wert' in clean_text.lower()):
            return 'Mittelwert'
            
        # For Greven
        if clean_text.lower() in ['greven', 'neverG', 'neverG'.lower()]:
            return 'Greven'
            
        # Check for Differenz pattern
        if 'differenz' in clean_text.lower() or 'ffntdi' in clean_text.lower():
            return 'Differenz'
            
        # If result is too short or doesn't make sense, return original
        if len(clean_text) < 3:
            return text
            
        return text
        
    def _transform_relative_table(self, year: int, table_dict: Dict[str, Any]) -> list:
        transformed_rows = []
        
        full_table_data = table_dict.get('raw_table_data')
        reference_row_content = table_dict.get('reference_row_content')
        treatment = table_dict.get('treatment')
        source = table_dict.get('source')
        trait = table_dict.get('trait')

        if not full_table_data or reference_row_content is None:
            logger.warning("Relative table transformation missing essential data (full_table_data or reference_row_content).")
            return []

        # Log table structure
        logger.info(f"[cyan]Table Structure Analysis:[/cyan]")
        logger.info(f"Total rows: {len(full_table_data)}")
        for idx, row in enumerate(full_table_data):
            logger.info(f"Row {idx}: {row}")

        # Find the actual index of reference_row_content within full_table_data
        actual_reference_row_idx = -1
        for i, r_data in enumerate(full_table_data):
            if r_data == reference_row_content:
                actual_reference_row_idx = i
                break
        
        if actual_reference_row_idx == -1:
            logger.warning("Could not re-locate reference_row_content in full_table_data.")
            return []

        logger.debug(f"Processing relative table. Year: {year}, Trait: {trait}")
        logger.debug(f"Reference row (idx {actual_reference_row_idx}): {reference_row_content[:7]}")

<<<<<<< HEAD
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
                        reconstructed_location = self._reconstruct_vertical_text(str(raw_header_text))
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
=======
        # Find trial names (locations) from the table
        trial_names = []
        excluded_columns = set()  # Keep track of columns to exclude
>>>>>>> aad3dccbd548be587baef5c04477742559373fa6
        
        for row_idx, row_data in enumerate(full_table_data):
            if row_idx < actual_reference_row_idx:  # Look in header rows
                if row_data and len(row_data) > 1 and row_data[1] and "Versuch" in str(row_data[1]):
                    logger.info(f"Found trial row at index {row_idx}: {row_data}")
                    # Extract trial names from this row
                    for col_idx in range(2, len(row_data)):
                        if col_idx < len(row_data) and row_data[col_idx]:
                            raw_trial_name = str(row_data[col_idx]).strip()
                            trial_name = self._reconstruct_vertical_text(raw_trial_name)
                            
                            # Check if this column should be excluded
                            if (trial_name in ["Mittelwert", "Ertrag bei angepasstem Pflanzenschutz"] or 
                                "Differenz" in trial_name or 
                                self._matches_exclusion_pattern(raw_trial_name, self.exclude_columns)):
                                excluded_columns.add(col_idx)
                                logger.debug(f"Excluding column {col_idx}: Location {trial_name} is in exclusion list")
                                continue
                                
                            if trial_name:
                                trial_names.append((col_idx, trial_name))
                                logger.info(f"Extracted trial name: {trial_name} (from: {raw_trial_name}) at column {col_idx}")

        logger.info(f"Extracted trial names: {[name for _, name in trial_names]}")
        logger.info(f"Excluded columns: {sorted(excluded_columns)}")

        # Process data rows
        for row_idx, row_data in enumerate(full_table_data):
            if row_idx <= actual_reference_row_idx:  # Skip header rows and reference row
                continue
                
            if not row_data or len(row_data) < 2:
                continue
                
            variety = self._clean_text(row_data[1] if len(row_data) > 1 else row_data[0])
            if not variety or self._matches_exclusion_pattern(variety, self.exclude_rows):
                logger.debug(f"Skipping row {row_idx}: {variety} (excluded or mean value)")
                continue

            logger.info(f"Processing variety: {variety} at row {row_idx}")

            # Process each data column
            for col_idx, trial_name in trial_names:  # Only process columns with valid locations
                if col_idx >= len(row_data) or col_idx >= len(reference_row_content):
                    continue
                    
                value = self._clean_value(row_data[col_idx])
                if value is not None:
                    # Calculate absolute value from relative value
                    ref_value = self._clean_value(reference_row_content[col_idx])
                    abs_value = None
                    if ref_value is not None and ref_value != 0:
                        abs_value = round(value * ref_value / 100.0, 2)
                        logger.debug(f"Calculated absolute value: {abs_value} from relative {value} and reference {ref_value}")

                    transformed_rows.append({
                        'Year': year,
                        'Trial': f"{year}_whw_de_prt_lsv",
                        'Trait': trait if trait else "Default Trait",
                        'Variety': variety,
                        'Location': trial_name,
                        'Treatment': treatment if treatment else "Default",
                        'RelativeValue': value,
                        'AbsoluteValue': abs_value,
                        'Source': source
                    })
                    logger.info(f"Added data point: Variety={variety}, Location={trial_name}, Value={value}, AbsValue={abs_value}")

        return transformed_rows

<<<<<<< HEAD
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
        
        # We need to collect all vertical_labels from the input tables to preserve them
        valid_location_markers = []
        for table in tables_from_processor:
            if 'page_vertical_labels' in table and table['page_vertical_labels']:
                valid_location_markers.extend(table['page_vertical_labels'])
    
        # Convert to DataFrame and ensure all standard columns are present
        df = pd.DataFrame(all_transformed_rows)
        for col in self.standard_columns:
            if col not in df.columns:
                df[col] = None
    
        # Only filter out completely empty locations or pure separator lines
        if 'Location' in df.columns and not df.empty:
            initial_count = len(df)
            # Remove rows with empty locations
            df = df[df['Location'].str.strip() != '']
            
            # More careful filtering - only filter pure separator lines
            # BUT preserve any locations that came from verified vertical labels
            separator_mask = df['Location'].apply(lambda x: 
                # Only consider it a separator if:
                # 1. It matches the separator pattern AND
                # 2. It's not in our valid location markers list AND
                # 3. It has 2 or fewer unique characters
                bool(re.match(r'^[-=_*]+$', str(x).strip())) and 
                str(x).strip() not in valid_location_markers and
                len(set(str(x).strip())) <= 2
            )
            df = df[~separator_mask]
            
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                logger.info(f"[yellow]Filtered out {filtered_count} rows with empty or separator-only locations[/yellow]")
            
        # Reorder columns to match standard format
        df = df[self.standard_columns]
        
        # Sort by Treatment, Location, and Variety
        df.sort_values(['Treatment', 'Location', 'Variety'], inplace=True)
        
        return df
    
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
=======
    def _transform_absolute_table(self, year: int, headers: list, rows: list, treatment: Optional[str] = None,
                                  source: Optional[str] = None, trait: Optional[str] = None) -> list:
        """Transform absolute table data into standardized format."""
        transformed_rows = []
        logger.info(f"Transforming absolute table. Initial headers: {headers}, Initial rows count: {len(rows)}")
        
        is_both_treatment = False
        secondary_header_row_used = False

        # 1. Check primary headers (the 'headers' argument)
        if headers: # Ensure headers list is not empty
            for h_val in headers:
                if h_val and any(pattern in str(h_val).lower() for pattern in ['stufe 1', 'st 1', 'stufe 2', 'st 2']):
                    is_both_treatment = True
                    logger.info("Detected Stufe 1/2 pattern in primary headers.")
                    break
        
        # 2. If not found in primary headers, check the first row of 'rows' data
        if not is_both_treatment and rows: # Ensure rows list is not empty
            first_potential_header_row = rows[0]
            stufe_pattern_count = 0
            for cell in first_potential_header_row:
                if cell and any(pattern in str(cell).lower() for pattern in ['stufe 1', 'st 1', 'stufe 2', 'st 2']):
                    stufe_pattern_count +=1
            
            # Heuristic: if multiple Stufe mentions, or if config treatment is 'Both' and we see at least one
            config_suggests_both = (treatment == "Both")
            if stufe_pattern_count > 1 or (config_suggests_both and stufe_pattern_count > 0) : 
                is_both_treatment = True
                secondary_header_row_used = True
                logger.info(f"Detected Stufe 1/2 pattern in the first data row (stufe_pattern_count: {stufe_pattern_count}). Treating it as a secondary header.")
        
        actual_data_rows = rows
        if secondary_header_row_used:
            actual_data_rows = rows[1:] # Data starts from the second row of original 'rows'

        # Log for debugging header interpretation
        logger.debug(f"For table with source '{source}', primary headers used for locations: {headers}")
        if secondary_header_row_used and rows:
             logger.debug(f"Secondary header row (from rows[0]): {rows[0]}")
        logger.debug(f"Effective data rows for processing: {len(actual_data_rows)}")
        logger.debug(f"is_both_treatment flag: {is_both_treatment} (secondary_header_row_used: {secondary_header_row_used}), config treatment: {treatment}")

        trial_names = [] # Stores (idx_st1, idx_st2, name) for both, or (idx, name) for single
        excluded_data_indices = set() 

        if is_both_treatment:
            logger.info("Extracting trial names for 'both treatments' table structure.")
            # Based on logged primary headers: ['', None, 'LOC1', None, 'LOC2', None, ...]
            # LOC1 header at index h_idx corresponds to Stufe 1 data at data column h_idx, and Stufe 2 at data column h_idx + 1.
            
            h_idx = 0
            while h_idx < len(headers):
                # A location is typically followed by a None or empty string in the primary header for this Stufe 1/2 table structure.
                # The location name itself (headers[h_idx]) must be non-empty.
                if headers[h_idx] and (h_idx + 1 < len(headers)): # Check if current header is non-empty AND there's a next column for Stufe 2 placeholder
                    raw_location_name = str(headers[h_idx])
                    reconstructed_name = self._reconstruct_vertical_text(raw_location_name)

                    # Filter out common non-location headers like "Sorte" or empty strings after reconstruction.
                    if reconstructed_name.lower() == "sorte" or not reconstructed_name.strip():
                        h_idx += 1 # Move to next header cell
                        continue

                    # Check for exclusion
                    if (reconstructed_name in ["Mittelwert", "Ertrag bei angepasstem Pflanzenschutz"] or
                        "Differenz" in reconstructed_name or
                        self._matches_exclusion_pattern(raw_location_name, self.exclude_columns)):
                        
                        excluded_data_indices.add(h_idx)      # Stufe 1 data column index
                        excluded_data_indices.add(h_idx + 1)  # Stufe 2 data column index
                        logger.debug(f"Excluding data columns for '{reconstructed_name}' (from '{raw_location_name}' at header index {h_idx}): data indices {h_idx}, {h_idx + 1}.")
                        h_idx += 2 # Processed this header and its pair, move past both
                        continue
                    
                    # If not excluded, add to trial_names.
                    # Ensure st2_data_idx (h_idx + 1) is valid for data rows if possible (though this check is also done during data extraction)
                    st1_data_idx = h_idx
                    st2_data_idx = h_idx + 1
                    
                    # Basic check against actual_data_rows width if available (first data row)
                    if actual_data_rows and actual_data_rows[0] and st2_data_idx >= len(actual_data_rows[0]):
                        logger.warning(f"Location '{reconstructed_name}' at header index {h_idx}: Stufe 2 data column {st2_data_idx} would be out of bounds for data row width {len(actual_data_rows[0])}. Skipping this trial pair.")
                        h_idx += 2 # Move past this header and its presumed pair
                        continue
                        
                    trial_names.append((st1_data_idx, st2_data_idx, reconstructed_name))
                    logger.info(f"Found trial: '{reconstructed_name}' (from header '{raw_location_name}' at h_idx {h_idx}). Stufe 1 data col: {st1_data_idx}, Stufe 2 data col: {st2_data_idx}.")
                    h_idx += 2 # Processed a location and its Stufe 2 partner, advance by 2
                else:
                    h_idx += 1 # Header cell is empty, or no partner column; advance by 1
        
        else: # Single treatment table
            logger.info("Extracting trial names for 'single treatment' table structure.")
            max_cols_to_scan = len(headers)
            if actual_data_rows and len(actual_data_rows[0]) > max_cols_to_scan:
                 max_cols_to_scan = len(actual_data_rows[0])

            for i in range(1, max_cols_to_scan): # Iterate through potential data column indices
                if i < len(headers) and headers[i]: # Check if there's a header for this column
                    raw_trial_name = str(headers[i])
                    location = self._reconstruct_vertical_text(raw_trial_name)
                    if (location in ["Mittelwert", "Ertrag bei angepasstem Pflanzenschutz"] or
                        "Differenz" in location or
                        self._matches_exclusion_pattern(raw_trial_name, self.exclude_columns)):
                        excluded_data_indices.add(i)
                        logger.debug(f"Excluding data column {i}: Location '{location}' (from '{raw_trial_name}') is in exclusion list.")
                    else:
                        trial_names.append((i, location)) # Store data column index and location name
                        logger.info(f"Found trial: '{location}' (Single treatment: col {i})")
        
        logger.info(f"Final extracted trial_names: {trial_names}")
        logger.info(f"Excluded data column indices: {sorted(list(excluded_data_indices))}")
        
        trait_str = trait if trait else "Default Trait"
        
        for row_idx, row_data in enumerate(actual_data_rows):
            if not row_data or len(row_data) == 0: # Skip empty rows
                logger.debug(f"Skipping empty row_data at index {row_idx}.")
                continue

            # Variety is usually in the first column (index 0)
            variety = self._normalize_german_text(row_data[0])
            if not variety or self._matches_exclusion_pattern(variety, self.exclude_rows) or "Mittelwert" in variety:
                logger.debug(f"Skipping variety: '{variety}' at row_idx {row_idx} (excluded or mean value).")
                continue

            logger.info(f"Processing variety: '{variety}' at data_row index {row_idx}")

            if is_both_treatment:
                for st1_data_idx, st2_data_idx, trial_name_loc in trial_names:
                    if st1_data_idx >= len(row_data) or st2_data_idx >= len(row_data):
                        logger.warning(f"Data column indices {st1_data_idx}/{st2_data_idx} out of bounds for row_data (len {len(row_data)}) for variety '{variety}'. Skipping.")
                        continue
                        
                    st1_value = self._clean_value(row_data[st1_data_idx])
                    if st1_value is not None:
                        transformed_rows.append({
                            'Year': year, 'Trial': f"{year}_whw_de_prt_lsv", 'Trait': trait_str,
                            'Variety': variety, 'Location': trial_name_loc, 'Treatment': 'intensive',
                            'RelativeValue': None, 'AbsoluteValue': st1_value, 'Source': source
                        })
                        logger.debug(f"Added: Var={variety}, Loc={trial_name_loc}, Treat=intensive, Val={st1_value}")
                    
                    st2_value = self._clean_value(row_data[st2_data_idx])
                    if st2_value is not None:
                        transformed_rows.append({
                            'Year': year, 'Trial': f"{year}_whw_de_prt_lsv", 'Trait': trait_str,
                            'Variety': variety, 'Location': trial_name_loc, 'Treatment': 'extensive',
                            'RelativeValue': None, 'AbsoluteValue': st2_value, 'Source': source
                        })
                        logger.debug(f"Added: Var={variety}, Loc={trial_name_loc}, Treat=extensive, Val={st2_value}")
            else: # Single treatment
                # Use the treatment from config, or "Default"
                current_treatment_val = treatment if treatment and treatment != "Both" else "Default" 
                if treatment == "Both" and not is_both_treatment: # Config said "Both" but structure isn't Stufe1/2
                    logger.warning(f"Config treatment is 'Both' but table structure is not Stufe 1/2. Applying '{current_treatment_val}'.")

                for data_col_idx, trial_name_loc in trial_names:
                    if data_col_idx >= len(row_data):
                        logger.warning(f"Data column index {data_col_idx} out of bounds for row_data (len {len(row_data)}) for variety '{variety}'. Skipping.")
                        continue
                    
                    value = self._clean_value(row_data[data_col_idx])
                    if value is not None:
                        transformed_rows.append({
                            'Year': year, 'Trial': f"{year}_whw_de_prt_lsv", 'Trait': trait_str,
                            'Variety': variety, 'Location': trial_name_loc, 'Treatment': current_treatment_val,
                            'RelativeValue': None, 'AbsoluteValue': value, 'Source': source
                        })
                        logger.debug(f"Added: Var={variety}, Loc={trial_name_loc}, Treat={current_treatment_val}, Val={value}")
        return transformed_rows

    def _clean_value(self, value: Any) -> Optional[float]:
        """Clean and convert value to float, handling German number format."""
        if value is None:
            return None
        try:
            # Convert to string and normalize German number format
            value_str = str(value)
            # Replace German thousand separator (dot) with nothing
            value_str = value_str.replace('.', '')
            # Replace German decimal separator (comma) with dot
            value_str = value_str.replace(',', '.')
            # Remove any remaining non-numeric characters except decimal point and minus sign
            cleaned = re.sub(r'[^\d.-]', '', value_str)
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize German text."""
        if not text:
            return ""
        # Remove special characters but keep German characters
        cleaned = re.sub(r'[^a-zA-ZäöüÄÖÜß0-9\s\-]', '', str(text))
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def transform_tables(self, tables_from_processor: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform extracted table data into standardized format."""
        all_transformed_data = []
        
        for table_idx, table_data_dict in enumerate(tables_from_processor):
            try:
                logger.info(f"[cyan]Processing table {table_idx + 1}[/cyan]")
                year_val = table_data_dict.get('year')
                table_type = table_data_dict.get('table_type')
                
                if year_val is None:
                    logger.warning(f"Skipping table transformation, year is None. Source: {table_data_dict.get('source')}")
                    continue
                if not isinstance(year_val, int):
                    try:
                        year_val = int(year_val)
                    except ValueError:
                        logger.warning(f"Skipping table transformation, year '{year_val}' is not a valid integer. Source: {table_data_dict.get('source')}")
                        continue
>>>>>>> aad3dccbd548be587baef5c04477742559373fa6

                logger.info(f"Table type: {table_type}, Year: {year_val}")

                if table_type == "relative":
                    transformed_rows = self._transform_relative_table(year_val, table_data_dict)
                elif table_type == "absolute":
                    headers = table_data_dict.get('headers', [])
                    rows = table_data_dict.get('rows', [])
                    transformed_rows = self._transform_absolute_table(
                        year_val, headers, rows,
                        table_data_dict.get('treatment'),
                        table_data_dict.get('source'),
                        table_data_dict.get('trait')
                    )
                else:
                    logger.warning(f"Unknown table type: {table_type}")
                    continue

                logger.info(f"Transformed {len(transformed_rows)} rows from table {table_idx + 1}")
                all_transformed_data.extend(transformed_rows)
                
<<<<<<< HEAD
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
            
    def _reconstruct_vertical_text(self, text: str) -> str:
        """
        Reconstruct vertical text into proper location name.
        Only attempts reconstruction if text appears vertical (contains newlines).
        Otherwise returns the original text.
        """
        if not text:
            return ""
        
        # If text doesn't contain newlines, it's likely already horizontal - return as is
        if '\n' not in str(text):
            return str(text).strip()
            
        # More careful check for separator lines
        if re.match(r'^[-=_*\s]+$', str(text)) and len(set(str(text).strip())) <= 2:
            logger.debug(f"Skipping separator line: {repr(text)}")
            return ""
        
        # For logging purposes, store the original text
        original_text = text
            
        # Special handling for vertical text patterns
        # Map of scrambled patterns to actual location names
        vertical_patterns = {
            # Original patterns with newlines
            'r )\nui h\n-B nic\nn e\ne v\np r\nr ö\ne N\nK (': 'Kerpen-Buir (Nörvenich)',
            'z- h\nn t\ne a\nel nr\nk e\nr V\nE': 'Erkelenz-Venrath', 
            't\nr\ne\nw\nel\nt\nt\nMi': 'Mittelwert',
            'e\ns .)\ns h\nü g\nD n\ns ti\nu s\na O\nH (': 'Haus Düsse (Oberkassel)',
            'n\ne- e\ng d\na ei\nL H': 'Hagen-Leidenhausen',
            'n\ne\nv\ne\nr\nG': 'Greven',
            '-\nn n\nei e\nt g\ns a\nWar All': 'Warstein-Allagen',
            'n\n- e\ng f\nr ö\ne h\nb n\nm e\no st\nBl ol\nH': 'Blomberg-Hohenfels',
            'm *\nz\ne t\nt a\nr s\ne n\nzi ei\nu z\nd t\ne u\nr h\nei sc\nb n\nze\ng\na n\ntr a\nEr Pfl': 'Ertrag bei angepasstem Pflanzenschutz',
            
            # New locations patterns from Winterweizen_Spelzweizen_2023 PDF
            'MÜ /\nBiedesheim': 'MÜ / Biedesheim',
            'NW / Herx-\nheim': 'NW / Herx-heim',
            'KH /\nWallertheim': 'KH / Wallertheim', 
            'SIM /\nKümbdchen': 'SIM / Kümbdchen',
            
            # Patterns after PDF processor cleaning (spaces between characters)
            'r ) ui h -B nic n e e v p r r ö e N K (': 'Kerpen-Buir (Nörvenich)',
            'r ) uih -Bnicn ee vp rr öe NK (': 'Kerpen-Buir (Nörvenich)',
            'z- h n t e a el nr k e r V E': 'Erkelenz-Venrath',
            'z- hn te aelnrk er VE': 'Erkelenz-Venrath',
            't r e w el t t Mi': 'Mittelwert',
            'e s .) s h ü g D n s ti u s a O H (': 'Haus Düsse (Oberkassel)',
            'es .) sh üg Dn stiu sa OH (': 'Haus Düsse (Oberkassel)',
            'n e- e g d a ei L H': 'Hagen-Leidenhausen',
            'ne- eg da eiL H': 'Hagen-Leidenhausen',
            'n e v e r G': 'Greven',
            '- n n ei e t g s a War All': 'Warstein-Allagen',
            '- nn eie tg sa WarAll': 'Warstein-Allagen',
            'n - e g f r ö e h b n m e o st Bl ol H': 'Blomberg-Hohenfels',
            'n - eg fr öe hb nm eo stBlolH': 'Blomberg-Hohenfels',
            'm * z e t t a r s e n zi ei u z d t e u r h ei sc b n ze g a n tr a Er Pfl': 'Ertrag bei angepasstem Pflanzenschutz',
            'm * ze tt ar se nzieiu zd te ur heiscb nzeg an tra ErPfl': 'Ertrag bei angepasstem Pflanzenschutz',
            
            # New locations with spaces
            'MÜ / Biedesheim': 'MÜ / Biedesheim',
            'NW / Herx- heim': 'NW / Herx-heim',
            'KH / Wallertheim': 'KH / Wallertheim', 
            'SIM / Kümbdchen': 'SIM / Kümbdchen',
            
            # Add pattern for Differenz
            'm . h e c t s s n s a e p z n e g a n fl a P u m z e z v n si e n er e ff nt Di /i': 'Differenz',
            'm . h e c t s s n s a e p z n e g a n fl a P u m z e z v n si e n er e ff nt Di': 'Differenz',
        }
        
        # Check if text matches any known vertical pattern exactly
        if text in vertical_patterns:
            return vertical_patterns[text]
        
        # Try to reconstruct from vertical text
        # Remove newlines and extra spaces to form a clean string
        clean_text = re.sub(r'\s+', '', text)
        
        # Check if the cleaned text matches any known mapping (using regex)
        location_mappings = {
            r'r\)uih-Bnicneevprr[öo]eNK\(\)': 'Kerpen-Buir (Nörvenich)',
            r'z-hnte[ae]nrelkr[ne]rVE': 'Erkelenz-Venrath',
            r'trewelttMi': 'Mittelwert',
            r'es[.)]sh[üu]gDnsti[ou]saOH\(\)': 'Haus Düsse (Oberkassel)',
            r'ne-egdaeiLH': 'Hagen-Leidenhausen',
            r'neverG': 'Greven',
            r'-nneietgsaWarAll': 'Warstein-Allagen',
            r'n-egfr[öo]ehbnmeostBlolH': 'Blomberg-Hohenfels',
            r'mzettarsenzieiuzdteurheiscbnzegantraErPfl': 'Ertrag bei angepasstem Pflanzenschutz',
            
            # New location regex patterns
            r'MÜ/Biedesheim': 'MÜ / Biedesheim',
            r'NW/Herx-heim': 'NW / Herx-heim',
            r'KH/Wallertheim': 'KH / Wallertheim',
            r'SIM/Kümbdchen': 'SIM / Kümbdchen',
            
            # Add pattern for Differenz
            r'm\.hectssnsaepznenganflaPumzezvnsienereffntDi': 'Differenz',
            r'm\.hectssnsaepznenganflaPumzezvnsienereffntDi/i': 'Differenz',
        }
        
        for pattern, location in location_mappings.items():
            try:
                if re.match(pattern, clean_text, re.IGNORECASE):
                    return location
            except re.error as e:
                logger.warning(f"Regex error with pattern '{pattern}': {str(e)}")
                continue
        
        # If no match but contains newlines, try to create a meaningful location
        if '\n' in text:
            # Try to join text lines intelligently
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            joined = ' '.join(lines)
            
            # If joined text looks reasonable, use it
            if len(joined) >= 3 and not re.match(r'^[-=_*\s]+$', joined):
                logger.info(f"Created location from vertical text: {joined} (original: {repr(text)})")
                return joined
                
            # Log unmapped vertical pattern but don't discard the data
            formatted_original = repr(original_text).replace('\\n', '\\n\n')
            logger.warning(f"[yellow]UNMAPPED VERTICAL PATTERN:[/yellow] {formatted_original}")
    
        # Return the original text if we couldn't transform it
        # This ensures we don't lose data when pattern doesn't match
        return text.strip()
    
    def reconstruct_vertical_text(self, text_from_cell: str) -> Optional[str]:
        """
        Reconstruct vertical text from cell content.
        Calls the comprehensive _reconstruct_vertical_text implementation.
        """
        if not text_from_cell:
            return None
        
        # Use the comprehensive implementation
        result = self._reconstruct_vertical_text(text_from_cell)
        
        # Return None if empty result
        if not result or not result.strip():
            return None
            
        return result

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
        
        # Skip separator lines (sequences of just dashes, equals signs, or other separator characters)
        if re.match(r'^[-=_*\s]+$', str(location)):
            return ""
            
        # If location appears to be vertical text
        if '\n' in str(location) or isinstance(location, str) and ' ' in location and len(location.split()) > 3:
            reconstructed = self._reconstruct_vertical_text(location)
            if reconstructed:
                return reconstructed
                
        return str(location).strip()
=======
            except Exception as e:
                logger.error(f"Error transforming table {table_idx + 1}: {str(e)}")
                continue

        if not all_transformed_data:
            logger.warning("No data was transformed successfully.")
            return pd.DataFrame(columns=self.standard_columns)

        df = pd.DataFrame(all_transformed_data)
        logger.info(f"Final DataFrame shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Sample data:\n{df.head()}")
        return df
>>>>>>> aad3dccbd548be587baef5c04477742559373fa6
