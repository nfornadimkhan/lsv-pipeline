"""
Module for transforming extracted table data into standardized format.
"""
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
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
        Example: 'r )\nui h\n-B nic\nn e\ne v\np r\nr ö\ne N\nK (' -> 'Köln-Bonn'
        """
        if not text:
            return ""
            
        # Special handling for vertical text patterns
        # Map of scrambled patterns to actual location names
        vertical_patterns = {
            # Original patterns with newlines
            'r )\nui h\n-B nic\nn e\ne v\np r\nr ö\ne N\nK (': 'Köln-Bonn',
            'z- h\nn t\ne a\nel nr\nk e\nr V\nE': 'Erkelenz-Venrath', 
            't\nr\ne\nw\nel\nt\nt\nMi': 'Mittelwert',
            'e\ns .)\ns h\nü g\nD n\ns ti\nu s\na O\nH (': 'Haus Düsse (Oberkassel)',
            'n\ne- e\ng d\na ei\nL H': 'Hagen-Leidenhausen',
            'n\ne\nv\ne\nr\nG': 'Greven',
            '-\nn n\nei e\nt g\ns a\nWar All': 'Warstein-Allagen',
            'n\n- e\ng f\nr ö\ne h\nb n\nm e\no st\nBl ol\nH': 'Blomberg-Hohenfels',
            'm *\nz\ne t\nt a\nr s\ne n\nzi ei\nu z\nd t\ne u\nr h\nei sc\nb n\nze\ng\na n\ntr a\nEr Pfl': 'Ertrag bei angepasstem Pflanzenschutz',
            # Patterns after PDF processor cleaning (spaces between characters)
            'r ) ui h -B nic n e e v p r r ö e N K (': 'Köln-Bonn',
            'r ) uih -Bnicn ee vp rr öe NK (': 'Köln-Bonn',
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
            'm * ze tt ar se nzieiu zd te ur heiscb nzeg an tra ErPfl': 'Ertrag bei angepasstem Pflanzenschutz'
        }
        
        # Check if text matches any known vertical pattern exactly
        if text in vertical_patterns:
            return vertical_patterns[text]
        
        # Try to reconstruct from vertical text
        # Remove newlines and extra spaces
        clean_text = re.sub(r'\s+', '', text)
        
        # Known location mappings based on character sequences
        location_mappings = {
            'r)uih-Bnicneevprr[öo]eNK[(]': 'Köln-Bonn',
            'z-hnte[ae]nrelkr[ne]rVE': 'Erkelenz-Venrath',
            'trewelttMi': 'Mittelwert',
            'es[.)]sh[üu]gDnsti[ou]saOH[(]': 'Haus Düsse (Oberkassel)',
            'ne-egdaeiLH': 'Hagen-Leidenhausen',
            'neverG': 'Greven',
            '-nneietgsaWarAll': 'Warstein-Allagen',
            'n-egfr[öo]ehbnmeostBlolH': 'Blomberg-Hohenfels',
            'mzettarsenzieiuzdteurheiscbnzegantraErPfl': 'Ertrag bei angepasstem Pflanzenschutz'
        }
        
        # Check if the cleaned text matches any known mapping (using regex)
        for pattern, location in location_mappings.items():
            if re.match(pattern, clean_text, re.IGNORECASE):
                return location
            
        # If no exact match, try to identify patterns
        # Handle Mittelwert pattern
        if 'mittel' in clean_text.lower() or ('mi' in clean_text.lower() and 'wert' in clean_text.lower()):
            return 'Mittelwert'
            
        # For Greven
        if clean_text.lower() in ['greven', 'neverG', 'neverG'.lower()]:
            return 'Greven'
            
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

        # Find trial names (locations) from the table
        trial_names = []
        for row_idx, row_data in enumerate(full_table_data):
            if row_idx < actual_reference_row_idx:  # Look in header rows
                if row_data and len(row_data) > 1 and row_data[1] and "Versuch" in str(row_data[1]):
                    logger.info(f"Found trial row at index {row_idx}: {row_data}")
                    # Extract trial names from this row
                    for col_idx in range(2, len(row_data)):
                        if col_idx < len(row_data) and row_data[col_idx]:
                            raw_trial_name = str(row_data[col_idx]).strip()
                            if raw_trial_name and not any(x in raw_trial_name.lower() for x in ['mittel', 'mw', 'versuch']):
                                trial_name = self._reconstruct_vertical_text(raw_trial_name)
                                if trial_name:
                                    trial_names.append(trial_name)
                                    logger.info(f"Extracted trial name: {trial_name} (from: {raw_trial_name}) at column {col_idx}")

        logger.info(f"Extracted trial names: {trial_names}")

        # Process data rows
        for row_idx, row_data in enumerate(full_table_data):
            if row_idx <= actual_reference_row_idx:  # Skip header rows and reference row
                continue
                
            if not row_data or len(row_data) < 2:
                continue
                
            variety = self._clean_text(row_data[1] if len(row_data) > 1 else row_data[0])
            if not variety or "Mittel" in variety or variety in self.exclude_rows:
                logger.debug(f"Skipping row {row_idx}: {variety} (excluded or mean value)")
                continue

            logger.info(f"Processing variety: {variety} at row {row_idx}")

            # Process each data column
            for col_idx in range(2, len(row_data)):
                if col_idx >= len(reference_row_content):
                    continue
                    
                value = self._clean_value(row_data[col_idx])
                if value is not None:
                    # Get trial name for this column
                    trial_name = trial_names[col_idx - 2] if col_idx - 2 < len(trial_names) else f"Location_{col_idx}"
                    
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

    def _transform_absolute_table(self, year: int, headers: list, rows: list, treatment: Optional[str] = None,
                                  source: Optional[str] = None, trait: Optional[str] = None) -> list:
        transformed_rows = []
        
        # Log table structure
        logger.info(f"[cyan]Absolute Table Structure Analysis:[/cyan]")
        logger.info(f"Headers: {headers}")
        logger.info(f"Total rows: {len(rows)}")
        for idx, row in enumerate(rows):
            logger.info(f"Row {idx}: {row}")

        # Extract trial names from headers
        trial_names = []
        for i, h in enumerate(headers):
            if h and not any(x in str(h).lower() for x in ['sorte', 'qualität', 'mittel', 'mw']) and str(h).strip() not in self.exclude_columns:
                trial_name = self._reconstruct_vertical_text(str(h))
                if trial_name:
                    trial_names.append((i, trial_name))
                    logger.info(f"Found trial name: {trial_name} (from: {h}) at column {i}")

        logger.info(f"Extracted trial names: {[name for _, name in trial_names]}")
        
        # Use trait from config or default
        trait_str = trait if trait else "Default Trait"
        
        for row_idx, row in enumerate(rows):
            if not row or any("Mittel" in str(cell) for cell in row):
                logger.debug(f"Skipping row {row_idx}: Contains mean value")
                continue
                
            variety = self._normalize_german_text(row[0])
            if not variety or "Mittel" in variety or variety in self.exclude_rows:
                logger.debug(f"Skipping variety: {variety} (excluded or mean value)")
                continue

            logger.info(f"Processing variety: {variety} at row {row_idx}")

            if treatment and treatment != 'Both':
                for col_idx, trial_name in trial_names:
                    if col_idx >= len(row):
                        continue
                    value = self._clean_value(row[col_idx])
                    if value is not None:
                        transformed_rows.append({
                            'Year': year,
                            'Trial': f"{year}_whw_de_prt_lsv",
                            'Trait': trait_str,
                            'Variety': variety,
                            'Location': trial_name,
                            'Treatment': treatment,
                            'RelativeValue': None,
                            'AbsoluteValue': value,
                            'Source': source
                        })
                        logger.info(f"Added data point: Variety={variety}, Location={trial_name}, Value={value}")
            else:
                for col_idx, trial_name in trial_names:
                    if col_idx >= len(row):
                        continue
                    st1_idx = col_idx + 1
                    st2_idx = col_idx + 2
                    if st1_idx < len(row) and row[st1_idx]:
                        value = self._clean_value(row[st1_idx])
                        if value is not None:
                            transformed_rows.append({
                                'Year': year,
                                'Trial': f"{year}_whw_de_prt_lsv",
                                'Trait': trait_str,
                                'Variety': variety,
                                'Location': trial_name,
                                'Treatment': 'St 1',
                                'RelativeValue': None,
                                'AbsoluteValue': value,
                                'Source': source
                            })
                            logger.info(f"Added data point: Variety={variety}, Location={trial_name}, Treatment=St 1, Value={value}")
                    if st2_idx < len(row) and row[st2_idx]:
                        value = self._clean_value(row[st2_idx])
                        if value is not None:
                            transformed_rows.append({
                                'Year': year,
                                'Trial': f"{year}_whw_de_prt_lsv",
                                'Trait': trait_str,
                                'Variety': variety,
                                'Location': trial_name,
                                'Treatment': 'St 2',
                                'RelativeValue': None,
                                'AbsoluteValue': value,
                                'Source': source
                            })
                            logger.info(f"Added data point: Variety={variety}, Location={trial_name}, Treatment=St 2, Value={value}")

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