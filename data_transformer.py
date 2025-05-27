"""
Module for transforming extracted table data into standardized format.
"""
import pandas as pd
import logging
from typing import List, Dict, Any
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
        
    def transform_tables(self, tables: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Transform extracted tables into standardized format.
        
        Args:
            tables: List of extracted tables with metadata
            
        Returns:
            DataFrame in standardized format
        """
        transformed_data = []
        
        for table in tables:
            try:
                year = table.get('year')
                table_type = table.get('table_type')
                treatment = table.get('treatment')
                source = table.get('source')
                if table_type == 'kornertrag_relativ':
                    full_table = table.get('full_table')
                    reference_row = table.get('reference_row')
                    reference_row_idx = table.get('reference_row_idx')
                    transformed_data.extend(
                        self._transform_kornertrag_relativ_table(year, full_table, reference_row, reference_row_idx, treatment, source)
                    )
                elif table_type == 'kornertrag_absolut':
                    headers = table['headers']
                    rows = table['rows']
                    transformed_data.extend(
                        self._transform_kornertrag_absolut_table(year, headers, rows, treatment, source)
                    )
                else:
                    headers = table['headers']
                    rows = table['rows']
                    if self._is_kornertrag_table(headers):
                        transformed_data.extend(
                            self._transform_kornertrag_absolut_table(year, headers, rows, treatment, source)
                        )
                    elif self._is_ertrage_table(headers):
                        transformed_data.extend(
                            self._transform_ertrage_table(year, headers, rows, treatment, source)
                        )
            except Exception as e:
                logger.error(f"Error transforming table: {str(e)}")
                continue
        # Create DataFrame
        df = pd.DataFrame(transformed_data, columns=self.standard_columns)
        # Clean and validate data
        df = self._clean_data(df)
        return df
        
    def _is_kornertrag_table(self, headers: List[str]) -> bool:
        """Check if table is a Kornertrag table."""
        return any("Kornertrag absolut" in str(h) for h in headers)
        
    def _is_ertrage_table(self, headers: List[str]) -> bool:
        """Check if table is an Erträge table."""
        return any("Erträge" in str(h) for h in headers) and any("Absoluter Ertrag" in str(h) for h in headers)
        
    def _transform_kornertrag_table(self, year: int, headers: List[str], rows: List[List[str]], treatment: str = None, source: str = None) -> List[Dict[str, Any]]:
        transformed_rows = []
        location_cols = [i for i, h in enumerate(headers) if h and not any(x in str(h).lower() for x in ['sorte', 'qualität'])]
        for row in rows:
            if not row or any("Mittel" in str(cell) for cell in row):
                continue
            variety = row[0]
            if not variety or "Mittel" in str(variety):
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
                            'Trait': "Kornertrag absolut, Sorten, Orte und Behandlungen",
                            'Variety': variety,
                            'Location': location,
                            'Treatment': treatment,
                            'Value': value,
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
                            'Trait': "Kornertrag absolut, Sorten, Orte und Behandlungen",
                            'Variety': variety,
                            'Location': location,
                            'Treatment': 'St 1',
                            'Value': self._clean_value(row[st1_idx]),
                            'Source': source
                        })
                    if st2_idx < len(row) and row[st2_idx]:
                        transformed_rows.append({
                            'Year': year,
                            'Trial': f"{year}_whw_de_prt_lsv",
                            'Trait': "Kornertrag absolut, Sorten, Orte und Behandlungen",
                            'Variety': variety,
                            'Location': location,
                            'Treatment': 'St 2',
                            'Value': self._clean_value(row[st2_idx]),
                            'Source': source
                        })
        return transformed_rows
        
    def _transform_ertrage_table(self, year: int, headers: List[str], rows: List[List[str]], treatment: str = None, source: str = None) -> List[Dict[str, Any]]:
        transformed_rows = []
        year_cols = [i for i, h in enumerate(headers) if str(year) in str(h)]
        for row in rows:
            if not row or any("Mittel" in str(cell) for cell in row):
                continue
            variety = row[0]
            if not variety or "Mittel" in str(variety):
                continue
            for year_idx in year_cols:
                if year_idx < len(row) and row[year_idx]:
                    transformed_rows.append({
                        'Year': year,
                        'Trial': f"{year}_whw_de_prt_lsv",
                        'Trait': "Erträge, Absoluter Ertrag",
                        'Variety': variety,
                        'Location': headers[year_idx - 1] if year_idx > 0 else "Unknown",
                        'Treatment': treatment if treatment and treatment != 'Both' else 'Standard',
                        'Value': self._clean_value(row[year_idx]),
                        'Source': source
                    })
        return transformed_rows
        
    def _transform_kornertrag_relativ_table(self, year: int, full_table: list, reference_row: list, reference_row_idx: int, treatment: str = None, source: str = None) -> list:
        transformed_rows = []
        location_row = None
        for row in full_table:
            if row and any("Anbaugebiet" in str(cell) for cell in row):
                location_row = row
                break
        if not location_row:
            return transformed_rows
        location_cols = []
        locations = []
        for idx, cell in enumerate(location_row):
            if idx < 2:
                continue
            if cell and isinstance(cell, str) and cell.strip() != '' and not self._is_excluded(cell, self.exclude_columns):
                location_cols.append(idx)
                locations.append(cell.strip())
        reference_map = {}
        for idx in location_cols:
            ref_val = self._clean_value(reference_row[idx])
            if ref_val:
                reference_map[idx] = ref_val
        trait_str = "Kornertrag absolut, Sorten, Orte und Behandlungen"
        if year:
            trait_str += f", {year}"
        for row in full_table[reference_row_idx+1:]:
            if not row or not row[1]:
                continue
            variety = str(row[1]).strip()
            logger.debug(f"[yellow]Processing variety:[/yellow] {repr(variety)}")
            
            # Check for excluded patterns
            if self._is_excluded(variety, self.exclude_rows):
                logger.debug(f"[red]Excluded variety due to pattern match:[/red] {repr(variety)}")
                continue
                
            # Check for other exclusion conditions
            if any(x in variety.lower() for x in ["anbaugebiet", "boden", "vgl. reduziert", "rel. grenzdifferenz", 
                                                "e-sorten", "a-sorten", "b-sorten", "c-sorten", "anhang", 
                                                "durum (hartweizen)"]):
                logger.debug(f"[red]Excluded variety due to content match:[/red] {repr(variety)}")
                continue
                
            if not variety or variety.strip() == '':
                continue

            for loc_idx, location in zip(location_cols, locations):
                rel_val = self._clean_value(row[loc_idx])
                ref_val = reference_map.get(loc_idx)
                abs_val = rel_val * ref_val / 100.0 if rel_val is not None and ref_val is not None else None
                if rel_val is not None and ref_val is not None:
                    transformed_rows.append({
                        'Year': year,
                        'Trial': f"{year}_whw_de_prt_lsv" if year else "whw_de_prt_lsv",
                        'Trait': trait_str,
                        'Variety': variety,
                        'Location': location,
                        'Treatment': treatment if treatment and treatment != 'Both' else 'Standard',
                        'RelativeValue': rel_val,
                        'AbsoluteValue': abs_val,
                        'Source': source
                    })
        return transformed_rows
        
    def _transform_kornertrag_absolut_table(self, year: int, headers: list, rows: list, treatment: str = None, source: str = None) -> list:
        transformed_rows = []
        location_cols = [i for i, h in enumerate(headers) if h and not any(x in str(h).lower() for x in ['sorte', 'qualität']) and str(h).strip() not in self.exclude_columns]
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
                            'Trait': "Kornertrag absolut, Sorten, Orte und Behandlungen",
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
                            'Trait': "Kornertrag absolut, Sorten, Orte und Behandlungen",
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
                            'Trait': "Kornertrag absolut, Sorten, Orte und Behandlungen",
                            'Variety': variety,
                            'Location': location,
                            'Treatment': 'St 2',
                            'RelativeValue': None,
                            'AbsoluteValue': self._clean_value(row[st2_idx]),
                            'Source': source
                        })
        return transformed_rows
        
    def _clean_value(self, value: str) -> float:
        """Clean and convert value to float."""
        if not value:
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

    def _is_excluded(self, value, exclusion_patterns):
        """Check if a value matches any exclusion pattern."""
        if not value:
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