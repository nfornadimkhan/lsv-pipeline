"""
Module for managing configuration settings.
"""
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("pdf_pipeline")

class ConfigManager:
    def __init__(self, structure_path: Path = Path('config/structure.yaml'), exclusions_path: Path = Path('config/exclusions.yaml')):
        """
        Initialize configuration manager.
        Args:
            structure_path: Path to the YAML structure configuration file
            exclusions_path: Path to the YAML exclusions configuration file
        """
        self.structure_path = structure_path
        self.exclusions_path = exclusions_path
        self.structure = self._load_yaml(self.structure_path)
        self.exclusions = self._load_yaml(self.exclusions_path)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}

    def get_pdf_config(self, pdf_filename: str) -> Optional[Dict[str, Any]]:
        for pdf_config in self.structure.get('pdfs', []):
            if pdf_config['filename'] == pdf_filename:
                return pdf_config
        return None

    def get_pages_to_process(self, pdf_filename: str) -> List[Dict[str, Any]]:
        """Get list of pages to process for a given PDF filename."""
        for pdf_config in self.structure.get('pdfs', []):
            if pdf_config.get('filename') == pdf_filename:
                pages = pdf_config.get('pages', [])
                logger.debug(f"[green]Found configuration for {pdf_filename}: {len(pages)} pages configured[/green]")
                return pages
                
        logger.debug(f"[yellow]No configuration found for {pdf_filename}[/yellow]")
        return []

    def get_pdf_year(self, pdf_filename: str) -> Optional[int]:
        pdf_config = self.get_pdf_config(pdf_filename)
        if pdf_config:
            return pdf_config.get('year')
        return None

    def _parse_exclusion_patterns(self, patterns):
        parsed = []
        for item in patterns:
            if isinstance(item, dict):
                for k, v in item.items():
                    parsed.append({'type': k, 'value': v})
            elif isinstance(item, str):
                # Default to 'contains' if just a string
                parsed.append({'type': 'contains', 'value': item})
        return parsed

    def get_exclude_columns(self) -> list:
        patterns = self.exclusions.get('exclude_columns', [])
        return self._parse_exclusion_patterns(patterns)

    def get_exclude_rows(self) -> list:
        patterns = self.exclusions.get('exclude_rows', [])
        return self._parse_exclusion_patterns(patterns)