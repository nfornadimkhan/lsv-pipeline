#!/usr/bin/env python3
"""
Main script for PDF processing pipeline.
"""
import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from pdf_processor import PDFProcessor
from data_transformer import DataTransformer
from config_manager import ConfigManager
from logging_config import setup_logging

# Initialize Typer app
app = typer.Typer(
    name="PDF Data Extractor",
    help="Extract structured data from PDF files containing agricultural trial results.",
    add_completion=False
)

# Initialize Rich console
console = Console()

def get_default_paths() -> tuple[Path, Path, Path, Path]:
    """Get default paths for input/output directories and config files."""
    base_dir = Path.cwd()
    return (
        base_dir / "input",  # input_dir
        base_dir / "output",  # output_dir
        base_dir / "config" / "structure.yaml",  # structure_config
        base_dir / "config" / "exclusions.yaml",  # exclusions_config
    )

@app.command()
def process_pdfs(
    input_dir: Path = typer.Option(
        None,
        "--input-dir", "-i",
        help="Directory containing PDF files",
        show_default="input/"
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory for output Excel files",
        show_default="output/"
    ),
    structure_config: Path = typer.Option(
        None,
        "--structure-config", "-s",
        help="Path to structure configuration file",
        show_default="config/structure.yaml"
    ),
    exclusions_config: Path = typer.Option(
        None,
        "--exclusions-config", "-e",
        help="Path to exclusions configuration file",
        show_default="config/exclusions.yaml"
    )
):
    """Process PDF files and extract data according to configuration."""
    # Setup logging
    logger = setup_logging()
    
    # Get default paths if not provided
    default_input, default_output, default_structure, default_exclusions = get_default_paths()
    
    # Use provided paths or defaults
    input_dir = input_dir or default_input
    output_dir = output_dir or default_output
    structure_config = structure_config or default_structure
    exclusions_config = exclusions_config or default_exclusions
    
    try:
        # Log configuration
        logger.info("[cyan]Starting PDF processing pipeline with configuration:[/cyan]")
        logger.info(f"[cyan]Input directory:[/cyan] {input_dir}")
        logger.info(f"[cyan]Output directory:[/cyan] {output_dir}")
        logger.info(f"[cyan]Structure config:[/cyan] {structure_config}")
        logger.info(f"[cyan]Exclusions config:[/cyan] {exclusions_config}")
        
        # Create directories if they don't exist
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        config_manager = ConfigManager(structure_config, exclusions_config)
        pdf_processor = PDFProcessor()
        data_transformer = DataTransformer(config_manager)
        
        # Get list of PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"[yellow]No PDF files found in {input_dir}[/yellow]")
            return
            
        logger.info(f"[cyan]Found {len(pdf_files)} PDF files to process[/cyan]")
        
        # Process each PDF file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for pdf_file in pdf_files:
                task = progress.add_task(f"Processing {pdf_file.name}...", total=None)
                
                try:
                    # Extract tables from PDF
                    tables = pdf_processor.extract_tables(pdf_file, config_manager)
                    
                    if not tables:
                        logger.warning(f"[yellow]No tables extracted from {pdf_file.name}[/yellow]")
                        continue
                    
                    # Transform tables
                    df = data_transformer.transform_tables(tables)
                    
                    if df.empty:
                        logger.warning(f"[yellow]No valid data extracted from {pdf_file.name}[/yellow]")
                        continue
                    
                    # Get year from config
                    year = config_manager.get_pdf_year(pdf_file.name)
                    if not year:
                        logger.warning(f"[yellow]No year configuration found for {pdf_file.name}[/yellow]")
                        continue
                        
                    # Save to Excel
                    output_file = output_dir / f"{year}_ww_de_prt_lsv.xlsx"
                    df.to_excel(output_file, index=False)
                    logger.info(f"[green]Successfully processed {pdf_file.name} -> {output_file}[/green]")
                    
                except Exception as e:
                    logger.error(f"[red]Error processing {pdf_file.name}: {str(e)}[/red]")
                    continue
                    
                progress.update(task, completed=True)
                
        logger.info("[green]PDF processing completed successfully![/green]")
        
    except Exception as e:
        logger.error(f"[red]Fatal error: {str(e)}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 