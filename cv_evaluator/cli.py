"""
Command-line interface for CV evaluation system.

Created by: Zied Boughdir (@zinzied)
GitHub: https://github.com/zinzied/cv-evaluation-system
"""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
import json

from .core.evaluator import CVEvaluator
from .core.batch_processor import BatchProcessor
from .core.criteria_loader import criteria_manager
from .utils.exceptions import CVEvaluatorError
from .utils.config import config

app = typer.Typer(
    name="cv-evaluator",
    help="AI-powered CV evaluation and analysis system by Zied Boughdir (@zinzied)",
    add_completion=False
)
console = Console()


@app.command()
def evaluate(
    cv_path: str = typer.Argument(..., help="Path to CV PDF file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output report path"),
    format: str = typer.Option("pdf", "--format", "-f", help="Report format (pdf, word, html)"),
    criteria: str = typer.Option("default", "--criteria", "-c", help="Evaluation criteria name"),
    job_template: Optional[str] = typer.Option(None, "--job-template", "-j", help="Job template"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Evaluate a single CV file."""
    
    if verbose:
        console.print(f"[blue]Evaluating CV:[/blue] {cv_path}")
        console.print(f"[blue]Criteria:[/blue] {criteria}")
        if job_template:
            console.print(f"[blue]Job Template:[/blue] {job_template}")
    
    try:
        # Initialize evaluator
        evaluator = CVEvaluator(criteria_name=criteria, job_template=job_template)
        
        # Validate input file
        if not Path(cv_path).exists():
            console.print(f"[red]Error:[/red] CV file not found: {cv_path}")
            raise typer.Exit(1)
        
        # Generate output path if not provided
        if not output:
            cv_file = Path(cv_path)
            output = f"{cv_file.stem}_evaluation.{format}"
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating CV...", total=None)
            
            # Evaluate CV and generate report
            result = evaluator.evaluate_and_report(
                cv_path, output, format=format
            )
        
        # Display results
        _display_evaluation_result(result['analysis_result'])
        
        console.print(f"\n[green]✓ Report generated:[/green] {result['report_path']}")
        
    except CVEvaluatorError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing CV files"),
    output_dir: str = typer.Argument(..., help="Output directory for reports"),
    pattern: str = typer.Option("*.pdf", "--pattern", "-p", help="File pattern to match"),
    format: str = typer.Option("pdf", "--format", "-f", help="Report format"),
    criteria: str = typer.Option("default", "--criteria", "-c", help="Evaluation criteria"),
    job_template: Optional[str] = typer.Option(None, "--job-template", "-j", help="Job template"),
    no_reports: bool = typer.Option(False, "--no-reports", help="Skip individual reports"),
    max_workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Max concurrent workers")
):
    """Process multiple CV files in batch."""
    
    console.print(f"[blue]Processing CVs from:[/blue] {input_dir}")
    console.print(f"[blue]Output directory:[/blue] {output_dir}")
    console.print(f"[blue]File pattern:[/blue] {pattern}")
    
    try:
        # Initialize batch processor
        evaluation_criteria = criteria_manager.get_criteria(criteria, job_template)
        processor = BatchProcessor(
            evaluation_criteria=evaluation_criteria,
            max_workers=max_workers
        )
        
        # Process directory
        results = processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            file_pattern=pattern,
            generate_reports=not no_reports,
            report_format=format
        )
        
        # Display results
        _display_batch_results(results)
        
    except CVEvaluatorError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list_templates():
    """List available job templates."""
    
    templates = criteria_manager.list_job_templates()
    
    if not templates:
        console.print("[yellow]No job templates available[/yellow]")
        return
    
    table = Table(title="Available Job Templates")
    table.add_column("Template Name", style="cyan")
    
    for template in templates:
        table.add_row(template)
    
    console.print(table)


@app.command()
def list_criteria():
    """List available evaluation criteria."""
    
    criteria_list = criteria_manager.list_available_criteria()
    
    if not criteria_list:
        console.print("[yellow]No criteria configurations available[/yellow]")
        return
    
    table = Table(title="Available Evaluation Criteria")
    table.add_column("Criteria Name", style="cyan")
    
    for criteria in criteria_list:
        table.add_row(criteria)
    
    console.print(table)


@app.command()
def validate(
    cv_path: str = typer.Argument(..., help="Path to CV file to validate")
):
    """Validate if a CV file can be processed."""
    
    try:
        evaluator = CVEvaluator()
        
        if evaluator.validate_cv_file(cv_path):
            console.print(f"[green]✓ Valid CV file:[/green] {cv_path}")
        else:
            console.print(f"[red]✗ Invalid CV file:[/red] {cv_path}")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error validating file:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def config_info():
    """Display current configuration information."""
    
    console.print(Panel.fit(
        f"[bold]CV Evaluator Configuration[/bold]\n\n"
        f"App Name: {config.app_name}\n"
        f"Version: {config.app_version}\n"
        f"Debug Mode: {config.debug}\n"
        f"Scoring Scale: {config.scoring_scale}\n"
        f"Temp Directory: {config.temp_dir}\n"
        f"Log Level: {config.log_level}",
        title="Configuration"
    ))


def _display_evaluation_result(analysis_result):
    """Display evaluation results in a formatted table."""
    
    # Overall score panel
    score_panel = Panel.fit(
        f"[bold green]{analysis_result.overall_score:.1f}/100[/bold green]\n"
        f"Job Fit: {analysis_result.fit_percentage:.1f}%",
        title="Overall Score"
    )
    console.print(score_panel)
    
    # Section scores table
    table = Table(title="Section Scores")
    table.add_column("Section", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Feedback", style="dim")
    
    for score in analysis_result.section_scores:
        percentage = (score.score / score.max_score) * 100
        table.add_row(
            score.section.title(),
            f"{score.score:.1f}/{score.max_score:.1f}",
            f"{percentage:.1f}%",
            score.feedback[:50] + "..." if len(score.feedback) > 50 else score.feedback
        )
    
    console.print(table)
    
    # Candidate info
    personal_info = analysis_result.cv_data.personal_info
    if personal_info.name:
        console.print(f"\n[bold]Candidate:[/bold] {personal_info.name}")
    if personal_info.email:
        console.print(f"[bold]Email:[/bold] {personal_info.email}")
    
    # Strengths and weaknesses
    if analysis_result.strengths:
        console.print(f"\n[bold green]Strengths:[/bold green]")
        for strength in analysis_result.strengths:
            console.print(f"  • {strength}")
    
    if analysis_result.weaknesses:
        console.print(f"\n[bold red]Areas for Improvement:[/bold red]")
        for weakness in analysis_result.weaknesses:
            console.print(f"  • {weakness}")


def _display_batch_results(results):
    """Display batch processing results."""
    
    # Summary panel
    summary_panel = Panel.fit(
        f"[bold]Total Files:[/bold] {results['total_files']}\n"
        f"[bold green]Successful:[/bold green] {results['successful']}\n"
        f"[bold red]Failed:[/bold red] {results['failed']}\n"
        f"[bold]Success Rate:[/bold] {(results['successful']/results['total_files']*100):.1f}%",
        title="Batch Processing Summary"
    )
    console.print(summary_panel)
    
    # Results table
    table = Table(title="Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Candidate", style="dim")
    
    for result in results['results']:
        if result['success']:
            status = "[green]✓[/green]"
            score = f"{result['overall_score']:.1f}"
            candidate = result.get('candidate_name', 'Unknown')[:30]
        else:
            status = "[red]✗[/red]"
            score = "N/A"
            candidate = "Error"
        
        file_name = Path(result['file_path']).name
        table.add_row(file_name, status, score, candidate)
    
    console.print(table)
    
    if results['summary_report']:
        console.print(f"\n[green]✓ Summary report:[/green] {results['summary_report']}")
    console.print(f"[green]✓ Output directory:[/green] {results['output_directory']}")


if __name__ == "__main__":
    app()
