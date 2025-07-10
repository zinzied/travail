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
from .core.interactive_criteria import InteractiveCriteriaBuilder, CriteriaFromFiles
from .core.participant_evaluator import ParticipantEvaluator
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
def create_criteria():
    """Create custom evaluation criteria interactively."""

    try:
        builder = InteractiveCriteriaBuilder()
        criteria = builder.build_criteria_interactively()

        if criteria:
            # Ask for filename
            filename = typer.prompt("Enter filename for criteria (without extension)", default="custom_criteria")
            filepath = builder.save_criteria_to_file(criteria, filename)
            console.print(f"[green]✓ Criteria saved successfully![/green]")
            console.print(f"Use with: --criteria {filename}")

    except Exception as e:
        console.print(f"[red]Error creating criteria:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def criteria_from_files(
    files: List[str] = typer.Argument(..., help="Job description or requirement files"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output criteria filename")
):
    """Extract evaluation criteria from job description files."""

    try:
        extractor = CriteriaFromFiles()
        criteria = extractor.extract_criteria_from_files(files)

        if criteria:
            filename = output or "extracted_criteria"
            builder = InteractiveCriteriaBuilder()
            filepath = builder.save_criteria_to_file(criteria, filename)
            console.print(f"[green]✓ Criteria extracted and saved![/green]")
            console.print(f"Use with: --criteria {filename}")

    except Exception as e:
        console.print(f"[red]Error extracting criteria:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def evaluate_participant(
    participant_id: str = typer.Argument(..., help="Unique participant identifier"),
    files: List[str] = typer.Option(..., "--file", "-f", help="Participant files (format: path:type:description)"),
    criteria: str = typer.Option("default", "--criteria", "-c", help="Evaluation criteria"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Evaluate a participant with multiple files."""

    try:
        # Parse file specifications
        participant_files = []
        for file_spec in files:
            parts = file_spec.split(':')
            file_path = parts[0]
            file_type = parts[1] if len(parts) > 1 else 'other'
            description = parts[2] if len(parts) > 2 else ''

            participant_files.append({
                'path': file_path,
                'type': file_type,
                'description': description
            })

        if verbose:
            console.print(f"[blue]Evaluating participant:[/blue] {participant_id}")
            console.print(f"[blue]Files:[/blue] {len(participant_files)}")
            for pf in participant_files:
                console.print(f"  - {pf['path']} ({pf['type']})")

        # Load criteria
        evaluation_criteria = criteria_manager.get_criteria(criteria)

        # Create participant evaluator
        evaluator = ParticipantEvaluator(evaluation_criteria)

        # Add participant and files
        evaluator.add_participant_files(participant_id, participant_files)

        # Process and evaluate
        with console.status("Processing participant files..."):
            evaluator.process_participant_files(participant_id)
            result = evaluator.evaluate_participant(participant_id)

        if result:
            # Display results
            _display_participant_results(participant_id, result, evaluator)

            # Save results if output specified
            if output:
                output_path = Path(output)
                output_path.mkdir(parents=True, exist_ok=True)

                # Export detailed results
                results_file = output_path / f"{participant_id}_results.json"
                evaluator.export_results(str(results_file))

                # Generate report
                report_file = output_path / f"{participant_id}_report.pdf"
                evaluator.evaluator.generate_report(result, str(report_file))

                console.print(f"[green]✓ Results saved to:[/green] {output_path}")
        else:
            console.print(f"[red]Failed to evaluate participant[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def list_ai_models():
    """List available AI models for evaluation and chat."""

    from .ai.free_models import list_available_models

    console.print(Panel.fit(
        "[bold blue]Available AI Models[/bold blue]\n"
        "Free models for enhanced CV evaluation and chat",
        title="AI Models"
    ))

    available_models = list_available_models()

    if not any(available_models.values()):
        console.print("[yellow]No AI models are currently available.[/yellow]")
        console.print("\n[bold]To set up free AI models:[/bold]")
        console.print("1. [cyan]Ollama[/cyan] (Recommended):")
        console.print("   - Download: https://ollama.ai")
        console.print("   - Install model: ollama pull llama2")
        console.print("   - Start server: ollama serve")
        console.print("\n2. [cyan]Hugging Face Transformers[/cyan]:")
        console.print("   - Install: pip install transformers torch")
        console.print("   - Models download automatically")
        console.print("\n3. [cyan]LocalAI or compatible API[/cyan]:")
        console.print("   - Set up at http://localhost:8080")
        return

    # Create table of available models
    table = Table(title="AI Model Status")
    table.add_column("Model", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Status", justify="center")
    table.add_column("Description", style="dim")

    model_info = {
        "ollama_llama2": ("Ollama", "General purpose LLM"),
        "ollama_codellama": ("Ollama", "Code-focused LLM"),
        "ollama_mistral": ("Ollama", "Fast and efficient LLM"),
        "hf_dialogpt": ("Hugging Face", "Conversational AI"),
        "hf_gpt2": ("Hugging Face", "Text generation"),
        "localai": ("LocalAI", "OpenAI-compatible API"),
        "textgen_webui": ("Text Gen WebUI", "Local text generation")
    }

    for model_name, is_available in available_models.items():
        model_type, description = model_info.get(model_name, ("Unknown", ""))
        status = "[green]✓ Available[/green]" if is_available else "[red]✗ Not Available[/red]"
        table.add_row(model_name, model_type, status, description)

    console.print(table)


@app.command()
def chat(
    cv_file: str = typer.Argument(..., help="CV file to analyze"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="AI model to use"),
    criteria: str = typer.Option("default", "--criteria", "-c", help="Evaluation criteria")
):
    """Interactive chat about a CV using AI models."""

    try:
        from .ai.free_models import set_ai_model, auto_select_ai_model, get_ai_response

        # Set up AI model
        if model:
            if not set_ai_model(model):
                console.print(f"[red]Failed to set model: {model}[/red]")
                raise typer.Exit(1)
        else:
            selected = auto_select_ai_model()
            if not selected:
                console.print("[red]No AI models available. Use 'list-ai-models' to see setup instructions.[/red]")
                raise typer.Exit(1)
            console.print(f"[green]Using AI model: {selected}[/green]")

        # Load and process CV
        console.print(f"[blue]Loading CV:[/blue] {cv_file}")

        evaluation_criteria = criteria_manager.get_criteria(criteria)
        evaluator = CVEvaluator(evaluation_criteria)

        # Process CV
        with console.status("Processing CV..."):
            result = evaluator.evaluate_cv(cv_file)

        if not result:
            console.print("[red]Failed to process CV[/red]")
            raise typer.Exit(1)

        # Display CV summary
        console.print(Panel.fit(
            f"[bold]Candidate:[/bold] {result.cv_data.personal_info.name or 'Unknown'}\n"
            f"[bold]Overall Score:[/bold] {result.overall_score:.1f}/100\n"
            f"[bold]Job Fit:[/bold] {result.fit_percentage:.1f}%",
            title="CV Summary"
        ))

        # Start interactive chat
        console.print("\n[bold green]🤖 AI Chat Assistant Ready![/bold green]")
        console.print("Ask questions about this CV. Type 'quit' to exit.\n")

        # Prepare CV context
        cv_context = _prepare_cv_context_for_chat(result.cv_data)

        while True:
            try:
                # Get user question
                question = typer.prompt("\nYour question")

                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                # Create AI prompt
                prompt = f"""
You are a CV evaluation expert. Here's information about a candidate:

{cv_context}

User question: {question}

Please provide a helpful, professional response about this candidate based on the CV information.
"""

                # Get AI response
                with console.status("Thinking..."):
                    response = get_ai_response(prompt, max_tokens=600)

                # Display response
                console.print(f"\n[bold blue]Assistant:[/bold blue] {response}")

            except KeyboardInterrupt:
                console.print("\n[yellow]Chat interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    except Exception as e:
        console.print(f"[red]Chat failed:[/red] {e}")
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


def _display_participant_results(participant_id: str, result, evaluator):
    """Display participant evaluation results."""

    participant = evaluator.participants[participant_id]

    # Participant info panel
    info_panel = Panel.fit(
        f"[bold]Participant ID:[/bold] {participant_id}\n"
        f"[bold]Name:[/bold] {participant.name or 'Unknown'}\n"
        f"[bold]Files Processed:[/bold] {len(participant.files)}\n"
        f"[bold]Overall Score:[/bold] {result.overall_score:.1f}/100\n"
        f"[bold]Job Fit:[/bold] {result.fit_percentage:.1f}%",
        title="Participant Evaluation"
    )
    console.print(info_panel)

    # Files table
    files_table = Table(title="Processed Files")
    files_table.add_column("File", style="cyan")
    files_table.add_column("Type", style="yellow")
    files_table.add_column("Status", justify="center")
    files_table.add_column("Description", style="dim")

    for file_obj in participant.files:
        status_color = "green" if file_obj.processing_status == "completed" else "red"
        status = f"[{status_color}]{file_obj.processing_status}[/{status_color}]"

        files_table.add_row(
            file_obj.file_path.name,
            file_obj.file_type,
            status,
            file_obj.description or "N/A"
        )

    console.print(files_table)

    # Section scores
    _display_evaluation_result(result)


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


def _prepare_cv_context_for_chat(cv_data) -> str:
    """Prepare CV context for chat interface."""
    context_parts = []

    # Personal info
    if cv_data.personal_info.name:
        context_parts.append(f"Candidate: {cv_data.personal_info.name}")

    # Skills
    if cv_data.skills:
        skills = ", ".join([skill.name for skill in cv_data.skills[:10]])
        context_parts.append(f"Skills: {skills}")

    # Experience
    if cv_data.work_experience:
        exp_summary = []
        for exp in cv_data.work_experience[:3]:
            exp_summary.append(f"{exp.position} at {exp.company}")
        context_parts.append(f"Experience: {'; '.join(exp_summary)}")

    # Education
    if cv_data.education:
        edu_summary = []
        for edu in cv_data.education[:2]:
            edu_summary.append(f"{edu.degree} from {edu.institution}")
        context_parts.append(f"Education: {'; '.join(edu_summary)}")

    return "\n".join(context_parts)


if __name__ == "__main__":
    app()
