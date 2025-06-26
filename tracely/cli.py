import os
import json
import click
import webbrowser
from pathlib import Path
from typing import Optional

from . import store


@click.group()
def cli():
    """Command line interface for Tracely experiment tracking."""
    pass


@cli.command()
@click.argument("project")
def init(project: str):
    """Initialize a new project."""
    if project_exists(project=project):
        click.echo(
            f"Project '{project}' already exists at {Path(store.TRACELY_DIR) / 'runs' / project}"
        )
        click.echo(
            f"You can continue using this project or delete it with 'tracely delete {project}'"
        )
        return

    project_dir = Path(store.TRACELY_DIR) / "runs" / project
    os.makedirs(project_dir, exist_ok=True)

    # Create empty config file
    config_path = project_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            f.write("{}\n")

    click.echo(f"Initialized project '{project}' at {project_dir}")


@cli.command(name="list")
@click.option("--project", default=None, help="Filter by project")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_runs(project: Optional[str] = None, as_json: bool = False):
    """List recent runs."""
    runs_dir = Path(store.TRACELY_DIR) / "runs"
    if not runs_dir.exists():
        click.echo("No runs found")
        return
    output = []

    for project_dir in runs_dir.iterdir():
        if not project_dir.is_dir():
            continue
        if project and project_dir.name != project:
            continue

        for run_dir in project_dir.iterdir():
            if not run_dir.is_dir():
                continue

            try:
                with open(run_dir / "run_meta.json") as f:
                    meta = json.load(f)
                run_info = {
                    "project": project_dir.name,
                    "run_id": run_dir.name,
                    "name": meta.get("name", "unnamed"),
                    "status": meta.get("status", "unknown"),
                    "start_time": meta.get("start_time", "unknown"),
                }
                output.append(run_info)
            except Exception:
                output.append(
                    {
                        "project": project_dir.name,
                        "run_id": run_dir.name,
                        "name": None,
                        "status": "error",
                        "start_time": None,
                    }
                )

    if as_json:
        click.echo(json.dumps(output, indent=2))
    else:
        grouped = {}
        for run in output:
            grouped.setdefault(run["project"], []).append(run)

        for project, runs in grouped.items():
            click.echo(f"\nProject: {project}")
            for run in runs:
                click.echo(
                    f"  {run['run_id'][:8]}... | {run['name']} | {run['status']}"
                )


@cli.command()
@click.option("--web", is_flag=True, help="Open in browser after launch")
def ui(web):
    """Launch the Streamlit UI."""
    try:
        import streamlit
        import subprocess
        import time
        from tracely import webapp

        # launch the streamlit app using subprocess
        process = subprocess.Popen(
            ["streamlit", "run", webapp.__file__],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        time.sleep(2)
        if web:
            webbrowser.open("http://localhost:8501")

        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            click.echo("\nShutting down Streamlit server...")

    except ImportError:
        click.echo("Streamlit not found. Install it with: pip install streamlit")
    except Exception as e:
        click.echo(f"Error launching UI: {str(e)}", err=True)
        process.terminate()
        exit(1)


@cli.command()
@click.argument("run_id")
def path(run_id: str):
    """Print full path to a run's directory."""
    try:
        run_dir = store._find_run_dir(run_id)
        click.echo(str(run_dir))
    except ValueError as e:
        click.echo(str(e), err=True)


@cli.command()
@click.argument("project")
@click.option("--force", is_flag=True, help="Force deletion without confirmation")
def delete(project: str, force: bool = False):
    """
    Delete a project and all its runs.

    Args:
        project (str): The name of the project to delete.
        force (bool): Whether to force deletion without confirmation.
    """
    try:
        if not force:
            runs_dir = Path(store.TRACELY_DIR) / "runs" / project
            runs_count = (
                sum(1 for _ in runs_dir.glob("**/run_meta.json"))
                if runs_dir.exists()
                else 0
            )

            if runs_count > 0:
                click.confirm(
                    f"Project '{project}' has {runs_count} runs that will be deleted. "
                    "Are you sure you want to continue?",
                    abort=True,
                )
            else:
                click.confirm(
                    f"Are you sure you want to delete project '{project}'? This action cannot be undone.",
                    abort=True,
                )

            # perform deletion
            store.delete_project(project)
            click.echo(f"Project '{project}' and all its runs have been deleted.")
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        exit(1)

    except click.Abort:
        click.echo("Operation cancelled.")
        exit(1)
    except Exception as e:
        click.echo(f"Error deleting project: {str(e)}", err=True)
        exit(1)


def project_exists(project: str) -> bool:
    """
    Check if a project already exists.

    Args:
        project (str): Name of the project to check

    Returns:
        bool: True if project exists, False otherwise
    """
    project_dir = Path(store.TRACELY_DIR) / "runs" / project
    return project_dir.exists()


def main():
    cli()


if __name__ == "__main__":
    main()
