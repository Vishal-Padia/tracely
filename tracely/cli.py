import os
import json
import click
import webbrowser
import subprocess
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
@click.option("--project", "project", required=False, type=str, help="Project name")
def ui(web, project):
    """Launch the Streamlit UI."""
    try:
        import streamlit
        import subprocess
        import time
        from tracely import webapp

        # set the project as an env var
        env = os.environ.copy()
        if project:
            env["TRACELY_PROJECT"] = project
            click.echo(f"Starting UI for project: {project}")
        else:
            click.echo("No project selected. Please select a project in the cli.")
            click.echo("You can view all your projects using `tracely list`")
            return

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


@cli.command(name="sync")
@click.option(
    "--from-ip",
    "from_ip",
    required=True,
    type=str,
    help="IP address where you're running the script",
)
@click.option(
    "--key",
    "key_path",
    required=True,
    type=str,
    help="Location to your private ssh key",
)
@click.option(
    "--user", "user", required=True, type=str, help="Username for the remote server"
)
@click.option(
    "--port",
    "port",
    required=False,
    type=int,
    help="Port to sync the run from",
    default=8501,
)
@click.option(
    "--remote-port",
    "remote_port",
    required=False,
    type=int,
    help="Port to sync the run from",
    default=8501,
)
@click.option("--project", "project", required=False, type=str, help="Project name")
def sync(
    from_ip: str,
    key_path: str,
    user: str,
    port: int = 8501,
    remote_port: int = 8501,
    project: str = None,
):
    """
    Sync your runs from a remote server to your local machine

    Args:
        project: The name of the project you're currently running the script.
        from_ip: The IP address you're currently running the script.
        key_path: Location to your private ssh key
        user: Username for the remote server
        port: Port to sync the run from
        remote_port: Port to sync the run to
    """
    if not from_ip:
        click.echo(
            "Error: Please provide the IP address of your cloud machine.", err=True
        )
        return
    expanded_key_path = os.path.expanduser(key_path)
    if not key_path or not os.path.exists(expanded_key_path):
        click.echo(
            "Error: Please provide a valid path to your private SSH key.", err=True
        )
        return
    if not project:
        click.echo("Error: Please provide a project name.", err=True)
        return

    # start streamlit on remote machine
    click.echo("Starting the interface on the remote machine")
    start_remote_ui_command = [
        "ssh",
        "-i",
        expanded_key_path,
        f"{user}@{from_ip}",
        "nohup tracely ui --project {project} > /dev/null 2>&1 & echo 'Interface started on remote machine'",
    ]
    try:
        subprocess.run(start_remote_ui_command, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error starting the interface on the remote machine: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        return

    # creating ssh tunnel
    ssh_command = [
        "ssh",
        "-i",
        expanded_key_path,
        f"{user}@{from_ip}",
        "-N",
        "-L",
        f"{port}:localhost:{remote_port}",
    ]
    try:
        # start tunnel process in the background
        tunnel_process = subprocess.Popen(
            ssh_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # giving time for the tunnel to be created
        import time

        time.sleep(3)

        # check if the tunnel process is still running
        if tunnel_process.poll() is not None:
            _, stderr = tunnel_process.communicate()
            click.echo(
                f"Error: SSH tunnel failed. Please check your SSH key and IP address. {stderr}",
                err=True,
            )
            return

        # open the browser locally
        browser_url = f"http://localhost:{port}"
        click.echo(f"Opening the browser at {browser_url}")
        webbrowser.open(browser_url)

        # keep the tunnel process running until the user interupts
        try:
            tunnel_process.wait()
        except KeyboardInterrupt:
            click.echo("Closing the sync process")
            tunnel_process.terminate()
            try:
                tunnel_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel_process.kill()
            click.echo("Sync process closed")
    except Exception as e:
        click.echo(
            f"Error: An Error occurred while syncing the run. Please check your SSH key and IP address {e}",
            err=True,
        )


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
