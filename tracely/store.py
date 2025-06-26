import os
import json
import uuid
import shutil
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# Base directory for all runs
TRACELY_DIR = os.path.expanduser("~/.tracely")


def _find_run_dir(run_id: str) -> Path:
    """
    Helper to find a run's directory given its ID.

    Args:
        run_id (str): The ID of the run to find the directory for.
    """
    for project_dir in (Path(TRACELY_DIR) / "runs").iterdir():
        if not project_dir.is_dir():
            continue
        candidate = project_dir / run_id
        if candidate.exists():
            return candidate
    raise ValueError(f"Run {run_id} not found")


def init_run(project: str, run_name: Optional[str], config: Optional[Dict]) -> str:
    """
    Initialize a new run by creating required directories and metadata files.

    Args:
        project (str): Name of the project this run belongs to
        run_name (Optional[str]): Optional name for this run
        config (Optional[Dict]): Optional configuration dictionary to save

    Returns:
        run_id (str): Unique identifier for this run
    """
    # Ensure base directory exists
    os.makedirs(TRACELY_DIR, exist_ok=True)

    # Generate unique run ID and create directory structure
    run_id = str(uuid.uuid4())
    run_dir = Path(TRACELY_DIR) / "runs" / project / run_id

    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir / "artifacts", exist_ok=True)

    # Initialize run metadata
    meta = {
        "run_id": run_id,
        "project": project,
        "name": run_name,
        "start_time": datetime.now().isoformat(),
        "status": "running",
        "config": config or {},
    }

    # Write initial metadata
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save system info
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    with open(run_dir / "system.json", "w") as f:
        json.dump(system_info, f, indent=2)

    return run_id


def finalize_run(
    run_id: str,
    success: bool,
    error: Optional[str] = None,
    traceback_str: Optional[str] = None,
):
    """
    Mark a run as complete and save final status.

    Args:
        run_id (str): ID of the run to finalize
        success (bool): Whether the run completed successfully
        error (Optional[str]): Optional error message if run failed
        traceback_str (Optional[str]): Optional traceback string if run failed
    """
    run_dir = _find_run_dir(run_id)

    # Update metadata
    meta_path = run_dir / "run_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    meta.update(
        {
            "end_time": datetime.now().isoformat(),
            "status": "completed" if success else "failed",
            "error": error,
            "traceback": traceback_str,
        }
    )

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def log_metric(
    run_id: str, key: str, value: float, step: Optional[int], timestamp: int
):
    """
    Log a metric value for the current run.

    Args:
        run_id (str): ID of the run
        key (str): Name of the metric
        value (float): Value to log
        step (Optional[int]): Optional step number
        timestamp (int): Unix timestamp in milliseconds
    """
    run_dir = _find_run_dir(run_id)

    metric_data = {"key": key, "value": value, "step": step, "timestamp": timestamp}

    if _HAS_PANDAS:
        # Use parquet if pandas is available
        metrics_file = run_dir / "metrics.parquet"
        if metrics_file.exists():
            df = pd.read_parquet(metrics_file)
            df = pd.concat([df, pd.DataFrame([metric_data])], ignore_index=True)
        else:
            df = pd.DataFrame([metric_data])
        df.to_parquet(metrics_file, index=False)
    else:
        # Fallback to JSON
        print(
            "[Tracely] pandas not installed; falling back to JSON. `pip install pandas pyarrow` for better performance."
        )
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
        else:
            metrics = []
        metrics.append(metric_data)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)


def log_artifact(run_id: str, name: str, path: str):
    """
    Save an artifact file or directory for the run.

    Args:
        run_id (str): ID of the run
        name (str): Name to give the artifact
        path (str): Path to the file/directory to save
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact path {path} does not exist.")

    run_dir = _find_run_dir(run_id)
    artifacts_dir = run_dir / "artifacts"
    dest = artifacts_dir / name

    # Copy file or directory
    if os.path.isdir(path):
        shutil.copytree(path, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(path, dest)


def delete_project(project: str, force: bool = False) -> bool:
    """
    Delete a project and all its associated runs.

    Args:
        project (str): Name of the project to delete
        force (bool): If True, delete without additional checks

    Returns:
        bool: True if deletion was successful, False otherwise

    Raises:
        ValueError: If project doesn't exist
    """
    # Check project exists
    project_dir = Path(TRACELY_DIR) / "projects" / project
    runs_dir = Path(TRACELY_DIR) / "runs" / project
    
    if not project_dir.exists() and not runs_dir.exists():
        raise ValueError(f"Project '{project}' does not exist")

    if not force:
        # Count number of runs
        run_count = sum(1 for _ in runs_dir.glob("**/run_meta.json"))
        if run_count > 0:
            raise ValueError(
                f"Project '{project}' has {run_count} runs. "
                "Use force=True to delete anyway"
            )

    # Delete project directory and all runs
    if project_dir.exists():
        shutil.rmtree(project_dir)
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    
    return True
