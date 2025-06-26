import time
import traceback
from functools import wraps
from typing import Optional, Callable, Dict

from . import store

# Internal global to track current run state
_current_run = {"id": None}


def track(
    project: str, run_name: Optional[str] = None, config: Optional[Dict] = None
) -> Callable:
    """
    Decorator to track an ML run.
    Automatically initializes run, sets up files, and finalizes with status.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            run_id = store.init_run(project=project, run_name=run_name, config=config)
            _current_run["id"] = run_id

            try:
                result = func(*args, **kwargs)
                store.finalize_run(run_id, success=True)
                return result
            except Exception as e:
                store.finalize_run(
                    run_id,
                    success=False,
                    error=str(e),
                    traceback_str=traceback.format_exc(),
                )
                raise
            finally:
                _current_run["id"] = None  # Clean up after run

        return wrapper

    return decorator


def log_metric(key: str, value: float, step: Optional[int] = None):
    """
    Log a metric value for the current run.

    Args:
        key (str): The name of the metric.
        value (float): The value of the metric.
        step (Optional[int]): The step number for the metric.
    """
    run_id = _current_run["id"]
    if run_id is None:
        raise ValueError(
            "No run is currently active. Please ensure you are using the @track decorator."
        )
    timestamp = int(time.time() * 1000)
    store.log_metric(run_id, key, value, step=step, timestamp=timestamp)


def log_artifact(name: str, path: str):
    """
    Log an artifact for the current run.

    Args:
        name (str): The name of the artifact.
        path (str): The path to the artifact.
    """
    run_id = _current_run["id"]
    if run_id is None:
        raise RuntimeError("No active run. Make sure you're using @track.")
    store.log_artifact(run_id, name, path)
