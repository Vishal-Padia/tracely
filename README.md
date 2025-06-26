# Tracely
Tracely — A zero-config, local-first ML observability tool built for hackers, not dashboards.

---

# What is tracely?
**Tracely** is a lightweight, plug-and-play experiment tracker and metric logger for ML developers who want visibility without the bloat.

## Quickstart

```bash
# Install
pip install -e .

# Initialize a project
tracely init mnist

# Run your training script
python examples/mnist_training.py

# List your runs
tracely list

# View in the UI
tracely ui --web
```

```python
from tracely import store

# Initialize a run
run_id = store.init_run(
    project="mnist",
    run_name="conv_net_basic",
    config={"batch_size": 64, "lr": 0.001}
)

# Log metrics
store.log_metric(
    run_id=run_id,
    key="accuracy",
    value=0.98,
    step=1,
    timestamp=int(time.time() * 1000)
)

# Save artifacts
store.log_artifact(
    run_id=run_id,
    name="model.pt",
    path="./checkpoints/model.pt"
)

# Finalize the run
store.finalize_run(run_id=run_id, success=True)
```

---

## What Gets Logged

* **Metrics** → Stored in `metrics.parquet` (or `metrics.json` if pandas not available)
* **Artifacts** → Saved in `artifacts/` directory
* **Configs** → Run config in `run_meta.json`, system info in `system.json`
* **Full run history** → All data in `~/.tracely/runs/<project>/<run_id>/`

---

## Features

### Implemented
* Project Management
  * Initialize projects (`tracely init <project>`)
  * Delete projects (`tracely delete <project>`)
  * List runs (`tracely list`)
* Run Tracking
  * Metric logging with timestamps and steps
  * Artifact saving and management
  * System info capture
  * Run status tracking (running/completed/failed)
* UI
  * Streamlit-based dashboard
  * Multi-run visualization
  * Metric plotting
  * Artifact viewing
* Error Handling
  * Graceful failure capture
  * Detailed error logging
  * Debug information in UI

### Coming Soon
* **Plugin System**
  * LLM & RAG evaluations
  * Triton profiling
  * Drift & distribution tracking
  * Custom metrics / alerts
* **Advanced Analytics**
  * Run comparison & diff viewer
  * Custom metric visualizations
  * Performance profiling
* **Integrations**
  * Export to W&B or MLflow
  * Git integration
  * Cloud storage support
* **Enhanced UI**
  * Self-hosted React+API interface
  * Custom dashboards
  * Real-time monitoring

---

## Why Tracely?

Most tools today either:
* Lock you into cloud dashboards,
* Require heavy config,
* Or force you to conform to their workflow.

**Tracely is different.** It's:
* **Local-first**: works offline, no accounts
* **Fast**: minimal setup, instant insights
* **Hackable**: bring your own eval, pipeline, or model stack

---

Built by one wannabe ML engineer fed up with bloated tools.