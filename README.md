# Tracely
Tracely: Effortless ML experiment tracking, no setup, no clutter, just your results.

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

## Roadmap

### Core UX (Already Implemented)
* **`tracely init`**
  Initialize a new experiment project directory with default config scaffolding.
* **`tracely list`**
  View all tracked runs across projects with status, metadata, and names.
* **`tracely delete`**
  Remove specific runs or entire projects cleanly from your `.tracely` store.
* **`tracely ui --web`**
  Launch Streamlit dashboard to explore configs, metrics, artifacts, and debug info.
* **`tracely sync` (Remote Viewer)**
  Secure SSH-based tunneling from remote machines to your local browser for real-time experiment viewing.

### Core UX (Coming Soon)

* **`tracely track` (SDK Decorator)**
  Decorator to auto-track run metadata, exceptions, and timestamps around your training functions.

* **Run Diffs**
  Visualize what changed between runs — configs, metrics, artifacts — in a structured diff panel.

* **Code Snapshot Panel**
  Automatically log and preview source code used for a run in the UI sidebar for reproducibility.

* **PyPI Publishing**
  One-liner `pip install tracely` with fully documented setup and CLI — open for community.

### Plugin System (Modular, Opt-in)

* **LLM & RAG Evaluation Plugins**
  Eval metrics like BLEU, faithfulness, factuality for NLP/RAG pipelines out-of-the-box.

* **Triton Profiler Plugin**
  Track and visualize GPU memory ops, kernel execution time, and occupancy for Triton kernels.

* **Drift & Distribution Trackers**
  Monitor and alert on feature drift, label imbalance, or data distribution shifts over time.

* **Custom Metrics & Alerts**
  Easily define domain-specific metrics or performance alerts that integrate into the Tracely run lifecycle.

### Advanced Analytics

* **Run Comparison & Diff Viewer**
  Select multiple runs and view deltas in config, metrics, and performance with visual diffs.

* **Custom Metric Visualization**
  Build metric dashboards from JSON, YAML, or CLI templates — customize your view per project.

* **Performance Profiling (Beta)**
  Trace runtime stats, memory usage, and step-level timing per run. Torch/Triton support first.

### Integrations

* **Export to W&B or MLflow**
  Send logs to W&B or MLflow in addition to Tracely — be compatible, not siloed.

* **Git Commit Hooking**
  Log git commit hash + diff automatically to track model-code coupling.

* **Cloud Storage Support**
  Store artifacts/metrics in S3, GCS, or Azure Blob for remote syncing and team workflows.

### Enhanced UI

* **Self-Hosted React + API UI**
  Replace Streamlit with a production-grade frontend backed by REST API for scalability.

* **Custom Dashboards**
  Build per-project dashboards that auto-refresh and organize key insights visually.

* **Real-time Monitoring**
  Live socket-based updates from training scripts to UI — graphs update without reload.

---

Built by one wannabe ML engineer fed up with bloated tools.
