import os
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional

TRACELY_DIR = os.path.expanduser("~/.tracely")


def main():
    st.set_page_config(page_title="Tracely", layout="wide")
    st.title("Tracely - lightweight ML monitoring tool")

    # get the project from the env var
    project = os.environ.get("TRACELY_PROJECT")

    if project:
        # verify whether the project exists
        project_dir = Path(TRACELY_DIR) / "runs" / project
        if not project_dir.exists():
            st.error(f"Project {project} does not exist.")
            st.stop()

        # get the run id
        run_id = sidebar_run_selector(project)
        if run_id:
            run_dir = Path(TRACELY_DIR) / "runs" / project / run_id
            show_run_details(run_dir)
    else:
        st.error(
            "No project selected when running the tracely cli. Please select a project in the cli."
        )
        st.stop()


def sidebar_run_selector(project: str) -> Optional[str]:
    project_dir = Path(TRACELY_DIR) / "runs" / project
    runs = sorted([r.name for r in project_dir.iterdir() if r.is_dir()])
    return st.sidebar.selectbox("Select Run", runs) if runs else None


def show_run_details(run_dir: Path):
    st.subheader(f"🔍 Run: {run_dir.name}")
    meta = load_run_meta(run_dir)

    tab1, tab2, tab3, tab4 = st.tabs(["Config", "Metrics", "Artifacts", "Debug"])

    with tab1:
        st.markdown("### Config & Metadata")
        st.json(meta.get("config", {}))
        st.write(f"**Status**: `{meta.get('status', 'unknown')}`")
        st.write(f"**Start Time**: `{meta.get('start_time', '-')}`")
        st.write(f"**End Time**: `{meta.get('end_time', '-')}`")

    with tab2:
        st.markdown("### Metrics")
        metrics_df = load_metrics(run_dir)
        if not metrics_df.empty:
            metric_keys = metrics_df["key"].unique().tolist()
            selected_keys = st.multiselect(
                "Select metrics", metric_keys, default=metric_keys[:1]
            )
            for key in selected_keys:
                plot_df = metrics_df[metrics_df["key"] == key]
                st.plotly_chart(plot_df.set_index("step")["value"])
        else:
            st.info("No metrics logged.")

    with tab3:
        st.markdown("### Artifacts")
        artifacts_dir = run_dir / "artifacts"
        if not artifacts_dir.exists():
            st.warning("No artifacts found.")
        else:
            for file in artifacts_dir.rglob("*"):
                if file.is_file():
                    st.write(f"📁 `{file.name}`")
                    if file.suffix in [".png", ".jpg", ".jpeg"]:
                        st.image(str(file))
                    elif file.suffix in [".txt", ".log", ".json"]:
                        st.text(file.read_text())
                    else:
                        st.download_button(
                            label=f"Download {file.name}",
                            data=open(file, "rb"),
                            file_name=file.name,
                        )

    with tab4:
        st.markdown("### Debug Info")
        if meta.get("status") == "failed":
            st.error(meta.get("error", "No error message"))
            st.code(meta.get("traceback", "No traceback"))
        else:
            st.info("Run completed successfully.")


def load_run_meta(run_dir: Path) -> dict:
    try:
        with open(run_dir / "run_meta.json") as f:
            return json.load(f)
    except Exception:
        return {}


def load_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path_parquet = run_dir / "metrics.parquet"
    metrics_path_json = run_dir / "metrics.json"
    try:
        if metrics_path_parquet.exists():
            return pd.read_parquet(metrics_path_parquet)
        elif metrics_path_json.exists():
            with open(metrics_path_json) as f:
                return pd.DataFrame(json.load(f))
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


if __name__ == "__main__":
    main()
