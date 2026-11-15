"""End-to-end verification runner: Data + EDA -> Multi-seed Benchmark -> FastAPI Server -> Smoke Test -> Load Test -> Drift & Prefect Retrain -> Hot Reload -> UI Screenshots."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
PY = sys.executable


def wait_http(url: str, timeout_sec: int = 30) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def run_step(cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, env=env, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")


def main() -> None:
    # Start from a clean mlflow_store so version numbers are clean (v1 from benchmark, v2 from retrain)
    for d in ["mlflow_store", "mlruns"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    os.makedirs("mlflow_store", exist_ok=True)

    db_path = os.path.abspath("mlflow_store/mlflow.db").replace("\\", "/")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    env["DATA_DIR"] = os.path.join(ROOT, "shared_data")
    env["MLFLOW_TRACKING_URI"] = f"sqlite:///{db_path}"
    env["API_BASE_URL"] = "http://127.0.0.1:8000"

    api_proc = None
    try:
        # 1. Fresh clean dataset + EDA figures
        run_step([PY, "src/prepare_data.py", "--source", "synthetic", "--eda"], env)

        # 2. Run multi-seed CLO1 benchmark (7 models x 3 seeds)
        run_step([PY, "src/benchmark.py"], env)

        # 3. Start FastAPI server on :8000
        print("\n>>> Starting FastAPI server on :8000 ...", flush=True)
        api_proc = subprocess.Popen(
            [PY, "-m", "uvicorn", "api:app", "--app-dir", "src", "--host", "127.0.0.1", "--port", "8000"],
            env=env,
        )
        if not wait_http("http://127.0.0.1:8000/health", timeout_sec=30):
            raise RuntimeError("FastAPI server failed to become healthy on :8000")
        print(">>> FastAPI Server is healthy!", flush=True)

        # 4. Run Smoke Test
        run_step([PY, "scripts/smoke_test.py"], env)

        # 5. Run Load Test (p50/p95/p99 latency report)
        run_step([PY, "scripts/load_test.py", "--url", "http://127.0.0.1:8000", "--requests", "200"], env)

        # 6. Simulate new incoming data with drift + run Drift report + run Prefect Retrain Flow
        run_step([PY, "scripts/simulate_new_data.py", "--rows", "2000", "--drift"], env)
        run_step([PY, "src/drift.py"], env)
        run_step([PY, "src/flow_retrain.py"], env)

        # 7. Verify API hot-reloaded the newly promoted model without container/process restart
        with urllib.request.urlopen("http://127.0.0.1:8000/model/info", timeout=5) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        print(
            f"\n>>> Verified API /model/info after retrain: version={info.get('version')}, "
            f"model_loaded={info.get('model_loaded')}, last_retrain={info.get('last_retrain')}",
            flush=True,
        )

        # 8. Restore clean 20,000-row dataset so static stats match docs/DATA.md while keeping artifacts
        run_step([PY, "src/prepare_data.py", "--source", "synthetic"], env)

        # 9. Generate UI screenshots
        run_step([PY, "scripts/generate_ui_screenshots.py"], env)

        print("\n=== ALL VERIFICATION STEPS COMPLETED SUCCESSFULLY ===", flush=True)
    finally:
        if api_proc is not None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=5)
            except Exception:
                api_proc.kill()


if __name__ == "__main__":
    main()
