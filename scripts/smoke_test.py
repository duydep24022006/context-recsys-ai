"""End-to-end smoke test for Context-Aware Recommender API."""
from __future__ import annotations

import os
import sys
import urllib.request
import json

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _get(path: str):
    with urllib.request.urlopen(f"{API_URL}{path}", timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print(f"[smoke] Testing API at {API_URL} ...")
    st, health = _get("/health")
    assert st == 200 and health["status"] == "ok", f"Health failed: {health}"
    print(f"[smoke] /health OK: {health}")

    st1, rec_mob = _post("/recommend", {"user_id": 15, "device": "mobile", "hour": 20, "top_k": 10})
    assert st1 == 200 and len(rec_mob["items"]) == 10, "Mobile recommend failed"

    st2, rec_dsk = _post("/recommend", {"user_id": 15, "device": "desktop", "hour": 10, "top_k": 10})
    assert st2 == 200 and len(rec_dsk["items"]) == 10, "Desktop recommend failed"

    mob_pids = [it["product_id"] for it in rec_mob["items"]]
    dsk_pids = [it["product_id"] for it in rec_dsk["items"]]
    print(f"[smoke] Mobile 20h top-5 PIDs : {mob_pids[:5]} ({rec_mob['latency_ms']} ms)")
    print(f"[smoke] Desktop 10h top-5 PIDs: {dsk_pids[:5]} ({rec_dsk['latency_ms']} ms)")
    assert mob_pids != dsk_pids, "Expected context-sensitive recommendations to differ between Mobile 20h and Desktop 10h"

    # Cold-start check
    st3, rec_cold = _post("/recommend", {"user_id": 99999, "device": "mobile", "hour": 21, "top_k": 5})
    assert st3 == 200 and len(rec_cold["items"]) == 5, "Cold-start recommend failed"
    print(f"[smoke] Cold-start user 99999 OK (cold_start={rec_cold.get('cold_start')})")

    with urllib.request.urlopen(f"{API_URL}/metrics", timeout=10) as resp:
        metrics_body = resp.read().decode("utf-8")
    assert "recsys_requests_total" in metrics_body, "Prometheus metrics missing"
    print("[smoke] ALL SMOKE CHECKS PASSED!")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[smoke] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
