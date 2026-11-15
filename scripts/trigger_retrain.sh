#!/usr/bin/env bash
set -euo pipefail

echo "=== Triggering Prefect Retraining Pipeline ==="
if docker compose ps recsys-api >/dev/null 2>&1; then
  docker compose exec -T recsys-api python src/flow_retrain.py
else
  python src/flow_retrain.py
fi

echo "=== Notifying API /admin/reload ==="
curl -s -X POST "${API_BASE_URL:-http://localhost:8000}/admin/reload" || true
echo ""
echo "=== Done! ==="
