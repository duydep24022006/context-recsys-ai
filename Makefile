.PHONY: data benchmark up down test lint smoke retrain drift load-test

PYTHON ?= python

data:
	$(PYTHON) src/prepare_data.py --source synthetic --eda

benchmark:
	docker compose up -d mlflow-server
	docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 recsys-api python src/benchmark.py

up:
	docker compose up -d --build

down:
	docker compose down

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:
	$(PYTHON) scripts/smoke_test.py

retrain:
	docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 -e API_BASE_URL=http://recsys-api:8000 recsys-api python src/flow_retrain.py

drift:
	$(PYTHON) src/drift.py

load-test:
	$(PYTHON) scripts/load_test.py
