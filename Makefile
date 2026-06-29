# Thin convenience wrappers around the `rtb` CLI (src/rtb_rl/cli.py).
# On Windows without `make`, call the CLI directly, e.g.  `python -m rtb_rl.cli demo`.

.PHONY: install dev demo data features train sim serve retrain test lint type fmt compose-up compose-down clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Full offline pipeline: synthetic data -> features+affinity -> DQN -> sim eval. No services.
demo:
	python -m rtb_rl.cli demo

data:
	python -m rtb_rl.cli generate-data

features:
	python -m rtb_rl.cli build-features

train:
	python -m rtb_rl.cli train

sim:
	python -m rtb_rl.cli sim

serve:
	python -m rtb_rl.cli serve

retrain:
	python -m rtb_rl.cli retrain --once

test:
	pytest -q

lint:
	ruff check src tests scripts

type:
	mypy src

fmt:
	ruff format src tests scripts
	ruff check --fix src tests scripts

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

clean:
	rm -rf data/raw/* data/processed/* checkpoints/* results/* registry/* 2>/dev/null || true
