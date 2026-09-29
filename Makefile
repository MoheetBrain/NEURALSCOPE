.PHONY: setup dev demo api web test test-fast test-slow test-web lint build clean-data acceptance experiment-e1

setup:        ## install Python + web deps and download model weights
	./scripts/setup.sh

dev:          ## run API (real model) + web on http://localhost:3000
	./scripts/dev.sh

demo:         ## run with the random-weight DEMO fixture model (UI development only)
	./scripts/dev.sh --demo

api:
	./scripts/api.sh

web:
	./scripts/web.sh

test: test-fast test-web   ## fast Python tests + web tests

test-fast:    ## Python tests on the fixture model + Math Lab oracle (seconds)
	.venv/bin/python -m pytest -m "not slow" -q

test-slow:    ## integration tests on a real pretrained model (Qwen2.5-0.5B-Instruct)
	.venv/bin/python -m pytest -m slow -q

test-web:
	cd apps/web && npm run test

lint:
	cd apps/web && npm run lint && npm run typecheck

build:
	cd apps/web && npm run build

clean-data:   ## delete local runs, experiments and activation files
	rm -rf data/neuralscope.db data/demo data/activations/*.safetensors data/experiments/*.json

acceptance:   ## V1 acceptance flow over HTTP against a running API (real model)
	.venv/bin/python scripts/acceptance.py

experiment-e1:  ## Experiment E1 (carry heads) in-process; stop the API first on 16 GB machines
	.venv/bin/python scripts/experiments/carry_heads.py --n 40
