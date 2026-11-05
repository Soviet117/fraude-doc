SHELL := /bin/bash
COMPOSE := docker compose
CONFIG ?= configs/base.yaml
SAMPLES ?= data/muestras

.DEFAULT_GOAL := help

.PHONY: help setup up down shell gpu demo test lint fmt lock synth audit-samples ocr-samples build clean

help: ## Muestra los comandos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Crea .env con el UID/GID del usuario actual
	@printf "UID=%s\nGID=%s\n" "$$(id -u)" "$$(id -g)" > .env
	@echo "Creado .env con UID=$$(id -u) GID=$$(id -g)"

up: ## Levanta el contenedor de desarrollo
	$(COMPOSE) up -d dev

down: ## Detiene y elimina los contenedores
	$(COMPOSE) down

shell: ## Abre una shell dentro del contenedor
	$(COMPOSE) exec dev bash

gpu: ## Abre una shell con GPU (solo laboratorio)
	$(COMPOSE) --profile gpu run --rm gpu bash

demo: ## Ejecuta la demo Streamlit en localhost:8501
	$(COMPOSE) exec dev streamlit run apps/demo/app.py \
		--server.address 0.0.0.0 --server.port 8501

test: ## Ejecuta las pruebas
	$(COMPOSE) exec dev pytest

lint: ## Verifica estilo y errores
	$(COMPOSE) exec dev ruff check .
	$(COMPOSE) exec dev ruff format --check .

fmt: ## Formatea el código
	$(COMPOSE) exec dev ruff format .
	$(COMPOSE) exec dev ruff check --fix .

lock: ## Regenera uv.lock (ejecutar en el host)
	uv lock

synth: ## Genera el dataset sintético v1 en el host (ARGS="--count 42")
	uv run python -m fraude.synth.build $(ARGS)

audit-samples: ## Audita el corpus local y ejecuta calidad/forense sin OCR
	uv run python -m fraude.eval.corpus_audit $(SAMPLES) \
		--pipeline --out artifacts/eval/muestras_audit.json

ocr-samples: ## Ejecuta OCR real con docTR sobre el corpus local (requiere extra ml)
	uv run python -m fraude.eval.ocr_samples $(SAMPLES) \
		--out artifacts/eval/ocr_samples.json --text-dir artifacts/ocr/samples

build: ## Reconstruye la imagen
	$(COMPOSE) build

clean: ## Elimina contenedores, caché local y artefactos generados
	$(COMPOSE) down
	rm -rf .cache artifacts/runs artifacts/mlruns
