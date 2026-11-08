SHELL := /bin/bash
COMPOSE := docker compose
CONFIG ?= configs/base.yaml
SAMPLES ?= data/muestras
TODAY ?= 2026-11-05

.DEFAULT_GOAL := help

.PHONY: help setup up down shell gpu demo test lint fmt lock synth synth-smoke synth-v2 validate-v2 folds-v2 train-f4 eval-f4 figures-f4 gradcam-f4 audit-samples ocr-samples eval-samples build clean

help: ## Muestra los comandos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

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

synth-smoke: ## Genera el lote de humo de 3 documentos con sus manifiestos
	uv run python -m fraude.synth.smoke

synth-v2: ## Genera el corpus v2 de 50 documentos (25 auténticos / 25 manipulados)
	uv run python -m fraude.synth.build --out data/synthetic/v2 \
		--count 25 --identities 10 --variants scan,photo,jpeg,rotate,shadow \
		--per-attack T1=4,T2=4,T3=3,T4=3,T5=3,T6=3,T7=3,T8=2 --holdout T5

validate-v2: ## Valida integridad, balance y particiones del corpus v2
	uv run python -m fraude.synth.validate data/synthetic/v2 \
		--expect-authentic 25 --expect-tampered 25 --require-splits

folds-v2: ## Genera los folds agrupados de 5 vías sobre el corpus v2
	uv run python -m fraude.synth.folds data/synthetic/v2 --folds 5 --seed 42

train-f4: ## Entrena F4 con folds agrupados (ARGS="--epochs 12 --device cpu")
	uv run python -m fraude.models.train --data data/synthetic/v2 \
		--out artifacts/models/f4 $(ARGS)

eval-f4: ## Evalúa las predicciones out-of-fold y genera la tabla
	uv run python -m fraude.models.evaluate --oof artifacts/models/f4/oof.jsonl \
		--out artifacts/models/f4/eval.json --markdown artifacts/models/f4/eval.md

figures-f4: ## Genera curvas, matriz de confusión, recall por ataque y calibración
	uv run python -m fraude.models.report --metrics artifacts/models/f4/metrics.json \
		--eval artifacts/models/f4/eval.json --out artifacts/models/f4/figures

gradcam-f4: ## Genera Grad-CAM de TP/TN/FP/FN del fold con mejor PR-AUC
	uv run python -m fraude.models.gradcam --checkpoint artifacts/models/f4/fold_2.pt \
		--data data/synthetic/v2 --oof artifacts/models/f4/oof.jsonl \
		--cache-dir artifacts/models/f4/cache --out artifacts/models/f4/gradcam

audit-samples: ## Audita el corpus local y ejecuta calidad/forense sin OCR
	uv run python -m fraude.eval.corpus_audit $(SAMPLES) \
		--pipeline --out artifacts/eval/muestras_audit.json

ocr-samples: ## Ejecuta OCR real con docTR sobre el corpus local (requiere extra ml)
	uv run python -m fraude.eval.ocr_samples $(SAMPLES) \
		--out artifacts/eval/ocr_samples.json --text-dir artifacts/ocr/samples

eval-samples: ## Evalúa el pipeline completo sobre las muestras en modos as_is y stripped
	uv run python -m fraude.eval.samples_eval $(SAMPLES) \
		--mode both --out artifacts/eval/samples_pipeline.json --today $(TODAY)

build: ## Reconstruye la imagen
	$(COMPOSE) build

clean: ## Elimina contenedores, caché local y artefactos generados
	$(COMPOSE) down
	rm -rf .cache artifacts/runs artifacts/mlruns
