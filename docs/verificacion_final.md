# Verificación final del prototipo

**Fecha:** 2026-11-11
**Rama:** `feat/cierre-sistema`
**Commits:** `576cb4b`, `786b53d`, `ee94d2a`
**Destino:** sustentación del curso (2026-11-12)

Este documento reúne la evidencia de que el sistema está integrado y es
presentable. No sustituye el Cap. IV ni las model cards; sirve como checklist
de cierre.

## 1. Checklist técnico

| Verificación | Comando | Resultado |
|---|---|---|
| Tests | `uv run pytest -q` | **254 tests en verde** |
| Lint | `uv run ruff check .` | **All checks passed** |
| Formato | `uv run ruff format --check .` | **135 archivos formateados** |
| Tablas y figuras OOF | `make eval` | F4/F5/F6/F7 + seeds + figuras regeneradas |
| Sanity integral | `make eval-system` | 50 documentos v2 procesados sin errores |
| Demo (AppTest) | `streamlit.testing.v1.AppTest` | **0 excepciones**; descargas JSON/PDF y feedback SQLite presentes |
| Config Docker | `docker compose config` | válido |
| Enlaces de documentación | script de verificación | **0 enlaces rotos** |

## 2. Artefactos que deben existir

| Ruta | Contenido |
|---|---|
| `artifacts/models/f4/fold_2.pt` | checkpoint F4 usado por la demo |
| `artifacts/models/f5/fold_0.pt` | checkpoint F5 usado por la demo |
| `artifacts/models/f7/fusion.joblib` | modelo + calibrador F7 |
| `artifacts/models/f7/metrics.json` | métricas, umbrales, ablaciones, SHAP |
| `artifacts/models/f7/seeds/seed_sweep.json` | barrido de 5 semillas |
| `artifacts/history.db` | historial SQLite (documentos, análisis, feedback) |
| `docs/informe_ejemplo.pdf` | informe PDF de ejemplo |
| `docs/figuras/*.png` | casos de uso, módulos, clases, secuencia, estados, despliegue, base de datos, flujo y resultados |
| `MANUAL_USUARIO.txt` | manual de usuario paso a paso |
| `docs/modulos.md` | detalle módulo por módulo |

Los datos (`data/`) y artefactos (`artifacts/`) están ignorados por git; en un
clon limpio se regeneran con los comandos del README. Las figuras de
`docs/figuras/` sí están versionadas para las diapositivas.

## 3. Resultados que se pueden citar

| Rama | Resultado | Condición |
|---|---|---|
| F3 `stripped` | F1 0.200, FPR 0.083 | 20 muestras v1 |
| F4 página | PR-AUC 0.712, F1 0.541, FPR 0.080 | 270 OOF v2 |
| F5 localización | Pixel-F1 calibrado 0.915 | 270 OOF; T1–T4 ≈ 0 |
| F6 extracción | Tipo documental 0.960 | 50 documentos v2 |
| F7 fusión | PR-AUC 0.980, F1 0.898, FPR 0.080, ECE 0.070 | 50 OOF v2 |
| F7 semillas | PR-AUC 0.9800 ± 0.0010 | 5 semillas |
| Pipeline integral | 25/25 auténticos en bajo; 24/25 manipulados en alto | **in-sample**, no validación |

Todo el corpus es sintético. No hay documentos reales del CETPRO ni validación
RENIEC/MINEDU. El pipeline integral in-sample se usa solo para verificar el
cableado; la métrica que se cita es la OOF.

## 4. Demo

```bash
git switch feat/cierre-sistema
make demo-local
# o, con Docker encendido:
make up && make demo
```

Pestañas: **Análisis integral** (F3+F4+F5+F6+F7), **Análisis F3**, **F4
experimental**, **Historial** (SQLite + revisión humana) y **Fases**.

El historial local quedó sembrado con dos ejemplos (un auténtico en riesgo bajo
y un T7 en riesgo alto) para que la pestaña *Historial* no aparezca vacía en la
sustentación. Esos datos viven en `artifacts/`, no en git.

## 5. Límites declarados

1. **Todo sintético**; sin documentos reales del CETPRO.
2. **n = 50 documentos base y 10 personas**; IC anchos.
3. **F5 no localiza T1–T4**; a 128 px las ediciones pequeñas se pierden.
4. **T7 es una copia exacta**; no mide reutilización real.
5. **F8 parcial**: hay 5 semillas, pero faltan robustez, McNemar y kappa.
6. **F0**: no se verificó la imagen Docker en las 3 PCs; aquí no hay daemon de
   Docker ni GPU. `docker compose config` sí es válido.
7. **El LLM no decide**: la plantilla es el fallback probado; Ollama es opcional.

## 6. Cierre pendiente después de la presentación

- Robustez JPEG/rotación/desenfoque y kappa (F8).
- Verificación en laptop del compañero y laboratorio (F0).
- Etiqueta `v1.0` en git y Cap. IV definitivo.
