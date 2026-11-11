# Manual breve de la demo

Este manual describe cómo arrancar el sistema completo y qué mostrar en la
sustentación. La demo es local: no envía documentos a ningún servicio externo.

## 1. Arranque

### Con Docker (entorno oficial)

```bash
make setup
make up
make demo          # http://localhost:8501
```

### Sin Docker (host con el venv del repo)

```bash
uv sync --extra ml
uv run streamlit run apps/demo/app.py
```

Si los artefactos no existen, generarlos una vez:

```bash
make synth-v2
make validate-v2
make folds-v2
make train-f4 ARGS="--epochs 12 --device cpu"
make eval-f4
make train-f5 ARGS="--epochs 12 --device cpu --image-size 128"
make eval-f5
make pipeline-v2
make eval-f6
make features-f7
make train-f7
```

La demo funciona igual si falta F4/F5/F7: muestra las señales disponibles y lo
declara. Si falta el corpus v2, se puede usar el corpus v1 o subir una imagen.

## 2. Pestañas

| Pestaña | Qué muestra |
|---|---|
| **Análisis integral** | F3 + F4 + F5 + F6 + F7: banda, score, evidencias SHAP, mapa F5, explicación, PDF/JSON |
| **Análisis F3** | Calidad, forense, OCR, campos, reglas y registro por separado |
| **F4 experimental** | Score de página y Grad-CAM |
| **Historial** | Análisis guardados en SQLite y revisión humana |
| **Fases del proyecto** | Estado real F0–F9 |

## 3. Guion de 5–7 minutos

1. **Abrir la demo** y mostrar la barra lateral. Explicar que hay dos corpus
   sintéticos y que el v2 es el que entrena F4/F5/F7.
2. **Análisis integral con un auténtico:** elegir `dni_0000` (v2). Mostrar banda
   baja, score, evidencias protectoras y mapa F5. Aclarar que un mapa con
   probabilidad alta en una zona no implica fraude: F5 es experimental.
3. **T7:** elegir `dni_0007_t7_c1`. Mostrar la evidencia de reutilización y el
   score alto. Decir que T7 es una copia exacta en este corpus y que no mide
   reutilización real.
4. **T1:** elegir `dni_0000_t1`. Mostrar que F4/F5 no ven la edición pequeña,
   pero la regla de registro sí; la fusión sube el riesgo. Es el argumento de
   por qué el sistema es híbrido.
5. **T6:** elegir `certificado_modulo_0005_t6`. Mostrar score alto y mapa F5
   cubriendo la página. Explicar que el tipo detectado puede fallar; la fusión
   usa varias señales.
6. **PDF/JSON:** descargar el informe y mostrar que incluye evidencias, campos,
   explicación, mapa y limitaciones.
7. **Historial:** mostrar el análisis guardado y registrar una revisión humana.
   Esto demuestra la trazabilidad y el principio de “ninguna decisión
   automática”.
8. **Fases:** cerrar con F0–F7 cerradas en local, F8 parcial y F9 en curso.
   Mencionar que no hay documentos reales y que la validación institucional es
   trabajo futuro.

## 4. Cifras que se pueden citar

| Rama | Resultado | Corpus |
|---|---|---|
| F3 (stripped) | F1 0.200, FPR 0.083 | 20 muestras v1 |
| F4 página | PR-AUC 0.712, F1 0.541, FPR 0.080 | 270 OOF v2 |
| F5 localización | Pixel-F1 calibrado 0.915, pero T1–T4 ≈ 0 | 270 OOF v2 |
| F6 extracción | Tipo 0.960; recall T7 1.000 (trivial) | 50 documentos v2 |
| F7 fusión | PR-AUC 0.980, F1 0.898, FPR 0.080, ECE 0.070 | 50 OOF v2 |

No citar F7 como “el sistema detecta el 98 % de fraudes reales”. Decir:
“en el corpus sintético, la fusión separa las clases con estas métricas de
viabilidad”.

## 5. Archivos que genera

| Ruta | Contenido |
|---|---|
| `artifacts/runs/<hash>/` | Imagen subida, mapa F5 (`*_f5_map.npy`) y overlay (`*_f5_overlay.png`) |
| `artifacts/reports/<hash>.pdf` | Informe PDF descargable |
| `artifacts/history.db` | SQLite con documentos, análisis y revisiones |
| `artifacts/models/f7/figures/` | ROC/PR, confusión, calibración, umbrales, ablaciones, SHAP |

## 6. Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| “F5 no disponible” | Falta `artifacts/models/f5/fold_0.pt` o la imagen no es legible |
| “F7 no disponible” | Falta `artifacts/models/f7/fusion.joblib`; ejecutar `make train-f7` |
| La explicación dice `template` | Ollama no está corriendo; el fallback es correcto y probado |
| El PDF tarda | Se genera en cada análisis; se guarda por hash y se reutiliza |
| El historial está vacío | Cada sesión nueva guarda al analizar; revisar `artifacts/history.db` |
| `uv` falla por caché | Usar `.venv/bin/python` o configurar `UV_CACHE_DIR` en el workspace |

## 7. Apagar y limpiar

```bash
make down          # contenedores
make clean         # contenedores + caché + runs temporal
```

`make clean` no borra `data/` ni `artifacts/models/`; para una limpieza total,
hacerlo a mano con cuidado.
