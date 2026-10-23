# Plan de desarrollo — Sistema de detección de fraude documental (CETPRO San Agustín)

**Tipo de proyecto:** prototipo funcional de investigación aplicada (curso de 8.º ciclo).
**No es** un sistema en producción ni un despliegue institucional.
**Meta única de cierre:** todo lo planificado debe estar terminado y sustentado **antes de que acabe el ciclo**.
**Equipo:** 2 personas. **Infraestructura:** 2 laptops + 1 PC de laboratorio con Ubuntu y GPU.

---

## 0. Resumen ejecutivo

Se construirá un prototipo que recibe la imagen de un documento académico (DNI, certificado de estudios, partida, ficha de matrícula, acta de evaluación, certificado de módulo, título de Auxiliar Técnico), lo analiza en cuatro capas —**calidad, forense de imagen, OCR/reglas, consistencia semántica**— y devuelve un **nivel de riesgo de autenticidad con evidencias explicables**: mapa de calor de la zona sospechosa, campos extraídos, reglas violadas y una explicación en lenguaje natural.

El aporte no es "un modelo que dice falso/verdadero". El aporte es un **pipeline híbrido, calibrado y explicable**, donde cada señal es auditable y ninguna decisión es automática.

**Tesis técnica que sostiene el proyecto:** en documentos escaneados, ninguna señal aislada funciona; el valor está en la fusión de señales forenses, semánticas y de registro, con calibración explícita. Esto está respaldado por la literatura reciente (ver §1).

---

## 1. Base tecnológica verificada (octubre 2026)

Todo el plan se apoya en estas versiones/estados. Verificar antes de fijar el lockfile.

| Componente | Versión / estado | Implicación para el plan |
|---|---|---|
| PyTorch | **2.14.1** (sep-2026), torchvision 0.29.1, Python 3.10–3.14 | Fijar **Python 3.13** (no 3.15: `uv` 0.13 ya lo pone por defecto y torch aún no lo soporta) |
| CUDA wheels | 12.6 / 13.0 / 13.2 | Usar **cu126** por compatibilidad con drivers de laboratorio |
| uv | **0.13.5** | `uv.lock` versionado; `uv sync --frozen` en Docker |
| docTR | **1.1.0** (ago-2026), solo PyTorch | OCR primario: un solo framework, buena precisión en latino |
| PaddleOCR | **3.7.0** (PP-OCRv5/v6) | Solo como benchmark alternativo; añade PaddlePaddle como segundo framework |
| Qwen3 (Ollama) | `qwen3:4b` (2.5 GB), `qwen3:8b` (5.2 GB); variante `qwen3:4b-instruct-2507` | LLM local solo para redactar explicaciones, en modo `/no_think` |
| TruFor | SOTA general en localización, con mapa de fiabilidad | Se usa como **señal preentrenada**, no como verdad absoluta |
| ADCD-Net / DTD | Específicos de documentos, con código público | Candidatos para localización de texto alterado |
| DocTamper | 170k imágenes con máscaras a nivel de píxel | Preentrenamiento y evaluación cruzada (licencia: uso no comercial, solicitud con correo institucional) |
| DocForge-Bench (2026) | Hallazgo clave: **AUC alto, F1 casi cero** en documentos por mala calibración de umbral | Obliga a un protocolo de calibración explícito (§7.4) |

### Hallazgos que corrigen el planteo inicial del informe

1. **ELA y JPEG ghosts solo sirven si la imagen es JPEG.** Los escaneos a 600 dpi suelen guardarse en PNG/TIFF sin pérdida: ahí ELA no revela nada. El plan incluye un **clasificador de tipo de captura** (escaneo / foto de celular / captura de pantalla / PDF reimpreso) y activa las señales forenses que corresponden a cada uno.
2. **La calibración es el problema real, no el modelo.** Un detector con Pixel-AUC 0.83 puede dar Pixel-F1 ≈ 0 si se usa el umbral 0.5. Se reserva un conjunto de calibración y se reportan métricas con umbral calibrado y umbral oráculo.
3. **Las señales de ruido/cromáticas superan a las de compresión en documentos escaneados sin pérdida.** Se implementan ambas familias y se reporta cuál funciona en cada tipo de captura.
4. **La GAN no es la vía principal de síntesis.** Se necesita *ground truth* a nivel de píxel para evaluar localización; eso lo da un generador programático de manipulaciones, no una GAN. La GAN queda como **ablación experimental** documentada.

---

## 2. Decisiones de arquitectura (ADR)

| ID | Decisión | Razón |
|---|---|---|
| ADR-01 | El entregable es un **prototipo demostrable**, no un despliegue | Alcance real del curso; se declara en limitaciones |
| ADR-02 | **Docker como único entorno** en las 3 PCs | Elimina divergencia de entornos y el "en mi máquina funciona" |
| ADR-03 | **Una sola imagen** que corre con o sin GPU | Menos modos de fallo; mismo binario en laptop y laboratorio |
| ADR-04 | **docTR** como OCR primario; PaddleOCR solo benchmark | Un solo framework (PyTorch), instalación limpia |
| ADR-05 | El **LLM nunca decide**; solo redacta la explicación | Riesgo legal/académico de alucinación; fallback a plantilla |
| ADR-06 | **Síntesis programática** como fuente principal; GAN como ablación | Etiquetas a nivel de píxel y control del tipo de ataque |
| ADR-07 | **Calibración obligatoria** y umbrales por tipo documental | Evidencia de DocForge-Bench 2026 |
| ADR-08 | Forense **condicionada al tipo de captura** | ELA/JPEG inútil en escaneos sin pérdida |
| ADR-09 | Splits **agrupados** + conjunto de **ataque no visto** | Evitar fuga de datos entre aumentaciones del mismo documento |
| ADR-10 | **Streamlit + SQLite** en vez de frontend/backend completo | Coste mínimo, suficiente para demostrar el sistema |
| ADR-11 | **Git para código, `rsync` para datos** | Los datasets no entran al repositorio |
| ADR-12 | Contrato de datos congelado (`Observation`) desde la fase 0 | Permite trabajar en paralelo sin pisarse |

---

## 3. Stack definitivo

| Capa | Herramienta | Nota |
|---|---|---|
| Lenguaje | Python **3.13** + `uv` | `.python-version` fijado; `uv.lock` versionado |
| Deep learning | PyTorch 2.14 + torchvision + **timm** | AMP, `torch.compile` opcional |
| CV clásico | OpenCV, scikit-image, **albumentations** | Preproceso y señales forenses |
| OCR | **docTR** | Detección + reconocimiento con coordenadas |
| NLP | **spaCy** `es_core_news_lg`, **rapidfuzz**, **sentence-transformers** | NER, fuzzy match, embeddings |
| LLM local | **Ollama + qwen3:4b-instruct-2507** | Solo explicación; `/no_think`, temperatura 0 |
| Fusión | **LightGBM** + scikit-learn (calibración isotónica) | Score final calibrado |
| Explicabilidad | SHAP (fusión), mapas de TruFor/ADCD, reglas con severidad | Evidencia auditable |
| Demo | **Streamlit** | Subida, resultados, heatmap, exportación |
| Persistencia | **SQLite** + sistema de archivos | Sin servidores adicionales |
| Experimentos | **MLflow** (file store) | Cada corrida atada a git SHA + hash de config |
| Entorno | **Docker + Docker Compose** | Perfil `gpu` opcional |
| Acceso remoto | **SSH + tmux + Tailscale** | Tailscale resuelve el acceso desde fuera del campus |
| Calidad | ruff, pytest, pre-commit, GitHub Actions (CPU) | El CI nunca requiere GPU |

---

## 4. Estructura del repositorio

```
fraude-doc/
├── pyproject.toml  uv.lock  .python-version
├── Dockerfile  compose.yaml  Makefile  .env.example
├── .pre-commit-config.yaml  .gitignore  README.md
├── configs/
│   ├── base.yaml
│   ├── data/{synthetic.yaml, real.yaml}
│   ├── model/{doctype.yaml, tamper.yaml, localize.yaml, vae.yaml, fusion.yaml}
│   └── eval/{protocol.yaml, robustness.yaml}
├── src/fraude/
│   ├── core/          # config, logging, rutas, esquemas Pydantic
│   ├── ingest/        # hash, dedupe, metadatos de archivo
│   ├── preprocess/    # calidad, deskew, recorte, normalización, tipo de captura
│   ├── forensics/     # ela, noise, chromatic, jpeg, metadata, copymove, textino
│   ├── ocr/           # motor docTR, layout, texto y coordenadas
│   ├── extract/       # extractores por tipo documental
│   ├── rules/         # fechas, catálogos, aritmética, registro
│   ├── nlp/           # embeddings, duplicados, llm_explain
│   ├── models/        # doctype, tamper, localize, vae, fusion, calibration
│   ├── synth/         # plantillas, render, tamper, augment, manifest
│   ├── pipeline/      # orquestador run.py, report.py
│   └── eval/          # métricas, robustez, ablaciones, tablas
├── apps/demo/         # app.py (Streamlit)
├── notebooks/         # 01_eda, 02_baseline, 03_ablaciones, 04_figuras
├── tests/             # unitarios + smoke end-to-end
├── data/              # NO versionado
│   ├── raw/ interim/ processed/ synthetic/ registry/
├── artifacts/         # NO versionado: models/, runs/, reports/
└── docs/
    ├── adr/  guia_anotacion.md  manual_demo.md
    └── cap4/   # figuras y tablas generadas por scripts
```

**Regla dura:** ningún resultado que aparezca en el Cap. IV puede provenir de una celda de notebook no reproducible. Todo número debe salir de un target de `make`.

---

## 5. Infraestructura: 3 PCs, un solo entorno

### 5.1 Modelo mental

- La PC del laboratorio es un **servidor de cómputo**, no una segunda máquina de desarrollo.
- `sshd` corre permanentemente allí. No se "activa" al llegar: te conectas desde donde estés.
- Con `sudo` disponible, **Tailscale** elimina la dependencia de que el firewall del campus permita SSH entrante.
- El código viaja por **GitHub**; los datos por **`rsync`**; los modelos por `artifacts/`.

### 5.2 Imagen única

`Dockerfile` con Python 3.13, `uv`, dependencias del sistema (`libgl1`, `libglib2.0-0`, `git`, `tmux`) y el proyecto instalado en `/opt/venv` (fuera de `/app`, para que el bind-mount no borre el entorno). torch se instala desde el índice **cu126**: la misma imagen funciona en CPU y en GPU.

`compose.yaml` con dos servicios:
- `dev`: bind-mount del repo, `user: "${UID}:${GID}"`, puertos 8501 (Streamlit), 5000 (MLflow).
- `gpu`: extiende `dev`, perfil `gpu`, reserva del dispositivo NVIDIA.

`.env` local por máquina con `UID`/`GID` (`id -u`, `id -g`) para evitar archivos `root:root`.

### 5.3 Preparación del laboratorio (una vez, con sudo)

1. `openssh-server` + `tmux`, servicio habilitado.
2. `tailscale up` → hostname estable accesible desde cualquier red.
3. Docker + `usermod -aG docker` + **nvidia-container-toolkit**.
4. Verificación obligatoria: `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`.
5. Evitar suspensión del equipo (coordinar con el administrador).
6. Anotar modelo exacto de GPU y driver (`nvidia-smi`).

### 5.4 Ciclo de trabajo

```
LAPTOP                          PC DEL COMPAÑERO                LABORATORIO
docker compose up -d            docker compose up -d            ssh lab
git push            ──────────▶ git pull            ──────────▶ git pull
forense, OCR, demo, tests       OCR/NLP, demo, tests            tmux new -s train
                                                                make train CONFIG=...
git pull ◀──────────────────────────────────────────────────── rsync artifacts/
```

Convenciones:
- Todo job largo en `tmux` (nunca en una sesión SSH desnuda).
- `ssh -N -L 5000:localhost:5000 -L 8501:localhost:8501 lab` para ver MLflow y la demo.
- `CUDA_VISIBLE_DEVICES` acordado cuando el laboratorio se comparta.
- Los datasets grandes viven fuera del repo y se sincronizan con `rsync`.

---

## 6. Datos

### 6.1 Fuentes

| Fuente | Uso | Volumen |
|---|---|---|
| Documentos reales del CETPRO (anonimizados) | Realismo, validación cualitativa, subconjunto de prueba | 300 según el informe; el plan tolera menos |
| Plantillas institucionales (formatos en blanco) | Base para generar documentos auténticos sintéticos | 5–7 tipos |
| Documentos sintéticos auténticos (render + simulación de captura) | Entrenamiento y volumen | según necesidad |
| Documentos sintéticos manipulados | Etiquetas a nivel de píxel y de campo | 2–4× los auténticos |
| DocTamper (externo) | Preentrenamiento de localización y prueba cruzada | 170k |

Si el CETPRO no entrega documentos a tiempo: el proyecto **no se detiene**; se ejecuta con sintéticos + DocTamper y se declara la limitación. Esto queda escrito desde el inicio.

### 6.2 Taxonomía de fraude (etiquetas)

| ID | Tipo | Capa que debería detectarlo |
|---|---|---|
| T1 | Sustitución de texto (fecha, nombre, nota) | Forense local + OCR/reglas |
| T2 | Inserción de sello o firma | Forense local + plantilla |
| T3 | Duplicación de sello/firma (copy-move) | Copy-move + forense |
| T4 | Borrado/limpieza (inpainting) | Ruido + localización |
| T5 | Reemplazo de fotografía (documento de identidad) | Forense + comparación |
| T6 | Documento completo fabricado | Plantilla + registro (no píxeles) |
| T7 | Reutilización de documento de otra persona | Hash + embeddings + registro |
| T8 | Inconsistencia semántica sin manipulación visual | Reglas + NLP |

Cada muestra lleva: etiqueta de documento, de página, **máscara/bbox de la región alterada** y **valor original vs alterado** por campo.

### 6.3 Anonimización y ética

- Consentimiento institucional por escrito; anonimización de nombres, DNI, direcciones y fotografías antes de cualquier procesamiento.
- Los datos crudos **nunca** salen del equipo local ni se suben a la nube.
- El LLM corre local; no se envían documentos a APIs externas.
- Registro de acceso y política de borrado al cierre del proyecto.
- El sistema **no rechaza**: solo prioriza revisión humana.

### 6.4 Particiones (anti-fuga)

- Split agrupado por documento-fuente/persona: ninguna aumentación del mismo documento cae en train y test a la vez.
- Estratificado por tipo documental y por tipo de ataque.
- **Conjunto de ataque no visto**: una familia de manipulación queda fuera del entrenamiento para medir generalización real.
- **Conjunto de calibración** separado del de prueba.
- **Conjunto real** (si existe) reservado para validación cualitativa y reporte honesto.

---

## 7. Pipeline técnico

### 7.1 Contrato de datos (congelado en fase 0)

Todo módulo recibe una ruta de imagen y devuelve un objeto `Observation` serializable. Es la interfaz entre las dos personas y el registro de auditoría del sistema.

```json
{
  "document_id": "sha256:...",
  "source": {"filename": "...", "format": "png", "bytes": 123},
  "capture": {"type": "scan|photo|screenshot|pdf_render", "dpi_est": 600},
  "quality": {"ok": true, "blur": 0.12, "skew_deg": -0.8, "issues": []},
  "doctype": {"label": "acta_evaluacion", "template_version": "2023", "confidence": 0.97},
  "forensics": {
    "ela": {"applicable": false, "score": null},
    "noise": {"score": 0.71, "map": "path.npy"},
    "chromatic": {"score": 0.22},
    "jpeg": {"applicable": true, "ghost_score": 0.35},
    "metadata": {"suspicious": true, "reasons": ["software=GIMP 2.10"]},
    "localization": {"model": "adcd-net", "score": 0.68, "mask": "path.png"}
  },
  "ocr": {"engine": "doctr", "mean_conf": 0.91, "n_words": 214},
  "fields": {"nombres": "...", "dni": "...", "fecha_emision": "..."},
  "rules": [{"code": "FECHA_FUTURA", "severity": "high", "message": "..."}],
  "registry": {"matched": true, "score": 0.94, "diffs": []},
  "fusion": {"p_fake": 0.83, "risk": "alto", "evidences": []},
  "timings_ms": {"preprocess": 320, "ocr": 1400, "models": 900, "total": 2620}
}
```

### 7.2 Etapas

1. **Ingesta** — hash SHA-256, deduplicación exacta, metadatos de archivo, asignación de ID.
2. **Puerta de calidad** — resolución, desenfoque, inclinación, contraste, brillo, recorte incompleto. Si falla, la salida es **"recapturar"**, no "fraude". Evita castigar al usuario por una mala foto.
3. **Normalización** — orientación EXIF, deskew, detección y recorte de página, normalización de resolución.
4. **Clasificador de tipo de captura** — decide qué señales forenses tienen sentido.
5. **Clasificador de tipo documental + versión de plantilla** — compara contra una galería de plantillas (hash perceptual + layout). Detecta el caso T6 (documento fabricado).
6. **Señales forenses** — ELA (solo JPEG), residuo de ruido, inconsistencias cromáticas, JPEG ghost/DQ, grilla de bloques, metadatos, copy-move (SIFT/ORB), inconsistencia de trazo/tinta por línea de texto, y opcionalmente TruFor/ADCD-Net.
7. **OCR + layout** — docTR; separación en regiones (encabezado, cuerpo, tabla, sello, firma).
8. **Extracción de campos** — esquema por tipo documental: regex + priors de posición + validación de tipo; LLM opcional solo con salida validada contra esquema.
9. **Reglas semánticas** — fechas coherentes (nacimiento < admisión < egreso, no futuras), edad plausible, nombre consistente entre campos, DNI con formato válido, catálogo de programas/módulos, correlativos por año, promedios, autoridades por periodo, versión de logo.
10. **Cruce con registro interno** — `students.csv`/`registry.json` del CETPRO (real o espejo sintético): nombre, DNI, programa, fechas. Coincidencia difusa con umbral.
11. **Duplicados y reutilización** — hash perceptual + embeddings del texto para detectar el mismo escaneo reutilizado.
12. **Fusión y calibración** — LightGBM sobre las señales; calibración isotónica; umbrales por tipo documental; decisión sensible al coste (un falso negativo pesa más que un falso positivo).
13. **Explicación** — evidencias ordenadas por contribución (SHAP + severidad de reglas) + resumen en lenguaje natural (LLM con plantilla de respaldo).
14. **Registro** — SQLite con `Observation` completo, versión de modelo, hash del documento y fecha.

### 7.3 Dos rutas de inferencia

- **Ruta rápida** (siempre): preproceso + forense clásico + OCR + reglas + registro.
- **Ruta profunda** (solo zona gris o cuando hay GPU): localización neuronal + LLM de explicación.

Esto mantiene la demo usable en laptops sin GPU y deja la parte pesada donde corresponde.

### 7.4 Protocolo de calibración (obligatorio)

1. Conjunto de calibración independiente del de prueba.
2. Calibración de probabilidades (isotónica o Platt) antes de fijar umbrales.
3. Umbral por tipo documental, elegido sobre la curva precision-recall según coste.
4. Reportar siempre: umbral fijo 0.5, umbral calibrado y umbral oráculo (techo).
5. Reportar ECE/Brier además de AUC.
6. Si el modelo no se calibra bien, la salida se degrada a **bandas de riesgo**, no a probabilidad.

---

## 8. Fases del proyecto

Sin duraciones. Cada fase tiene un **criterio de salida verificable**. No se avanza sin cumplirlo.

### F0 — Base operativa
Repositorio, estructura, Docker funcionando en las 3 PCs, contrato `Observation`, CI en verde, Streamlit "hola mundo".
**Salida:** `make up && make demo` funciona desde un clon limpio en laptop, laptop del compañero y laboratorio; `nvidia-smi` se ve dentro del contenedor.

### F1 — Datos y terreno
Solicitud formal al CETPRO, inventario documental, definición de la taxonomía T1–T8, guía de anotación, primeras 50 muestras etiquetadas, EDA (formatos, resoluciones, calidad, OCR baseline).
**Salida:** data card v1, tabla de caracterización documental, acuerdo de anonimización firmado.

### F2 — Preproceso y síntesis
Pipeline de calidad/normalización, clasificador de tipo de captura, generador de documentos sintéticos auténticos desde plantillas, generador de manipulaciones con máscaras, aumentación de captura, manifiesto reproducible por semilla.
**Salida:** dataset v1 con particiones agrupadas y conjunto de ataque no visto; muestras antes/después; manifiesto que reconstruye el dataset con un comando.

### F3 — Línea base sin deep learning
Forense clásico + OCR + extracción + reglas + cruce de registro. Score heurístico.
**Salida:** primera tabla de resultados (precisión, recall, F1, FPR) y matriz de confusión. Este es el **piso** que todo modelo debe superar.

### F4 — Modelo visual de documento
Clasificador de tipo documental y clasificador de manipulación a nivel de página y de parche (EfficientNet/ConvNeXt vía timm), con transfer learning, AMP y pesos balanceados.
**Salida:** curvas de entrenamiento, matriz de confusión, comparación contra F3, primeras ablaciones.

### F5 — Localización y explicabilidad visual
Modelo de segmentación (U-Net/DeepLab con encoder timm) o ajuste de ADCD-Net/DTD; inferencia de TruFor; mapas de calor superpuestos.
**Salida:** Pixel-F1, IoU y AUC con umbral calibrado y oráculo; figuras de aciertos y errores; comparación CNN vs ViT vs VAE un-clase.

### F6 — Rama semántica
NER, embeddings, detección de duplicados, reglas sobre datos extraídos, explicación con LLM local y plantilla de respaldo.
**Salida:** tabla de errores de extracción por campo, tasa de acierto de reglas, ejemplos de explicaciones; prueba de que el sistema funciona con el LLM apagado.

### F7 — Fusión y calibración
LightGBM sobre todas las señales, calibración, umbrales por tipo, análisis de importancia (SHAP).
**Salida:** modelo calibrado; barrido de umbrales; ECE/Brier; tabla de ablaciones completa A1–A10.

### F8 — Evaluación integral
Protocolo completo: validación cruzada agrupada con semillas, intervalos de confianza bootstrap, robustez (JPEG 50–90, rotación, desenfoque, reescalado, foto de celular), prueba con ataque no visto, comparación pre/post (revisión manual vs sistema), kappa con evaluadores humanos.
**Salida:** tabla maestra de resultados del Cap. IV + figuras + análisis de errores. Todo reproducido por `make eval`.

### F9 — Demo y cierre documental
Streamlit completo, manual breve, model card, data card, capítulo IV redactado con figuras generadas por scripts, ensayo de sustentación.
**Salida:** demo que corre desde un clon limpio; documento cerrado; repositorio etiquetado.

### Dependencias

```
F0 → F1 → F2 → F3 → F4 → F5 → F7 → F8 → F9
                    └→ F6 ──┘
```
F6 puede correr en paralelo con F4/F5 (otra persona). F9 arranca en paralelo desde F7 con las figuras ya disponibles.

---

## 9. Backlog priorizado

Prioridad MoSCoW. **MVP** = todo lo marcado como M.

| ID | Historia | Criterio de aceptación | Prioridad |
|---|---|---|---|
| E0-1 | Entorno Docker reproducible | `make up` funciona en 3 PCs; `make demo` abre | M |
| E0-2 | CI sin GPU | lint + tests en cada push | M |
| E0-3 | Acceso remoto al laboratorio | SSH por Tailscale + `nvidia-smi` en contenedor | M |
| E1-1 | Inventario documental | Tabla con tipo, año, formato, resolución, estado | M |
| E1-2 | Anonimización | Cero PII en `data/processed`; verificado por script | M |
| E1-3 | Guía de anotación | Taxonomía T1–T8 con ejemplos visuales | M |
| E1-4 | Etiquetado en Label Studio | ≥50 muestras con máscara y campo alterado | M |
| E2-1 | Puerta de calidad | Detecta blur, skew, recorte; salida "recapturar" | M |
| E2-2 | Generador de auténticos | N documentos desde plantillas con manifiesto | M |
| E2-3 | Generador de manipulaciones | 8 tipos T1–T8 con máscara y campo original | M |
| E2-4 | Aumentación de captura | escaneo, foto, JPEG, rotación, sombra | M |
| E2-5 | Particiones sin fuga | Script verifica que ningún grupo cruza train/test | M |
| E3-1 | Señales forenses clásicas | ELA, ruido, cromática, metadatos, copy-move | M |
| E3-2 | OCR + layout | Texto con coordenadas y confianza | M |
| E3-3 | Extracción por tipo | Esquema por documento; salida validada | M |
| E3-4 | Motor de reglas | ≥15 reglas con severidad y mensaje | M |
| E3-5 | Cruce de registro | Match difuso con umbral y diff de campos | M |
| E4-1 | Clasificador de tipo documental | F1 ≥ 0.95 en test | M |
| E4-2 | Clasificador de manipulación | Supera a F3 en F1 y PR-AUC | M |
| E4-3 | MLflow | Cada corrida con git SHA, config y métricas | M |
| E5-1 | Localización | Pixel-F1 con umbral calibrado > umbral 0.5 | S |
| E5-2 | TruFor como señal | Integrado y comparado | S |
| E5-3 | CNN vs ViT vs VAE | Tabla comparativa con IC | S |
| E6-1 | NER y normalización | Extrae nombres, fechas, DNI con métrica por campo | M |
| E6-2 | Duplicados | Detecta reutilización de escaneo | S |
| E6-3 | Explicación LLM | 3 líneas en español; fallback si falla o tarda | S |
| E7-1 | Fusión | Supera a la mejor señal individual | M |
| E7-2 | Calibración | ECE < 0.10; umbrales por tipo | M |
| E7-3 | Ablaciones A1–A10 | Tabla reproducible | M |
| E8-1 | Robustez | Métricas por degradación | S |
| E8-2 | Pre/post manual vs sistema | Tabla comparativa con tiempo por documento | S |
| E8-3 | Kappa con evaluadores | ≥2 evaluadores, 50 documentos | C |
| E9-1 | Demo completa | Subida → evidencias → heatmap → explicación → export | M |
| E9-2 | Manual y model card | Documento breve y ficha del modelo | M |
| E9-3 | Cap. IV | Secciones con figuras y tablas generadas por scripts | M |

M = obligatorio para el MVP · S = debería · C = si el tiempo lo permite.

---

## 10. División del trabajo

| Responsable | Módulos | Entregables propios |
|---|---|---|
| Persona A (tú) | `core`, `preprocess`, `forensics`, `models`, `eval`, infraestructura | Pipeline forense, modelos visuales, evaluación, CI |
| Persona B | `ingest`, `synth`, `ocr`, `extract`, `rules`, `nlp`, `apps/demo` | Dataset sintético, OCR/reglas, explicaciones, demo |
| Ambos | `pipeline`, `docs`, Cap. IV | Integración, resultados, redacción |

Reglas de convivencia:
- El contrato `Observation` solo cambia por PR conjunto.
- Cada módulo tiene dueño; nadie edita el de otro sin avisar.
- `main` protegida; ramas `feat/…`; PR con revisión del otro; merge squash.
- Un notebook nunca es fuente de verdad.

---

## 11. Evaluación

### 11.1 Métricas

| Nivel | Métricas |
|---|---|
| Documento | Precision, Recall, F1, FPR, ROC-AUC, **PR-AUC**, matriz de confusión |
| Calibración | Brier, ECE, curva de fiabilidad |
| Localización | Pixel-F1, IoU, Pixel-AUC (umbral calibrado y oráculo) |
| Extracción | Exactitud por campo, CER del OCR |
| Sistema | Latencia p50/p95, tiempo total por documento |
| Humano | Kappa de Cohen, SUS (si se consigue), tiempo de revisión manual |

### 11.2 Rigor

- Validación cruzada agrupada, varias semillas, **media ± desviación**.
- **Intervalos de confianza bootstrap** al 95 % (n pequeño: obligatorio).
- Prueba de McNemar para comparar modelos y para el pre/post.
- Desglose por tipo documental y por tipo de ataque: un promedio alto puede ocultar un tipo que falla siempre.
- Reportar siempre el caso "modelo sin calibración" para demostrar por qué la calibración importa.

### 11.3 Ablaciones obligatorias

A1 reglas · A2 +forense clásico · A3 +CNN · A4 +localización · A5 +NLP/registro · A6 +fusión calibrada · A7 CNN vs ViT vs VAE · A8 sintético solo vs +DocTamper · A9 robustez · A10 GAN on/off.

---

## 12. Demo (Streamlit)

Pantallas mínimas:

1. **Subida** — arrastrar imagen/PDF, seleccionar tipo declarado u "automático".
2. **Resultado** — banda de riesgo (bajo / medio / alto), `p_fake`, tiempo, tipo de captura detectado.
3. **Evidencias** — lista ordenada: "metadatos indican edición con GIMP", "fecha de emisión posterior al acta", "región del nombre con ruido inconsistente".
4. **Visor** — documento con heatmap superpuesto y bbox de la región sospechosa.
5. **Campos** — tabla extraído vs esperado vs registro, con diferencias resaltadas.
6. **Explicación** — párrafo en español, con etiqueta de "generado por LLM" y plantilla de respaldo.
7. **Historial** — consultas previas desde SQLite, con versión de modelo.
8. **Exportar** — PDF/JSON del informe.

Modo offline: si Ollama no está disponible, la demo funciona igual (explicación por plantilla). Esto se prueba explícitamente.

---

## 13. Mapeo con el Cap. IV y los objetivos del informe

| Objetivo específico (Cap. I) | Módulo | Evidencia en Cap. IV |
|---|---|---|
| OE1 Recopilar documentos | F1, `ingest`, `synth` | Tabla de caracterización, data card, taxonomía |
| OE2 Preparar documentos | F2, `preprocess` | Diagrama de pipeline, muestras antes/después |
| OE3 Diseñar el modelo | F3–F7 | Arquitectura, ablaciones, métricas |
| OE4 Desarrollar el sistema con interfaz | F9, `apps/demo` | Capturas, manual, diagrama C4 |
| OE5 Evaluar el desempeño | F8, `eval` | Tabla maestra, IC, pre/post, kappa |

Estructura sugerida del Cap. IV:

```
4.1 Introducción
4.2 Requerimientos (RF/RNF, casos de uso)
4.3 Arquitectura (C4: contexto, contenedores, componentes, despliegue)
4.4 Modelo de datos (ER + Observation)
4.5 Diseño del pipeline (flujo + contratos)
4.6 Dataset (composición, síntesis, etiquetado, ética)
4.7 Implementación (módulos, decisiones, capturas)
4.8 Experimentos y resultados (tablas, curvas, ablaciones, calibración, robustez)
4.9 Discusión (comparación con antecedentes: 88.8 % de Zorrilla, etc.)
4.10 Limitaciones
4.11 Conclusiones y trabajo futuro (despliegue en el CETPRO)
```

Lo que **no** va en el Cap. IV: despliegue institucional, capacitación real, SUS con personal del CETPRO, integración con RENIEC/MINEDU, blockchain. Va en trabajo futuro.

---

## 14. Riesgos y mitigaciones

| # | Riesgo | Impacto | Mitigación |
|---|---|---|---|
| R1 | El CETPRO no entrega documentos | Alto | Plan B con sintéticos + DocTamper; limitación declarada |
| R2 | Pocos datos reales (n≈300) | Alto | Transfer learning, parches, preentrenamiento externo, CV agrupada, IC bootstrap |
| R3 | Escaneos sin pérdida anulan ELA/JPEG | Alto | Clasificador de captura + señales de ruido/cromáticas + localización neuronal |
| R4 | Calibración deficiente | Alto | Protocolo §7.4; reportar umbral calibrado y oráculo; bandas de riesgo |
| R5 | Sobreajuste a artefactos del generador | Alto | Dominio aleatorizado, ataque no visto, prueba con fotos reales |
| R6 | Errores de OCR contaminan reglas | Medio | Gating por confianza, fuzzy matching, reglas tolerantes |
| R7 | Alucinación del LLM | Medio | El LLM no decide; esquema validado; plantilla de respaldo |
| R8 | 8 GB VRAM insuficientes | Medio | 384 px, batch pequeño, AMP, acumulación de gradiente, parches |
| R9 | Laboratorio compartido | Medio | `tmux`, coordinación de GPU, usuario propio en contenedor |
| R10 | Coordinación de 2 personas | Medio | Contrato congelado, propiedad por módulo, PR obligatorio |
| R11 | Deriva de alcance | Alto | ADR-01: prototipo; todo lo demás a trabajo futuro |
| R12 | Datos personales (Ley 29733) | Alto | Anonimización, local-only, consentimiento, borrado |

---

## 15. Definición de "terminado"

El proyecto está cerrado cuando:

1. Un clon limpio + `make up && make demo` funciona en las 3 PCs.
2. `make eval` reproduce la tabla de resultados completa, con IC, sin intervención manual.
3. La demo muestra riesgo, evidencias, heatmap, campos y explicación, y funciona con el LLM apagado.
4. Existen model card y data card con métricas, datos y limitaciones.
5. El Cap. IV tiene todas sus secciones con figuras y tablas generadas por scripts del repo.
6. Las limitaciones están escritas con honestidad: alcance sintético, muestra pequeña, sin validación oficial de RENIEC/MINEDU, sin despliegue.
7. El repositorio está etiquetado (`v1.0`) y el README permite a un tercero ejecutarlo.

---

## 16. Arranque inmediato (primera sesión)

1. Crear repositorio privado en GitHub e invitar al compañero.
2. Generar scaffold: `pyproject.toml` (Python 3.13 fijado), `uv.lock`, `Dockerfile`, `compose.yaml`, `Makefile`, `.env.example`, `.gitignore`, `.pre-commit-config.yaml`.
3. Implementar `core/schemas.py` con el `Observation` y un test que lo serialice.
4. Streamlit mínimo que suba una imagen y muestre el JSON del `Observation` vacío.
5. CI: ruff + pytest en CPU.
6. Probar en laptop, en la laptop del compañero y en el laboratorio.
7. En el laboratorio: SSH + Tailscale + Docker + NVIDIA toolkit + verificación con `nvidia-smi`.
8. En paralelo y sin depender de nada técnico: **enviar la solicitud formal de documentos al CETPRO** y preparar la guía de anotación.

---

## 17. Decisiones abiertas (con plan B)

| Decisión | Estado | Plan B si falla |
|---|---|---|
| SSH desde fuera del campus | Por confirmar | Tailscale (requiere sudo una vez, ya disponible) |
| Modelo exacto de GPU | Por confirmar | Ajustar batch/resolución; Kaggle/Colab solo con datos sintéticos |
| Sistema operativo del compañero | Por confirmar | Windows → Docker Desktop + WSL2 |
| Entrega de documentos reales | Por confirmar | Sintéticos + DocTamper; limitación declarada |
| Acceso a evaluadores expertos | Por confirmar | Kappa entre los dos integrantes + docente como tercero |

---

## 18. Principios que no se negocian

1. **Ningún resultado sin reproducibilidad.** Todo número del documento sale de un comando del repo.
2. **Ninguna decisión automática.** El sistema prioriza revisión humana; nunca rechaza.
3. **Ninguna señal sin explicación.** Si no se puede explicar, no se usa como evidencia.
4. **Ningún dato personal fuera del equipo.** Local, anonimizado, con consentimiento.
5. **Ningún modelo sin calibración.** Un score sin calibrar no es un riesgo, es un número.
6. **Ningún alcance nuevo.** Lo que no está en este plan va a trabajo futuro.
