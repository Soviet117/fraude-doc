# AGENTS.md — Contexto del proyecto para agentes de IA

Este archivo es la guía operativa para cualquier agente de IA (OpenCode, Claude Code, Codex, Cursor, etc.) que trabaje en este repositorio. **Léelo antes de tocar nada.**

---

## 1. Qué es este proyecto

Prototipo de investigación aplicada (curso de 8.º ciclo de Ingeniería de Sistemas) para detectar fraude documental en el CETPRO San Agustín, Nueva Cajamarca (Perú). Recibe la imagen de un documento académico —DNI, certificado de estudios, partida de nacimiento, ficha de matrícula, acta de evaluación, certificado de módulo ocupacional o título de Auxiliar Técnico— y devuelve un **nivel de riesgo de autenticidad con evidencias explicables**.

El sistema combina cuatro capas:

1. **Calidad** de la imagen (¿es analizable?).
2. **Forense de imagen** (¿fue manipulada?).
3. **OCR + reglas** (¿el contenido es coherente?).
4. **Consistencia semántica y de registro** (¿coincide con los datos institucionales?).

**No es** un sistema en producción. No hay despliegue institucional, ni integración con RENIEC/MINEDU, ni blockchain. Todo eso es trabajo futuro.

---

## 2. Fuente de verdad

El documento de referencia es **`plan-sistema-fraude-documental.md`** (raíz del repo). Contiene 18 secciones con stack, arquitectura, pipeline, fases, backlog, evaluación y riesgos.

Orden de lectura recomendado para un agente:

| Si vas a… | Lee |
|---|---|
| Entender el proyecto | §0, §1, §7 |
| Escribir código | §2 (ADRs), §3 (stack), §4 (estructura), §7.1 (contrato) |
| Planificar tareas | §8 (fases), §9 (backlog), §10 (división) |
| Entrenar o evaluar | §6 (datos), §7.4 (calibración), §11 (métricas) |
| Escribir el Cap. IV | §13 (mapeo con objetivos) |

Si hay conflicto entre una instrucción suelta y el plan, **gana el plan**, salvo que el usuario indique explícitamente lo contrario.

---

## 3. Estado actual del repositorio

**F0–F4 cerrados en el entorno local; F5–F9 pendientes.** El repositorio ya contiene:

- `pyproject.toml` + `uv.lock` (Python 3.13, deps core y extra `ml`).
- `Dockerfile`, `compose.yaml`, `Makefile`, `.pre-commit-config.yaml`, CI en `.github/workflows/ci.yml`.
- `src/fraude/core/schemas.py` con el contrato `Observation`.
- Ingesta, calidad, forense clásico, OCR docTR, extracción, reglas, registro y pipeline integrado (F2–F3).
- Dataset sintético v1 (`data/muestras`), corpus v2 de 50 documentos base y 300 imágenes con manifiestos, validación y folds agrupados (F1–F2).
- `src/fraude/models/` con F4 experimental: EfficientNet-B0 de página, entrenamiento con folds agrupados, evaluación OOF, calibración, Grad-CAM e inferencia.
- `apps/demo/app.py` (Streamlit) con pestañas de análisis heurístico, F4 experimental y estado de fases.
- `docs/` con guía de anotación, línea base F3, auditoría/OCR de muestras, `f4_corpus_v2.md` con resultados y límites, y `presentacion.md` con el guion de sustentación.

La imagen Docker incluye el grupo `dev`, por lo que `make test` y `make lint` funcionan dentro del contenedor. **Pendiente de F0:** verificar la misma imagen en la laptop del compañero y en el laboratorio (SSH + `nvidia-smi`).

**Pendiente de F5–F9:** localización/parches, rama semántica, fusión calibrada, evaluación integral con varias semillas y cierre de demo/Cap. IV. No asumas que esas piezas existen: verifica antes de ejecutar. Los datos y artefactos (`data/`, `artifacts/`) están ignorados por git y no se versionan.

---

## 4. Stack (versiones fijadas)

| Componente | Versión / elección |
|---|---|
| Python | **3.13** (no 3.15: PyTorch 2.14 no lo soporta) |
| Gestor | `uv` 0.13.x, `uv.lock` versionado |
| Deep learning | PyTorch 2.14 + torchvision + timm (wheels **cu126**) |
| CV | OpenCV, scikit-image, albumentations |
| OCR | **docTR** (primario); PaddleOCR solo como benchmark |
| NLP | spaCy `es_core_news_lg`, rapidfuzz, sentence-transformers |
| LLM local | Ollama + `qwen3:4b-instruct-2507` (modo `/no_think`) |
| Fusión | LightGBM + calibración isotónica |
| Demo | Streamlit |
| Persistencia | SQLite + sistema de archivos |
| Experimentos | MLflow (file store) |
| Entorno | Docker + Docker Compose (perfil `gpu` opcional) |
| Calidad | ruff, pytest, pre-commit, GitHub Actions (sin GPU) |

No cambies de tecnología sin una razón documentada y aprobada. Las decisiones están justificadas en §2 del plan.

---

## 5. Reglas no negociables

1. **Ningún resultado sin reproducibilidad.** Todo número que vaya al informe debe salir de un comando del repo (`make eval`, `make train`), nunca de una celda de notebook no reproducible.
2. **Ninguna decisión automática.** El sistema prioriza revisión humana; nunca rechaza un documento por sí solo.
3. **Ninguna señal sin explicación.** Si una señal no se puede explicar al usuario, no se usa como evidencia.
4. **Ningún dato personal fuera del equipo.** Los documentos y datasets no salen del entorno local. Prohibido enviar imágenes o PII a APIs externas (OpenAI, Google, Anthropic, etc.).
5. **Ningún modelo sin calibración.** Un score sin calibrar no es un riesgo, es un número. Protocolo en §7.4.
6. **El LLM nunca decide.** Solo redacta explicaciones a partir de hallazgos estructurados. Siempre debe existir un fallback por plantilla que funcione con el LLM apagado.
7. **Ningún alcance nuevo.** Lo que no está en el plan va a "trabajo futuro". No agregues features, servicios ni infraestructura por iniciativa propia.
8. **No inventes datos.** Si faltan documentos reales, se usan sintéticos y se declara la limitación. Nunca simules métricas ni resultados.

---

## 6. Contrato de datos

El objeto `Observation` (§7.1 del plan) es la interfaz entre módulos y el registro de auditoría. Vive en `src/fraude/core/schemas.py`.

- **No cambies el esquema sin confirmación explícita del usuario**: lo consumen dos personas y todos los módulos.
- Todo módulo recibe una ruta de imagen y devuelve un `Observation` serializable.
- Los mapas (heatmaps, máscaras) se referencian por ruta, no se embeben en JSON.
- Cualquier campo nuevo debe ser opcional y documentado.

---

## 7. Convenciones de código

- Código, nombres de variables y comentarios: **inglés**. Documentación, UI y mensajes al usuario: **español**.
- Type hints obligatorios en funciones públicas.
- Pydantic v2 para esquemas; dataclasses solo para objetos internos simples.
- Configuración por YAML (`configs/`), nunca constantes mágicas dispersas.
- Semillas fijas y registradas en cada corrida.
- Sin `print` en código de librería: usar logging estructurado.
- Funciones puras y testeables en `preprocess/`, `forensics/`, `rules/`, `eval/`.
- Cada módulo nuevo necesita al menos un test unitario.

---

## 8. Flujo de trabajo

```
main protegida
  └── feat/<descripcion-corta>  → PR → revisión del otro integrante → squash merge
```

- Commits convencionales: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- `pre-commit` corre ruff format + ruff check antes de cada commit.
- CI: lint + tests en CPU. **El CI nunca debe requerir GPU.**
- Datos (`data/`) y artefactos (`artifacts/`) están en `.gitignore`. Nunca los commitees.
- Código viaja por git; datos por `rsync`. No copies datasets al repo para "probar".

### 3 PCs, un entorno

- Todas las máquinas usan la **misma imagen Docker**.
- La PC del laboratorio (Ubuntu + GPU 8 GB) es un **servidor de cómputo**, no una máquina de desarrollo.
- Jobs largos siempre en `tmux`, nunca en una sesión SSH desnuda.
- La imagen conserva el venv en `/opt/venv` para que el bind-mount de `/app` no lo borre.

---

## 9. Qué hacer y qué no al usar un agente

**Haz:**

- Pide leer el plan antes de proponer una solución.
- Propón cambios pequeños y verificables, con el comando que los valida.
- Si una tarea toca datos, modelos o calibración, cita la sección del plan que la gobierna.
- Actualiza la documentación (`docs/`, model card, data card) junto con el código.
- Si algo del plan es técnicamente inviable, dilo con evidencia y propón la alternativa.

**No hagas:**

- No agregues FastAPI, Postgres, Next.js, Redis, Kubernetes ni microservicios: el alcance es Streamlit + SQLite (ADR-10).
- No reemplaces docTR por PaddleOCR "porque sí".
- No introduzcas un LLM en la ruta de decisión (ADR-05).
- No entrenes con datos sin particiones agrupadas (ADR-09): produce fuga de datos y métricas falsas.
- No reportes accuracy sola: exige Precision, Recall, F1, FPR, PR-AUC e IC bootstrap (§11).
- No uses el umbral 0.5 sin reportar también el umbral calibrado y el oráculo (§7.4).
- No borres ni reescribas el plan sin autorización.

---

## 10. Comandos disponibles

| Comando | Función |
|---|---|
| `make setup` | Crea `.env` con el UID/GID del usuario |
| `make up` | Levanta el contenedor de desarrollo |
| `make shell` | Shell dentro del contenedor |
| `make gpu` | Consola con GPU (solo laboratorio) |
| `make demo` | Streamlit en `localhost:8501` |
| `make test` | pytest dentro del contenedor |
| `make lint` / `make fmt` | ruff check / format |
| `make lock` | Regenera `uv.lock` en el host |
| `make build` | Reconstruye la imagen |
| `make clean` | Detiene contenedores y limpia artefactos |

Sin Docker, para trabajo liviano (tests, lint, esquemas): `uv sync && uv run pytest`.
La extra `ml` (torch, OCR, modelos) solo se instala en la imagen o con
`uv sync --extra ml`.

---

## 11. Contexto de dominio mínimo

- **CETPRO**: Centro de Educación Técnico-Productiva (Perú). Emite títulos de Auxiliar Técnico y certificados de módulos ocupacionales.
- **Documentos del alcance**: admisión (DNI, certificado de estudios, partida), matrícula (ficha), egreso (actas, certificados de módulo), titulación (título de Auxiliar Técnico).
- **Taxonomía de fraude T1–T8**: ver §6.2 del plan. Toda etiqueta del dataset debe usar esos IDs.
- **Límite legal**: los documentos de identidad los emite RENIEC; el sistema no puede validarlos oficialmente, solo detectar manipulación e inconsistencias.
- **Ley N.° 29733**: protección de datos personales. Anonimización obligatoria antes de cualquier procesamiento.

---

## 12. Contacto y decisiones abiertas

Pendientes de confirmar (§17 del plan):

- Acceso SSH desde fuera del campus (plan B: Tailscale).
- Modelo exacto de GPU del laboratorio.
- Sistema operativo del compañero (si es Windows: Docker Desktop + WSL2).
- Entrega de documentos reales por parte del CETPRO.
- Disponibilidad de evaluadores expertos para el kappa.

Si una tarea depende de una de estas decisiones, **pregunta antes de asumir**.
