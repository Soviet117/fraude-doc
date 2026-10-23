# Datos

Este directorio **no se versiona** (excepto este README). Contiene los documentos
y datasets del proyecto. Nunca subas datos personales al repositorio.

```
data/
├── raw/         # documentos originales (anonimizados antes de copiarlos aquí)
├── interim/     # resultados intermedios del preproceso
├── processed/   # imágenes normalizadas listas para el pipeline
├── synthetic/   # documentos generados (auténticos y manipulados) + manifiesto
└── registry/    # registro interno del CETPRO (real o espejo sintético)
```

Reglas:

- Todo documento real debe estar anonimizado antes de entrar a `raw/`.
- Los datasets se sincronizan entre las 3 PCs con `rsync`, no con git.
- Cada versión del dataset debe tener un `manifest.csv` con `sha256` y etiquetas.
- El generador sintético es reproducible por semilla (ver plan §6.4).
