# Sistema de detección de fraude documental — CETPRO San Agustín

Prototipo de investigación aplicada que analiza documentos académicos (DNI, certificados de estudios, partidas, fichas de matrícula, actas, certificados de módulo y títulos de Auxiliar Técnico) combinando **visión por computadora** y **procesamiento de lenguaje natural**, y devuelve un **nivel de riesgo de autenticidad con evidencias explicables**.

> Estado: **fase F0** — scaffold operativo (Docker, contrato de datos, ingesta por hash y demo mínima). El pipeline de análisis se implementa en las fases F2–F7.

## Documentación

| Archivo | Para qué |
|---|---|
| [`plan-sistema-fraude-documental.md`](plan-sistema-fraude-documental.md) | Plan completo: stack, arquitectura, pipeline, fases, backlog, evaluación y riesgos |
| [`AGENTS.md`](AGENTS.md) | Contexto y reglas para agentes de IA que trabajen en el repo |

## Resumen técnico

- **Datos**: documentos reales anonimizados + generador programático de documentos sintéticos auténticos y manipulados (etiquetas a nivel de píxel) + DocTamper para preentrenamiento.
- **Pipeline**: calidad → tipo de captura → forense (ELA, ruido, cromática, metadatos, copy-move) → OCR → extracción de campos → reglas → cruce con registro → fusión calibrada (LightGBM) → explicación.
- **Stack**: Python 3.13, PyTorch 2.14, docTR, spaCy, LightGBM, Streamlit, SQLite, Docker.
- **Infraestructura**: una imagen Docker idéntica en 3 PCs; el laboratorio (Ubuntu + GPU) se usa por SSH solo para entrenamiento.

## Principios

1. Ningún resultado sin reproducibilidad.
2. Ninguna decisión automática: el sistema prioriza revisión humana.
3. Ninguna señal sin explicación.
4. Ningún dato personal fuera del equipo.
5. Ningún modelo sin calibración.

## Arranque

Ver §16 del plan. Primera sesión: repositorio + scaffold Docker + contrato `Observation` + Streamlit mínimo + CI, y en paralelo la solicitud de documentos al CETPRO.
