# Diagramas del sistema

Diagramas para la sustentación y el Cap. IV. Las versiones PNG están en
`docs/figuras/` y se regeneran con:

```bash
make diagrams
```

## 1. Casos de uso

![Casos de uso](figuras/casos_uso.png)

Actores: **Evaluador del CETPRO** (consulta, analiza, exporta y registra la
revisión), **Soporte técnico** (entrena y calibra), **Sistema** (orquesta el
pipeline), **Ollama local** (solo redacta; opcional) y **Registro CETPRO**
(espejo sintético para el cruce). La decisión final es siempre humana.

```mermaid
flowchart LR
    E[Evaluador] --> U1((Subir documento))
    E --> U2((Analizar documento))
    E --> U3((Ver evidencias y mapa))
    E --> U4((Consultar historial))
    E --> U5((Exportar PDF/JSON))
    E --> U6((Registrar revisión))
    S[Soporte técnico] --> U7((Entrenar y calibrar))
    SUT[Sistema] --> U2
    SUT --> U8((Redactar explicación))
    O[Ollama local] -. opcional .-> U8
    R[Registro CETPRO] --> U2
```

## 2. Arquitectura de componentes

![Arquitectura](figuras/arquitectura.png)

Una sola imagen Docker. Streamlit orquesta; el pipeline ejecuta calidad,
forense, OCR, extracción, reglas, registro, F6 semántica, F4, F5 y F7. SQLite
guarda el historial; `artifacts/` conserva checkpoints y mapas. Ollama es
opcional y nunca participa en la decisión.

```mermaid
flowchart TB
    subgraph Presentacion[Presentación]
        UI[Streamlit]
    end
    subgraph Orquestacion[Orquestación]
        FULL[pipeline/full.py]
        RUN[pipeline/run.py]
    end
    subgraph Senales[Señales]
        Q[Calidad]
        F[Forense clásico]
        OCR[OCR docTR]
        EX[Extracción]
        RU[Reglas]
        REG[Registro]
    end
    subgraph Modelos[Modelos]
        M4[F4 EfficientNet-B0]
        M5[F5 U-Net]
        M6[F6 semántica]
        M7[F7 LightGBM]
    end
    subgraph Datos[Datos y artefactos]
        DB[(SQLite)]
        ART[artifacts/]
    end
    UI --> FULL --> RUN
    RUN --> Q & F & OCR & EX & RU & REG
    FULL --> M4 & M5 & M6 & M7
    M7 --> DB
    M4 & M5 --> ART
```

## 3. Modelo de datos (SQLite)

![ER](figuras/er_historial.png)

```mermaid
erDiagram
    documents ||--o{ analyses : "1:N"
    analyses ||--o{ feedback : "1:N"

    documents {
        TEXT document_id PK
        TEXT sha256
        TEXT filename
        TEXT doctype
        TEXT first_seen
        TEXT last_seen
    }
    analyses {
        INTEGER id PK
        TEXT document_id FK
        TEXT created_at
        REAL p_fake
        TEXT risk
        TEXT calibration_method
        TEXT model_versions
        TEXT explanation_source
        TEXT payload_json
    }
    feedback {
        INTEGER id PK
        INTEGER analysis_id FK
        TEXT created_at
        TEXT reviewer
        TEXT label
        TEXT notes
    }
```

El payload completo del análisis se guarda como JSON para reconstruir el
informe. `feedback` registra la revisión humana: el sistema no decide solo.

## 4. Flujo del pipeline

![Flujo](figuras/flujo_pipeline.png)

```mermaid
flowchart LR
    A[1. Ingesta y calidad] --> B[2. Forense clásico]
    B --> C[3. OCR + extracción]
    C --> D[4. Reglas y registro]
    D --> E[5. Semántica F6]
    E --> F[6. Modelos F4 + F5]
    F --> G[7. Fusión F7]
    G --> H[8. Explicación y registro]
    H --> I[Evidencias + mapa F5]
    H --> J[SQLite + PDF/JSON]
```

La ruta rápida (calidad + forense + OCR + reglas) está siempre disponible. La
ruta profunda (F4/F5/F7) se usa cuando los checkpoints existen; si faltan, el
sistema muestra las señales base y lo declara.

## 5. Despliegue (3 PCs, un entorno)

```mermaid
flowchart LR
    L[Laptop A] -->|git push| G[(GitHub)]
    C[Laptop B] -->|git pull| G
    G -->|git pull| LAB[PC laboratorio<br/>Ubuntu + GPU]
    LAB -->|rsync artifacts/| L
    C -->|docker compose| D[Una imagen Docker]
    L -->|docker compose| D
    LAB -->|docker compose --profile gpu| D
```

Los datos y artefactos no viajan por git: `data/` y `artifacts/` están
ignorados. Los documentos del CETPRO, si existieran, se sincronizarían aparte.
