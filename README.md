# Smart Cargo — Predicción de Deserción de Talento (HR Attrition Prediction)

Sistema de machine learning que anticipa qué empleados tienen mayor probabilidad
de abandonar la empresa, para que el área de Recursos Humanos pueda intervenir
antes de perder el talento.

---

## 1. Contexto de Negocio

La rotación de personal es uno de los costos más silenciosos de una organización.
Cada salida arrastra:

- **Costo de reemplazo** — reclutamiento, selección y onboarding de un perfil nuevo.
- **Pérdida de productividad** — meses de curva de aprendizaje hasta recuperar el
  rendimiento del empleado que se fue.
- **Impacto en el equipo** — sobrecarga y desmotivación de quienes se quedan.
- **Riesgo de fuga en cascada** — la salida de perfiles clave arrastra a otros.

Detectar la rotación *después* de que ocurre solo permite medir el daño. El valor
real está en **anticiparla**: identificar de forma temprana a los empleados en
riesgo y actuar con planes de retención focalizados (revisión salarial, ajuste de
carga, intervención sobre clima y estrés).

## 2. Solución Predictiva

El proyecto construye un modelo de clasificación binaria que estima la
probabilidad de deserción de cada empleado (`DESERCION = 1`).

- **Datos:** registros de empleados con variables de cargo, sueldo, antigüedad,
  clima laboral, estrés, ausentismo y atenciones médicas.
- **Enfoque:** limpieza e imputación guiada por los datos, ingeniería de
  variables, balanceo con SMOTE y entrenamiento de tres modelos comparables.
- **Modelos:** XGBoost, Random Forest y Regresión Logística.
- **Métrica principal:** **PR-AUC**, la adecuada ante un target fuertemente
  desbalanceado. Se reportan además F1-Score y ROC-AUC.
- **Interpretabilidad:** valores **SHAP** para explicar qué variables empujan el
  riesgo de cada predicción, requisito indispensable para que Recursos Humanos
  confíe en la herramienta.

El pipeline es **modular y reproducible**: la limpieza, la ingeniería de
variables, el entrenamiento y la evaluación viven en módulos independientes bajo
`src/`, y no hay dependencias de entornos cloud.

## 3. Estructura del Repositorio

```
hr-attrition-prediction/
├── data/
│   ├── raw/                     # Datos originales (no versionados)
│   │   └── base_proyecto_.xlsx
│   └── processed/               # Datos limpios generados por el pipeline
├── notebooks/
│   └── eda.ipynb                # Fase 1: análisis exploratorio y visualizaciones
├── src/
│   ├── __init__.py
│   ├── config.py                # Rutas, parámetros y nombres de columnas
│   ├── data_processing.py       # Carga, estandarización e imputación
│   ├── feature_engineering.py   # Fase 2: features, encoding, escalado, split
│   ├── train_model.py           # Fase 3: XGBoost, Random Forest, Reg. Logística
│   ├── evaluate.py              # Fases 4 y 5: métricas, gráficos y SHAP
│   └── main.py                  # Orquestador end-to-end
├── models/                      # Artefactos de modelos entrenados (no versionados)
├── reports/
│   └── figures/                 # Gráficos exportados (no versionados)
├── .gitignore
├── requirements.txt
└── README.md
```

**Nota sobre datos:** `data/` está excluido vía `.gitignore` para evitar subir
información de empleados (posible PII) a un repositorio público. Coloca el
archivo original en `data/raw/` antes de ejecutar.

## 4. Instalación y Uso Local

### Requisitos

- Python 3.10 o superior
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/hr-attrition-prediction.git
cd hr-attrition-prediction

# 2. Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar el dataset original
#    Copia el archivo en: data/raw/base_proyecto_.xlsx

# 5. Ejecutar el pipeline completo
python -m src.main
```

Opciones disponibles:

```bash
python -m src.main --input data/raw/otro_archivo.xlsx   # ruta de entrada explícita
python -m src.main --output data/processed/clean.csv    # ruta de salida explícita
python -m src.main --no-save                            # no persistir datos limpios
```

### Análisis exploratorio

El notebook de la Fase 1 es autocontenido y consume los módulos de `src/`:

```bash
jupyter notebook notebooks/eda.ipynb
```

### Salidas

- `data/processed/` — dataset limpio listo para modelar.
- `reports/figures/` — curvas ROC y Precision-Recall, matriz de confusión,
  importancias de variables y resumen SHAP.
- Consola — tabla comparativa maestra de métricas y modelo ganador.

## 5. Fases del Proyecto

| Fase | Contenido | Ubicación |
|------|-----------|-----------|
| 1 | Exploración y análisis inicial | `notebooks/eda.ipynb` |
| 2 | Feature engineering y preprocesamiento | `src/feature_engineering.py` |
| 3 | Modelamiento (3 algoritmos) | `src/train_model.py` |
| 4 | Resultados y visualizaciones | `src/evaluate.py` |
| 5 | Cuantificación y comparación | `src/evaluate.py` |

---

Proyecto de portafolio de analítica de datos aplicada a Recursos Humanos.
