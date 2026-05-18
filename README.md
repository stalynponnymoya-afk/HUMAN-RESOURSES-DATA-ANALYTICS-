# Smart Cargo - Talent Retention Analytics

## Descripción / Description

**Español:**
Smart Cargo es un proyecto enfocado en la retención de talento usando análisis de datos y machine learning. El script `smart_cargo.py` realiza limpieza de datos, estandarización de variables categóricas, imputación de valores faltantes y análisis exploratorio inicial para preparar un dataset de recursos humanos en el que se busca predecir la deserción o retención de empleados.

**English:**
Smart Cargo is a project focused on talent retention using data analytics and machine learning. The `smart_cargo.py` script performs data cleaning, categorical variable standardization, missing value imputation, and initial exploratory analysis to prepare a human resources dataset for predicting employee churn.

## Qué hace el script / What the script does

- Carga un archivo Excel llamado `base_proyecto_.xlsx`.
- Muestra información general del dataset y describe sus variables.
- Estandariza columnas categóricas a minúsculas y reemplaza espacios por `_`.
- Identifica y corrige valores nulos en columnas clave como `sueldo`, `area`, y `edad`.
- Imputa sueldos faltantes basándose en cargos específicos (`asesor_comercial`, `auxiliar_de_servicios_en_general`) y por empresa.
- Calcula promedios y modas para realizar imputaciones más inteligentes.
- Ejecuta una fase inicial de exploración de datos para analizar la distribución del target `DESERCION`, el desbalance de clases y los outliers.
- Genera gráficos de análisis exploratorio usando `matplotlib`.

## Dependencias / Dependencies

El script requiere las siguientes librerías de Python:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- shap (opcional)
- openpyxl

## Instrucciones / Instructions

1. Coloca el archivo de datos `base_proyecto_.xlsx` en el mismo directorio o súbelo si estás en Google Colab.
2. Ejecuta el script `smart_cargo.py`.
3. Revisa la salida en consola y las visualizaciones generadas para entender las etapas de limpieza y análisis.

**Nota:** El script está diseñado para ejecutarse en un entorno tipo Jupyter/Google Colab, ya que utiliza `files.upload()` para cargar archivos.

## Archivo de entrada / Input file

- `base_proyecto_.xlsx` - Dataset principal de empleados.

## Estructura esperada del dataset / Expected dataset structure

Se espera que el dataset contenga columnas relevantes de recursos humanos como:

- `cargo`
- `sueldo`
- `empresa_gestion`
- `area`
- `edad`
- `DESERCION`

## Uso rápido / Quick start

```bash
python smart_cargo.py
```

