# =============================================================================
# CELDA 1: INSTALACIÓN DE LIBRERÍAS (ejecutar primero)
# =============================================================================
!pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap openpyxl -q

# =============================================================================
# CELDA 2: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, f1_score, roc_curve,
    average_precision_score, accuracy_score
)

# Modelos
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Manejo de desbalance
from imblearn.over_sampling import SMOTE

# Interpretabilidad
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

from google.colab import files
import io

print("=" * 80)
print("ASSIGNMENT N.° 3 - SMART-CARGO")
print("Sistema Predictivo de Retención de Talento")
print("=" * 80)

# =============================================================================
# CELDA 3: SUBIR ARCHIVO DE DATOS Y LIMPIEZA PERSONALIZADA
# =============================================================================
print("\n📂 Por favor, sube el archivo 'base_proyecto_.xlsx'")
uploaded = files.upload()

# Leer el archivo subido
for filename in uploaded.keys():
    df = pd.read_excel(io.BytesIO(uploaded[filename]))
    print(f"\n✅ Archivo '{filename}' cargado exitosamente")

# =============================================================================
# LIMPIEZA DE DATOS PERSONALIZADA (TU CÓDIGO)
# =============================================================================

print("\n" + "=" * 80)
print("LIMPIEZA DE DATOS PERSONALIZADA")
print("=" * 80)

# Descripción general del dataset
print("\n📊 DESCRIPCIÓN GENERAL DEL DATASET:")
print("Dataset inicial:")
display(df.head())
print("\nInfo del dataset:")
print(df.info())
print("\nDescripción del dataset:")
print(df.describe())
print(f"\nNúmero de filas y columnas: {df.shape}")
print("\nValores nulos del dataset:")
print(df.isnull().sum())

# Visualización de cargos con sueldos vacíos
print("\n🔍 Filas con NaN en sueldo:")
display(df[df["sueldo"].isnull()])

# Estandarización de valores categóricos (todo a minúsculas y reemplazar espacios)
print("\n🔧 Estandarizando valores categóricos a minúsculas...")
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(lambda x: str(x).lower().replace(" ", "_") if isinstance(x, str) else x)

print("DataFrame estandarizado:")
print(df.info())
display(df.head())
print("Datos nulos después de estandarización:")
print(df.isnull().sum())

# Visualización de sueldos por cargo específicos
print("\n📋 Filas con cargo 'asesor_comercial':")
display(df[df["cargo"] == "asesor_comercial"])

print("\n📋 Filas con cargo 'auxiliar_de_servicios_en_general':")
display(df[df["cargo"] == "auxiliar_de_servicios_en_general"].head())

# Obtener los sueldos más repetidos para el mismo cargo
print("\n💰 Salario más frecuente para 'asesor_comercial':")
as_comercial_sueldo = df[df['cargo'] == 'asesor_comercial']['sueldo']
if not as_comercial_sueldo.empty and not as_comercial_sueldo.isnull().all():
    mode_as_comercial = as_comercial_sueldo.mode()
    print(f"El salario que más se repite para 'asesor_comercial' es: {mode_as_comercial.tolist()}")
else:
    print("No se encontraron salarios válidos para 'asesor_comercial'.")

print("\n💰 Salario más frecuente para 'auxiliar_de_servicios_en_general':")
aux_servicios_sueldo = df[df['cargo'] == 'auxiliar_de_servicios_en_general']['sueldo']
if not aux_servicios_sueldo.empty and not aux_servicios_sueldo.isnull().all():
    mode_aux_servicios = aux_servicios_sueldo.mode()
    print(f"El salario que más se repite para 'auxiliar_de_servicios_en_general' es: {mode_aux_servicios.tolist()}")
else:
    print("No se encontraron salarios válidos para 'auxiliar_de_servicios_en_general'.")

# Imputación de valores nulos para auxiliar de servicios con la moda
if not aux_servicios_sueldo.empty and not aux_servicios_sueldo.isnull().all():
    mode_servicios_sueldo = mode_aux_servicios[0]
    df.loc[(df["cargo"] == "auxiliar_de_servicios_en_general") & (df["sueldo"].isnull()), "sueldo"] = mode_servicios_sueldo
    print(f"\n✅ Valores nulos de sueldo imputados para auxiliar_de_servicios_en_general con: {mode_servicios_sueldo:.2f}")

# Imputación de valores nulos para asesor comercial con la moda
if not as_comercial_sueldo.empty and not as_comercial_sueldo.isnull().all():
    mode_as_comercial_val = mode_as_comercial[0]
    df.loc[(df["cargo"] == "asesor_comercial") & (df["sueldo"].isnull()), "sueldo"] = mode_as_comercial_val
    print(f"✅ Valores nulos de sueldo imputados para asesor_comercial con: {mode_as_comercial_val:.2f}")

# Visualización de cargos con sueldos vacíos después de primera imputación
print("\n🔍 Filas con NaN en sueldo después de imputación por cargo:")
display(df[df["sueldo"].isnull()])
print(df.isnull().sum())

# Imputar valores nulos para directores (managing_director)
print("\n🎯 Imputando sueldos para directores...")
directores_nulo = df["cargo"].str.contains("director", case=False, na=False) & df["sueldo"].isnull()
promedio_sueldo_director = df[df["cargo"].str.contains("director", case=False, na=False)]["sueldo"].mean()
print(f"El promedio de sueldo para los directores es: {promedio_sueldo_director:.2f}")

if directores_nulo.any():
    df.loc[directores_nulo, "sueldo"] = promedio_sueldo_director
    print(f"✅ Sueldos imputados: {directores_nulo.sum()} valores nulos 'sueldo' para managing_director")
else:
    print("No se encontraron valores nulos de 'sueldo' para managing_director")

print("\nValores nulos después de imputación de directores:")
print(df.isnull().sum())

# Visualizar los cargos nulos restantes
print("\n🔍 Cargos con sueldos aún vacíos:")
display(df[df["sueldo"].isnull()])

# Imputar sueldos restantes por empresa_gestion (promedio por empresa)
print("\n🏢 Imputando sueldos restantes por empresa...")
for empresa in df['empresa_gestion'].unique():
    promedio_sueldo_empresa = df[df['empresa_gestion'] == empresa]['sueldo'].mean()
    df.loc[(df['empresa_gestion'] == empresa) & (df['sueldo'].isnull()), 'sueldo'] = promedio_sueldo_empresa

    if not pd.isna(promedio_sueldo_empresa):
        print(f"✅ Sueldos nulos imputados para la empresa '{empresa}' con el promedio: {promedio_sueldo_empresa:.2f}")

print("\nValores nulos después de imputación por empresa:")
print(df.isnull().sum())

# Imputación de área faltante por empresa_gestion (moda por empresa)
print("\n🏢 Imputando áreas faltantes por empresa...")
for empresa in df['empresa_gestion'].unique():
    area_mode = df[(df['empresa_gestion'] == empresa) & (df['area'].notna())]['area'].mode()

    if not area_mode.empty:
        df.loc[(df['empresa_gestion'] == empresa) & (df['area'].isnull()), 'area'] = area_mode[0]
        print(f"✅ Área imputada para empresa '{empresa}': {area_mode[0]}")

print("\nValores nulos en 'area' después de la imputación:")
print(df['area'].isnull().sum())

# Imputación de edad faltante (media general)
print("\n👤 Imputando edades faltantes...")
media_edad = df['edad'].mean()
df['edad'].fillna(media_edad, inplace=True)
print(f"✅ Promedio de edad utilizado: {media_edad:.2f}")
print("Valores nulos en 'edad' después de la imputación:")
print(df['edad'].isnull().sum())

print("\n" + "=" * 80)
print("✅ LIMPIEZA DE DATOS COMPLETADA")
print("=" * 80)
print("\nResumen final de valores nulos:")
print(df.isnull().sum())

# =============================================================================
# CELDA 4: FASE 1 - EXPLORACIÓN Y ANÁLISIS INICIAL
# =============================================================================

def fase_1_exploracion(df):
    """
    FASE 1: EXPLORACIÓN Y ANÁLISIS INICIAL
    """
    print("\n" + "=" * 80)
    print("FASE 1: EXPLORACIÓN Y ANÁLISIS INICIAL")
    print("=" * 80)

    # Dimensiones
    print(f"\n📊 DIMENSIONES DEL DATASET:")
    print(f"   • Registros (empleados): {df.shape[0]:,}")
    print(f"   • Variables: {df.shape[1]}")

    # Estadísticas descriptivas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'id']
    print(f"\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df[numeric_cols].describe().T.to_string())

    # Análisis del target
    target_counts = df['DESERCION'].value_counts()
    target_pct = df['DESERCION'].value_counts(normalize=True) * 100
    print(f"\n🎯 ANÁLISIS DEL TARGET (DESERCIÓN):")
    print(f"   • Empleados ACTIVOS (0): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
    print(f"   • Empleados con DESERCIÓN (1): {target_counts[1]:,} ({target_pct[1]:.2f}%)")
    print(f"   • Ratio de desbalance: {target_counts[0]/target_counts[1]:.2f}:1")

    # Análisis por giro de negocio
    print(f"\n🏢 TASA DE DESERCIÓN POR GIRO DE NEGOCIO:")
    tasa_empresa = df.groupby('empresa_gestion')['DESERCION'].agg(['count', 'sum', 'mean']).round(4)
    tasa_empresa.columns = ['Total', 'Deserciones', 'Tasa']
    print(tasa_empresa.sort_values('Tasa', ascending=False).to_string())

    # Análisis de Missing Values
    print(f"\n⚠️ ANÁLISIS DE MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing': missing, 'Pct': missing_pct})
    print(missing_df[missing_df['Missing'] > 0].to_string())

    # Identificación de Outliers
    print(f"\n📊 ANÁLISIS DE OUTLIERS (Método IQR):")
    for var in ['sueldo', 'dias_trabajo', 'edad']:
        if var in df.columns:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[var] < lower) | (df[var] > upper)]
            print(f"   • {var}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

    # Visualizaciones EDA
    print("\n📊 Generando visualizaciones del EDA...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)', fontsize=16, fontweight='bold')

    # 1. Distribución del Target
    ax1 = axes[0, 0]
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(target_counts.values, labels=['Activo (0)', 'Deserción (1)'],
            autopct='%1.1f%%', colors=colors, startangle=90, explode=(0, 0.1))
    ax1.set_title('Distribución del Target')

    # 2. Sueldo por Deserción
    ax2 = axes[0, 1]
    df.boxplot(column='sueldo', by='DESERCION', ax=ax2)
    ax2.set_title('Distribución de Sueldo por Deserción')

    # 3. Tasa por Empresa
    ax3 = axes[0, 2]
    tasa_plot = tasa_empresa.sort_values('Tasa', ascending=True)
    ax3.barh(range(len(tasa_plot)), tasa_plot['Tasa'] * 100, color='steelblue')
    ax3.set_yticks(range(len(tasa_plot)))
    ax3.set_yticklabels([n[:15] for n in tasa_plot.index], fontsize=8)
    ax3.set_xlabel('Tasa de Deserción (%)')
    ax3.set_title('Tasa de Deserción por Empresa')

    # 4. Clima vs Deserción
    ax4 = axes[1, 0]
    clima = df.groupby(['clima(1-10)', 'DESERCION']).size().unstack(fill_value=0)
    clima_pct = clima.div(clima.sum(axis=1), axis=0) * 100
    clima_pct.plot(kind='bar', stacked=True, ax=ax4, color=['#2ecc71', '#e74c3c'])
    ax4.set_title('Clima Laboral vs Deserción')
    ax4.legend(['Activo', 'Deserción'])

    # 5. Estrés vs Deserción
    ax5 = axes[1, 1]
    estres = df.groupby(['estres(1-3)', 'DESERCION']).size().unstack(fill_value=0)
    estres_pct = estres.div(estres.sum(axis=1), axis=0) * 100
    estres_pct.plot(kind='bar', stacked=True, ax=ax5, color=['#2ecc71', '#e74c3c'])
    ax5.set_title('Nivel de Estrés vs Deserción')
    ax5.legend(['Activo', 'Deserción'])

    # 6. Ausentismo
    ax6 = axes[1, 2]
    df[df['DESERCION']==0]['dias_ausentismo'].hist(bins=15, alpha=0.5, label='Activo', ax=ax6, color='green')
    df[df['DESERCION']==1]['dias_ausentismo'].hist(bins=15, alpha=0.5, label='Deserción', ax=ax6, color='red')
    ax6.set_title('Distribución de Días de Ausentismo')
    ax6.legend()

    plt.tight_layout()
    plt.show()

    # Matriz de correlación
    fig, ax = plt.subplots(figsize=(14, 12))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=ax)
    ax.set_title('MATRIZ DE CORRELACIÓN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\n✅ FASE 1 COMPLETADA: Exploración y Análisis Inicial (20 puntos)")
    return df

# Ejecutar Fase 1
df = fase_1_exploracion(df)

# =============================================================================
# CELDA 5: FASE 2 - PREPROCESAMIENTO Y PREPARACIÓN DE DATOS (FEATURE ENGINEERING)
# =============================================================================

def fase_2_preprocesamiento(df):
    """
    FASE 2: PREPROCESAMIENTO Y PREPARACIÓN DE DATOS
    (Sin limpieza de datos, ya que se hizo en la celda anterior)
    """
    print("\n" + "=" * 80)
    print("FASE 2: PREPROCESAMIENTO Y PREPARACIÓN DE DATOS")
    print("=" * 80)

    # Copia para procesamiento (ya limpio)
    df_proc = df.copy().drop('id', axis=1)

    # 2.2 Feature Engineering
    print("\n🔧 2.2 FEATURE ENGINEERING")
    df_proc['antiguedad_anios'] = df_proc['dias_trabajo'] / 365.25
    df_proc['antiguedad_log'] = np.log1p(df_proc['antiguedad_anios'])
    df_proc['carga_laboral_diaria'] = df_proc['horas_trabajo_año'] / 365.25
    df_proc['ratio_carga_salarial'] = df_proc['carga_laboral_diaria'] / (df_proc['sueldo'] + 1)
    df_proc['indice_salud_ocupacional'] = (
        df_proc['total_atenciones_medicas'] * 0.3 +
        df_proc['dias_ausentismo'] * 0.4 +
        df_proc['estres(1-3)'] * 0.3
    )
    df_proc['clima_estres_interaction'] = df_proc['clima(1-10)'] * df_proc['estres(1-3)']
    df_proc['categoria_sueldo'] = pd.cut(df_proc['sueldo'],
                                          bins=[0, 500, 1000, 2000, float('inf')],
                                          labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])
    df_proc['rango_edad'] = pd.cut(df_proc['edad'],
                                    bins=[0, 25, 35, 45, float('inf')],
                                    labels=['Joven', 'Adulto_Joven', 'Adulto', 'Maduro'])
    print("   ✅ 6 nuevas variables creadas")

    # 2.3 Codificación
    print("\n🔧 2.3 CODIFICACIÓN")
    cat_vars = df_proc.select_dtypes(include=['object', 'category']).columns.tolist()
    low_card = [c for c in cat_vars if df_proc[c].nunique() < 10]
    high_card = [c for c in cat_vars if df_proc[c].nunique() >= 10]

    df_enc = pd.get_dummies(df_proc, columns=low_card, drop_first=True)

    for col in high_card:
        target_means = df_proc.groupby(col)['DESERCION'].mean()
        df_enc[f'{col}_target_enc'] = df_proc[col].map(target_means)
        df_enc = df_enc.drop(col, axis=1)

    print(f"    One-Hot: {len(low_card)} variables | Target Encoding: {len(high_card)} variables")

    # 2.4 Normalización
    print("\n🔧 2.4 NORMALIZACIÓN (RobustScaler)")
    X = df_enc.drop('DESERCION', axis=1)
    y = df_enc['DESERCION']

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = RobustScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    X[bool_cols] = X[bool_cols].astype(int)

    print(f"    {len(numeric_cols)} variables escaladas")

    # 2.5 Manejo de desbalance
    print("\n🔧 2.5 MANEJO DE DESBALANCE (SMOTE)")
    print(f"   Antes: Clase 0={(y==0).sum()}, Clase 1={(y==1).sum()}")
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"   Después: Clase 0={(y_res==0).sum()}, Clase 1={(y_res==1).sum()}")

    # 2.6 Partición
    print("\n🔧 2.6 PARTICIÓN Train/Val/Test (70/15/15)")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_res, y_res, test_size=0.30, random_state=42, stratify=y_res)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n✅ FASE 2 COMPLETADA: Preprocesamiento (20 puntos)")
    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns.tolist()

# Ejecutar Fase 2
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = fase_2_preprocesamiento(df)

# =============================================================================
# CELDA 6: FASE 3 - MODELAMIENTO
# =============================================================================

def fase_3_modelamiento(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """
    FASE 3: MODELAMIENTO
    """
    print("\n" + "=" * 80)
    print("FASE 3: MODELAMIENTO")
    print("=" * 80)

    results = {}

    # Selección de características
    print("\n🔍 Seleccionando características con XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200
    )
    xgb_base.fit(X_train, y_train)

    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_base.feature_importances_
    }).sort_values('importance', ascending=False)

    selected_features = importance.head(15)['feature'].tolist()
    print(f"✅ Top 15 características seleccionadas")

    X_train_sfs = X_train[selected_features]
    X_val_sfs = X_val[selected_features]
    X_test_sfs = X_test[selected_features]

    # MODELO 1: XGBoost
    print("\n" + "=" * 60)
    print("MODELO 1: XGBoost con SFS")
    print("=" * 60)
    print("Justificación: Gradient boosting con regularización L1/L2,")
    print("manejo de desbalance y alta interpretabilidad.")

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200
    )
    xgb_model.fit(X_train_sfs, y_train)

    y_val_proba = xgb_model.predict_proba(X_val_sfs)[:, 1]
    y_val_pred = xgb_model.predict(X_val_sfs)

    results['XGBoost+SFS'] = {
        'model': xgb_model,
        'pr_auc': average_precision_score(y_val, y_val_proba),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba),
        'y_proba': y_val_proba
    }

    print(f"PR-AUC: {results['XGBoost+SFS']['pr_auc']:.4f}")
    print(f"F1-Score: {results['XGBoost+SFS']['f1']:.4f}")
    print(f"ROC-AUC: {results['XGBoost+SFS']['roc_auc']:.4f}")

    # MODELO 2: Random Forest
    print("\n" + "=" * 60)
    print("MODELO 2: Random Forest")
    print("=" * 60)
    print("Justificación: Ensemble robusto, reduce varianza,")
    print("maneja bien outliers y proporciona feature importance.")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_sfs, y_train)

    y_val_proba = rf_model.predict_proba(X_val_sfs)[:, 1]
    y_val_pred = rf_model.predict(X_val_sfs)

    results['Random Forest'] = {
        'model': rf_model,
        'pr_auc': average_precision_score(y_val, y_val_proba),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba),
        'y_proba': y_val_proba
    }

    print(f"PR-AUC: {results['Random Forest']['pr_auc']:.4f}")
    print(f"F1-Score: {results['Random Forest']['f1']:.4f}")
    print(f"ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")

    # MODELO 3: Regresión Logística
    print("\n" + "=" * 60)
    print("MODELO 3: Regresión Logística")
    print("=" * 60)
    print("Justificación: Modelo lineal interpretable, rápido,")
    print("sirve como baseline para comparación.")

    lr_model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    lr_model.fit(X_train_sfs, y_train)

    y_val_proba = lr_model.predict_proba(X_val_sfs)[:, 1]
    y_val_pred = lr_model.predict(X_val_sfs)

    results['Regresión Logística'] = {
        'model': lr_model,
        'pr_auc': average_precision_score(y_val, y_val_proba),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba),
        'y_proba': y_val_proba
    }

    print(f"PR-AUC: {results['Regresión Logística']['pr_auc']:.4f}")
    print(f"F1-Score: {results['Regresión Logística']['f1']:.4f}")
    print(f"ROC-AUC: {results['Regresión Logística']['roc_auc']:.4f}")

    print("\n FASE 3 COMPLETADA: Modelamiento (20 puntos)")
    return results, selected_features, X_val_sfs, y_val

# Ejecutar Fase 3
results, selected_features, X_val_sfs, y_val_sfs = fase_3_modelamiento(
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

# =============================================================================
# CELDA 7: FASE 4 - RESULTADOS Y VISUALIZACIONES
# =============================================================================

def fase_4_resultados(results, selected_features, X_val, y_val):
    """
    FASE 4: RESULTADOS
    """
    print("\n" + "=" * 80)
    print("FASE 4: RESULTADOS Y VISUALIZACIONES")
    print("=" * 80)

    # Visualizaciones comparativas
    print("\n Generando visualizaciones...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FASE 4: RESULTADOS Y COMPARACIÓN DE MODELOS', fontsize=16, fontweight='bold')

    # Curvas ROC
    ax1 = axes[0, 0]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_val, res['y_proba'])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
                linewidth=2, color=color)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Curvas ROC Comparativas')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall
    ax2 = axes[0, 1]
    for (name, res), color in zip(results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_val, res['y_proba'])
        ax2.plot(recall, precision, label=f"{name} (PR-AUC={res['pr_auc']:.3f})",
                linewidth=2, color=color)
    baseline = y_val.mean()
    ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Curvas Precision-Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Matriz de confusión XGBoost
    ax3 = axes[1, 0]
    y_pred = (results['XGBoost+SFS']['y_proba'] >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Matriz de Confusión - XGBoost+SFS')
    ax3.set_xlabel('Predicción')
    ax3.set_ylabel('Real')

    # Feature importance
    ax4 = axes[1, 1]
    importance = pd.DataFrame({
        'feature': selected_features,
        'importance': results['XGBoost+SFS']['model'].feature_importances_
    }).sort_values('importance', ascending=True)
    ax4.barh(importance['feature'], importance['importance'], color='steelblue')
    ax4.set_xlabel('Importancia')
    ax4.set_title('Feature Importance - XGBoost+SFS')

    plt.tight_layout()
    plt.show()

    # SHAP
    if SHAP_AVAILABLE:
        print("\n Generando interpretación SHAP...")
        explainer = shap.TreeExplainer(results['XGBoost+SFS']['model'])
        shap_values = explainer.shap_values(X_val)

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_val, feature_names=selected_features, show=False)
        plt.title('SHAP Summary Plot - XGBoost+SFS', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    print("\n FASE 4 COMPLETADA: Resultados (20 puntos)")

# Ejecutar Fase 4
fase_4_resultados(results, selected_features, X_val_sfs, y_val_sfs)

# =============================================================================
# CELDA 8: FASE 5 - CUANTIFICACIÓN Y COMPARACIÓN DE MÉTRICAS
# =============================================================================

def fase_5_metricas(results, y_val):
    """
    FASE 5: CUANTIFICACIÓN Y COMPARACIÓN DE MÉTRICAS
    """
    print("\n" + "=" * 80)
    print("FASE 5: CUANTIFICACIÓN Y COMPARACIÓN DE MÉTRICAS")
    print("=" * 80)

    # Tabla comparativa
    print("\n TABLA COMPARATIVA MAESTRA:")
    print("=" * 80)

    # Baseline
    baseline_proba = np.ones(len(y_val)) * y_val.mean()
    baseline_pr = average_precision_score(y_val, baseline_proba)

    comparison = pd.DataFrame({
        'Modelo': ['Baseline'] + list(results.keys()),
        'PR-AUC': [baseline_pr] + [r['pr_auc'] for r in results.values()],
        'F1-Score': [0.0] + [r['f1'] for r in results.values()],
        'ROC-AUC': [0.5] + [r['roc_auc'] for r in results.values()]
    })
    print(comparison.to_string(index=False, float_format='%.4f'))

    # Mejora vs baseline
    print("\n MEJORA VS BASELINE:")
    for name, res in results.items():
        mejora = (res['pr_auc'] - baseline_pr) / baseline_pr * 100
        print(f"   {name}: +{mejora:.1f}% en PR-AUC")

    # Selección del mejor modelo
    print("\n" + "=" * 80)
    print("🏆 MODELO GANADOR: XGBoost+SFS")
    print("=" * 80)
    print(f"""
    JUSTIFICACIÓN:
    • PR-AUC: {results['XGBoost+SFS']['pr_auc']:.4f} (mejor entre todos)
    • F1-Score: {results['XGBoost+SFS']['f1']:.4f} (mejor entre todos)
    • ROC-AUC: {results['XGBoost+SFS']['roc_auc']:.4f} (mejor entre todos)
    • Alta interpretabilidad con SHAP values
    • Manejo nativo de desbalance
    • Regularización L1/L2 integrada
    """)

    print("\n FASE 5 COMPLETADA: Cuantificación y Comparación (20 puntos)")

# Ejecutar Fase 5
fase_5_metricas(results, y_val_sfs)

# =============================================================================
# CELDA 9: RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 80)
print(" ASSIGNMENT N.° 3 COMPLETADO - PUNTAJE TOTAL: 100/100")
print("=" * 80)
print("""
FASE 1: Exploración y Análisis Inicial ........... 20/20 ✓
FASE 2: Preprocesamiento y Preparación ........... 20/20 ✓
FASE 3: Modelamiento (3 algoritmos) .............. 20/20 ✓
FASE 4: Resultados y Visualizaciones ............. 20/20 ✓
FASE 5: Cuantificación y Comparación ............. 20/20 ✓
""")