import pandas as pd

# Leer el archivo Excel
df = pd.read_excel('Ventas+Videojuegos.xlsx')

# Mostrar las primeras 5 filas
print("Primeras 5 filas del archivo:")
print(df.head())

# Mostrar información general
print("\nInformación del dataset:")
print(df.info())

# Mostrar las columnas disponibles
print("\nColumnas disponibles:")
print(df.columns.tolist())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN INICIAL
# ============================================
print("🎮 INICIANDO ANÁLISIS DE VIDEOJUEGOS")
print("="*70)

# Crear carpeta para visualizaciones
if not os.path.exists('visualizaciones'):
    os.makedirs('visualizaciones')
    print("✅ Carpeta 'visualizaciones' creada")

# ============================================
# 1. CARGAR DATOS
# ============================================
print("\n📂 CARGANDO DATOS...")

# CAMBIO IMPORTANTE: Leer desde archivo local
df = pd.read_excel('Ventas+Videojuegos.xlsx')
print(f"✅ Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")

# ============================================
# 2. EXPLORACIÓN INICIAL
# ============================================
print("\n📊 EXPLORACIÓN INICIAL:")
print(f"Forma del dataset: {df.shape}")
print(f"\nColumnas: {df.columns.tolist()}")
print(f"\nValores nulos:\n{df.isnull().sum()}")

# ============================================
# 3. LIMPIEZA DE DATOS
# ============================================
print("\n🧹 LIMPIANDO DATOS...")

# Rellenar editoriales nulas
df['Editorial'] = df['Editorial'].fillna('No especificado')
print(f"✅ Editoriales 'No especificado': {(df['Editorial'] == 'No especificado').sum()}")

# Convertir columnas de ventas a float
columnas_ventas = ['Ventas NA', 'Ventas EU', 'Ventas JP', 'Ventas Otros', 'Ventas Global']
for col in columnas_ventas:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Limpiar columna Año
df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
df = df.dropna(subset=['Año'])
df['Año'] = df['Año'].astype(int)

print(f"✅ Dataset limpio: {df.shape[0]:,} registros")

# ============================================
# 4. CREAR VARIABLE DE ERA
# ============================================
df['Era'] = pd.cut(df['Año'],
                   bins=[0, 1999, 2005, 2010, 2020],
                   labels=['Pre-2000 (Retro)', '2000-2005 (PS2/Xbox)',
                          '2006-2010 (Wii/PS3/X360)', '2011-2016 (PS4/XOne)'])

print("\n🎮 DISTRIBUCIÓN POR ERA:")
print(df['Era'].value_counts().sort_index())

# ============================================
# 5. ANÁLISIS EXPLORATORIO
# ============================================
print("\n" + "="*70)
print("📊 ANÁLISIS EXPLORATORIO")
print("="*70)

# Top géneros
print("\n🏆 TOP 5 GÉNEROS:")
top_generos = df.groupby('Genero')['Ventas Global'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False).head(5)
for i, (genero, row) in enumerate(top_generos.iterrows(), 1):
    print(f"   {i}. {genero}: {row['sum']:.0f}M totales | {row['mean']:.2f}M promedio")

# Top plataformas
print("\n🕹️ TOP 5 PLATAFORMAS:")
top_plataformas = df.groupby('Plataforma')['Ventas Global'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False).head(5)
for i, (plat, row) in enumerate(top_plataformas.iterrows(), 1):
    print(f"   {i}. {plat}: {row['sum']:.0f}M totales | {row['mean']:.2f}M promedio")

# ============================================
# 6. VISUALIZACIONES
# ============================================
print("\n🎨 Generando visualizaciones...")
sns.set_style("whitegrid")

# VIZ 1: Top 10 Géneros
plt.figure(figsize=(12, 7))
generos_ventas = df.groupby('Genero')['Ventas Global'].sum().sort_values(ascending=False).head(10)
plt.barh(range(len(generos_ventas)), generos_ventas.values, color=sns.color_palette("viridis", len(generos_ventas)))
plt.yticks(range(len(generos_ventas)), generos_ventas.index)
plt.xlabel('Ventas Totales (Millones)', fontweight='bold')
plt.title(f'Top 10 Géneros por Ventas Globales ({df["Año"].min()}-{df["Año"].max()})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizaciones/01_top_generos.png', dpi=300, bbox_inches='tight')
print("   ✅ 01_top_generos.png")
plt.close()

# VIZ 2: Distribución de Ventas
plt.figure(figsize=(12, 7))
ventas_filtradas = df[df['Ventas Global'] < 5]['Ventas Global']
plt.hist(ventas_filtradas, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
plt.axvline(df['Ventas Global'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Media: {df["Ventas Global"].mean():.2f}M')
plt.axvline(df['Ventas Global'].median(), color='green', linestyle='--', linewidth=2,
            label=f'Mediana: {df["Ventas Global"].median():.2f}M')
plt.xlabel('Ventas Globales (Millones)', fontweight='bold')
plt.ylabel('Frecuencia', fontweight='bold')
plt.title('Distribución de Ventas', fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('visualizaciones/02_distribucion.png', dpi=300, bbox_inches='tight')
print("   ✅ 02_distribucion.png")
plt.close()

# VIZ 3: Ventas por Región
plt.figure(figsize=(10, 7))
ventas_regionales = {
    'Norteamérica': df['Ventas NA'].sum(),
    'Europa': df['Ventas EU'].sum(),
    'Japón': df['Ventas JP'].sum(),
    'Otros': df['Ventas Otros'].sum()
}
plt.pie(ventas_regionales.values(), labels=ventas_regionales.keys(), autopct='%1.1f%%',
        startangle=90, colors=['#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3'], explode=(0.05, 0, 0, 0))
plt.title('Distribución Geográfica de Ventas', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizaciones/03_regiones.png', dpi=300, bbox_inches='tight')
print("   ✅ 03_regiones.png")
plt.close()

# VIZ 4: Timeline
plt.figure(figsize=(14, 7))
ventas_por_anio = df.groupby('Año')['Ventas Global'].sum().sort_index()
plt.plot(ventas_por_anio.index, ventas_por_anio.values, linewidth=2.5, color='#2ecc71', marker='o', markersize=4)
plt.fill_between(ventas_por_anio.index, ventas_por_anio.values, alpha=0.3, color='#2ecc71')
año_peak = ventas_por_anio.idxmax()
ventas_peak = ventas_por_anio.max()
plt.axvline(año_peak, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.xlabel('Año', fontweight='bold')
plt.ylabel('Ventas Globales (Millones)', fontweight='bold')
plt.title('Evolución de Ventas (1980-2016)', fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizaciones/04_timeline.png', dpi=300, bbox_inches='tight')
print("   ✅ 04_timeline.png")
plt.close()

# ============================================
# 7. MODELO DE MACHINE LEARNING
# ============================================
print("\n" + "="*70)
print("🤖 ENTRENANDO MODELO DE MACHINE LEARNING")
print("="*70)

# Preparar datos para el modelo
df_modelo = df[df['Ventas Global'] > 0].copy()
print(f"✅ Datos para modelado: {len(df_modelo):,} registros")

# Codificar variables categóricas
le_genero = LabelEncoder()
le_plataforma = LabelEncoder()
le_editorial = LabelEncoder()

df_modelo['Genero_enc'] = le_genero.fit_transform(df_modelo['Genero'])
df_modelo['Plataforma_enc'] = le_plataforma.fit_transform(df_modelo['Plataforma'])
df_modelo['Editorial_enc'] = le_editorial.fit_transform(df_modelo['Editorial'])

# Features y target
X = df_modelo[['Genero_enc', 'Plataforma_enc', 'Editorial_enc', 'Año']]
y = df_modelo['Ventas Global']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

# Evaluar modelo
y_pred = modelo.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 MÉTRICAS DEL MODELO:")
print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"   RMSE: {rmse:.3f} millones")
print(f"   MAE: {mae:.3f} millones")

# Importancia de variables
importancias = pd.DataFrame({
    'Variable': ['Género', 'Plataforma', 'Editorial', 'Año'],
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)

print(f"\n🔍 IMPORTANCIA DE VARIABLES:")
print(importancias.to_string(index=False))

# Visualizar importancia
plt.figure(figsize=(10, 6))
colors = sns.color_palette("mako", len(importancias))
bars = plt.barh(importancias['Variable'], importancias['Importancia'], color=colors)
plt.xlabel('Importancia', fontweight='bold')
plt.title('Importancia de Variables en el Modelo', fontsize=16, fontweight='bold')
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
             ha='left', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizaciones/05_importancia.png', dpi=300, bbox_inches='tight')
print("   ✅ 05_importancia.png")
plt.close()

# ============================================
# 8. GUARDAR MODELO Y ENCODERS
# ============================================
print("\n💾 Guardando modelo y encoders...")

# Guardar en un diccionario
modelo_completo = {
    'modelo': modelo,
    'le_genero': le_genero,
    'le_plataforma': le_plataforma,
    'le_editorial': le_editorial,
    'metricas': {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }
}

with open('modelo_entrenado.pkl', 'wb') as f:
    pickle.dump(modelo_completo, f)

print("✅ modelo_entrenado.pkl guardado")

# ============================================
# 9. CARGAR A MONGODB (OPCIONAL)
# ============================================
print("\n" + "="*70)
respuesta = input("¿Deseas cargar los datos a MongoDB? (s/n): ")

if respuesta.lower() == 's':
    try:
        from pymongo import MongoClient
        
        connection_string = "mongodb+srv://alexxatarea_db_user:3ZpKwkOmd7SrNh4c@cluster0.xgqbzmr.mongodb.net/?appName=Cluster0"
        client = MongoClient(connection_string)
        client.admin.command('ping')
        print("✅ Conexión exitosa a MongoDB Atlas")
        
        db = client['video_games']
        collection = db['ProyectoFinal']
        
        collection.delete_many({})
        datos = df_modelo.to_dict('records')
        result = collection.insert_many(datos)
        
        print(f"✅ {len(result.inserted_ids)} documentos insertados en MongoDB")
        
    except Exception as e:
        print(f"❌ Error con MongoDB: {e}")
else:
    print("⏭️ MongoDB omitido")

# ============================================
# 10. RESUMEN FINAL
# ============================================
print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO")
print("="*70)
print("\n📁 ARCHIVOS GENERADOS:")
print("   - visualizaciones/01_top_generos.png")
print("   - visualizaciones/02_distribucion.png")
print("   - visualizaciones/03_regiones.png")
print("   - visualizaciones/04_timeline.png")
print("   - visualizaciones/05_importancia.png")
print("   - modelo_entrenado.pkl")

print(f"\n🎯 HALLAZGOS CLAVE:")
print(f"   - Período analizado: {df['Año'].min()}-{df['Año'].max()}")
print(f"   - Total juegos: {len(df):,}")
print(f"   - Género más exitoso: {top_generos.index[0]}")
print(f"   - Plataforma líder: {top_plataformas.index[0]}")
print(f"   - Año pico: {año_peak} ({ventas_peak:.0f}M ventas)")
print(f"   - Factor más importante: {importancias.iloc[0]['Variable']} ({importancias.iloc[0]['Importancia']:.1%})")

print("\n🚀 Siguiente paso: Ejecutar 'streamlit run app_streamlit.py'")
print("="*70)