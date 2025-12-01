import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN
# ============================================
print("🎮 ANÁLISIS AVANZADO DE VIDEOJUEGOS CON MACHINE LEARNING")
print("="*80)

# Crear carpeta para visualizaciones
if not os.path.exists('visualizaciones'):
    os.makedirs('visualizaciones')

# ============================================
# 1. CARGAR Y LIMPIAR DATOS
# ============================================
print("\n📂 CARGANDO DATOS...")
df = pd.read_excel('Ventas+Videojuegos.xlsx')
print(f"✅ Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")

# Limpieza
df['Editorial'] = df['Editorial'].fillna('No especificado')
columnas_ventas = ['Ventas NA', 'Ventas EU', 'Ventas JP', 'Ventas Otros', 'Ventas Global']
for col in columnas_ventas:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
df = df.dropna(subset=['Año'])
df['Año'] = df['Año'].astype(int)

# Crear eras
df['Era'] = pd.cut(df['Año'],
                   bins=[0, 1999, 2005, 2010, 2020],
                   labels=['Pre-2000 (Retro)', '2000-2005 (PS2/Xbox)',
                          '2006-2010 (Wii/PS3/X360)', '2011-2016 (PS4/XOne)'])

print(f"✅ Dataset limpio: {len(df):,} registros")

# ============================================
# 2. CREAR CATEGORÍAS DE ÉXITO (CLASIFICACIÓN)
# ============================================
print("\n" + "="*80)
print("🎯 CREANDO MODELO DE CLASIFICACIÓN DE ÉXITO")
print("="*80)

# Definir categorías basadas en ventas
def categorizar_exito(ventas):
    if ventas < 0.5:
        return 'Fracaso'
    elif ventas < 1.0:
        return 'Moderado'
    elif ventas < 3.0:
        return 'Éxito'
    else:
        return 'Blockbuster'

df['Categoria_Exito'] = df['Ventas Global'].apply(categorizar_exito)

# Mostrar distribución
print("\n📊 DISTRIBUCIÓN DE CATEGORÍAS DE ÉXITO:")
distribucion = df['Categoria_Exito'].value_counts()
for cat, count in distribucion.items():
    porcentaje = (count / len(df)) * 100
    print(f"   {cat:12} {count:6,} juegos ({porcentaje:5.1f}%)")

# Visualización de categorías
plt.figure(figsize=(12, 6))
colors = {'Fracaso': '#e74c3c', 'Moderado': '#f39c12', 'Éxito': '#3498db', 'Blockbuster': '#2ecc71'}
counts = df['Categoria_Exito'].value_counts()
plt.bar(counts.index, counts.values, color=[colors[cat] for cat in counts.index], edgecolor='black', linewidth=1.5)
plt.xlabel('Categoría de Éxito', fontweight='bold', fontsize=12)
plt.ylabel('Cantidad de Juegos', fontweight='bold', fontsize=12)
plt.title('Distribución de Juegos por Categoría de Éxito\n(Fracaso <0.5M | Moderado 0.5-1M | Éxito 1-3M | Blockbuster >3M)', 
          fontsize=14, fontweight='bold', pad=20)
for i, (cat, val) in enumerate(counts.items()):
    plt.text(i, val, f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizaciones/01_distribucion_categorias.png', dpi=300, bbox_inches='tight')
print("   ✅ 01_distribucion_categorias.png")
plt.close()

# ============================================
# 3. PREPARAR DATOS PARA CLASIFICACIÓN
# ============================================
print("\n🔧 PREPARANDO MODELO DE CLASIFICACIÓN...")

# Filtrar datos
df_modelo = df[df['Ventas Global'] > 0].copy()

# Codificar variables
le_genero = LabelEncoder()
le_plataforma = LabelEncoder()
le_editorial = LabelEncoder()
le_categoria = LabelEncoder()

df_modelo['Genero_enc'] = le_genero.fit_transform(df_modelo['Genero'].astype(str))
df_modelo['Plataforma_enc'] = le_plataforma.fit_transform(df_modelo['Plataforma'].astype(str))
df_modelo['Editorial_enc'] = le_editorial.fit_transform(df_modelo['Editorial'].astype(str))
df_modelo['Categoria_enc'] = le_categoria.fit_transform(df_modelo['Categoria_Exito'])

# Features y target
X = df_modelo[['Genero_enc', 'Plataforma_enc', 'Editorial_enc', 'Año']]
y = df_modelo['Categoria_enc']

from imblearn.over_sampling import SMOTE

# Después de crear X e y, ANTES del split:
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Ahora usa X_balanced y y_balanced en el split:
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.25, random_state=42
)
# Entrenar modelo
clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluar
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 RESULTADOS DEL MODELO DE CLASIFICACIÓN:")
print(f"   Precisión (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Cross-validation (5-fold): {cross_val_score(clf, X, y, cv=5).mean():.4f}")

# Reporte detallado
print("\n📋 REPORTE DE CLASIFICACIÓN POR CATEGORÍA:")
categorias = le_categoria.inverse_transform([0, 1, 2, 3])
print(classification_report(y_test, y_pred, target_names=categorias, zero_division=0))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categorias, yticklabels=categorias, 
            cbar_kws={'label': 'Cantidad'}, linewidths=1, linecolor='black')
plt.xlabel('Predicción', fontweight='bold', fontsize=12)
plt.ylabel('Real', fontweight='bold', fontsize=12)
plt.title('Matriz de Confusión - Modelo de Clasificación\n(Muestra qué tan bien predice cada categoría)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visualizaciones/02_matriz_confusion.png', dpi=300, bbox_inches='tight')
print("   ✅ 02_matriz_confusion.png")
plt.close()

# Importancia de variables
importancias_clf = pd.DataFrame({
    'Variable': ['Género', 'Plataforma', 'Editorial', 'Año'],
    'Importancia': clf.feature_importances_
}).sort_values('Importancia', ascending=False)

print(f"\n🔍 FACTORES QUE DETERMINAN EL ÉXITO:")
print(importancias_clf.to_string(index=False))

plt.figure(figsize=(10, 6))
colors_imp = sns.color_palette("viridis", len(importancias_clf))
bars = plt.barh(importancias_clf['Variable'], importancias_clf['Importancia'], color=colors_imp, edgecolor='black')
plt.xlabel('Importancia Relativa', fontweight='bold', fontsize=12)
plt.title('Factores que Determinan el Éxito de un Videojuego\n(Modelo Random Forest - Mayor valor = Mayor impacto)', 
          fontsize=14, fontweight='bold', pad=20)
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
             ha='left', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizaciones/03_importancia_variables.png', dpi=300, bbox_inches='tight')
print("   ✅ 03_importancia_variables.png")
plt.close()

# ============================================
# 4. CLUSTERING DE JUEGOS SIMILARES
# ============================================
print("\n" + "="*80)
print("🔬 ANÁLISIS DE CLUSTERING (Agrupación de Juegos Similares)")
print("="*80)

# Preparar datos para clustering
df_cluster = df_modelo[['Genero_enc', 'Plataforma_enc', 'Ventas NA', 'Ventas EU', 'Ventas JP']].copy()

# Normalizar
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)

# Método del codo para encontrar K óptimo
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_cluster_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='#e74c3c')
plt.xlabel('Número de Clusters (K)', fontweight='bold', fontsize=12)
plt.ylabel('Inercia (Suma de distancias al cuadrado)', fontweight='bold', fontsize=12)
plt.title('Método del Codo para Determinar K Óptimo\n(El "codo" indica el número ideal de grupos)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(K_range)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizaciones/04_metodo_codo.png', dpi=300, bbox_inches='tight')
print("   ✅ 04_metodo_codo.png")
plt.close()

# Aplicar K-Means con K=5
K_optimo = 5
kmeans = KMeans(n_clusters=K_optimo, random_state=42, n_init=10)
df_modelo['Cluster'] = kmeans.fit_predict(df_cluster_scaled)

print(f"\n📊 CARACTERÍSTICAS DE CADA CLUSTER (K={K_optimo}):")
print("\nEXPLICACIÓN: Los clusters agrupan juegos con características similares")
print("=" * 80)

for cluster_id in range(K_optimo):
    df_cluster_info = df_modelo[df_modelo['Cluster'] == cluster_id]
    print(f"\n🎮 CLUSTER {cluster_id + 1} ({len(df_cluster_info):,} juegos):")
    print(f"   Género predominante: {df_cluster_info['Genero'].mode()[0]}")
    print(f"   Plataforma más común: {df_cluster_info['Plataforma'].mode()[0]}")
    print(f"   Ventas promedio: {df_cluster_info['Ventas Global'].mean():.2f}M")
    print(f"   Región fuerte: ", end='')
    max_region = df_cluster_info[['Ventas NA', 'Ventas EU', 'Ventas JP']].mean().idxmax()
    print(f"{'Norteamérica' if max_region == 'Ventas NA' else 'Europa' if max_region == 'Ventas EU' else 'Japón'}")
    
    # Top 3 juegos representativos
    top_juegos = df_cluster_info.nlargest(3, 'Ventas Global')['Nombre'].tolist()
    print(f"   Juegos representativos:")
    for i, juego in enumerate(top_juegos, 1):
        print(f"      {i}. {juego[:50]}...")

# Visualización de clusters
plt.figure(figsize=(14, 8))
scatter = plt.scatter(df_modelo['Ventas NA'], df_modelo['Ventas EU'], 
                     c=df_modelo['Cluster'], cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.xlabel('Ventas Norteamérica (Millones)', fontweight='bold', fontsize=12)
plt.ylabel('Ventas Europa (Millones)', fontweight='bold', fontsize=12)
plt.title('Clustering de Videojuegos: Ventas NA vs EU\n(Cada color representa un grupo de juegos con características similares)', 
          fontsize=14, fontweight='bold', pad=20)
plt.colorbar(scatter, label='Cluster ID', ticks=range(K_optimo))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizaciones/05_clusters_visualizacion.png', dpi=300, bbox_inches='tight')
print("\n   ✅ 05_clusters_visualizacion.png")
plt.close()

# ============================================
# 5. ANÁLISIS DE SERIES DE TIEMPO
# ============================================
print("\n" + "="*80)
print("📅 ANÁLISIS DE SERIES DE TIEMPO")
print("="*80)
print("\nEXPLICACIÓN: Estudia cómo han evolucionado las ventas y el mercado a lo largo del tiempo")
print("=" * 80)

# Ventas por año
ventas_anuales = df.groupby('Año').agg({
    'Ventas Global': 'sum',
    'Nombre': 'count'
}).rename(columns={'Nombre': 'Cantidad_Juegos'})

# Ventas por región y año
ventas_regionales_tiempo = df.groupby('Año')[['Ventas NA', 'Ventas EU', 'Ventas JP', 'Ventas Otros']].sum()

# Encontrar pico y tendencias
año_peak = ventas_anuales['Ventas Global'].idxmax()
ventas_peak = ventas_anuales['Ventas Global'].max()

print(f"\n📈 HALLAZGOS TEMPORALES:")
print(f"   Año de mayor actividad: {año_peak} ({ventas_peak:.0f}M ventas)")
print(f"   Crecimiento promedio anual (2000-2008): {ventas_anuales.loc[2000:2008, 'Ventas Global'].pct_change().mean()*100:.1f}%")
print(f"   Juegos lanzados en peak: {ventas_anuales.loc[año_peak, 'Cantidad_Juegos']:.0f}")

# Gráfica de evolución temporal
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Ventas totales
ax1.plot(ventas_anuales.index, ventas_anuales['Ventas Global'], linewidth=3, color='#2ecc71', marker='o', markersize=5)
ax1.fill_between(ventas_anuales.index, ventas_anuales['Ventas Global'], alpha=0.3, color='#2ecc71')
ax1.axvline(año_peak, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(año_peak, ventas_peak*1.05, f'Peak: {año_peak}\n({ventas_peak:.0f}M)', 
         ha='center', fontweight='bold', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='red'))
ax1.set_xlabel('Año', fontweight='bold', fontsize=12)
ax1.set_ylabel('Ventas Globales (Millones)', fontweight='bold', fontsize=12)
ax1.set_title('Evolución de Ventas Globales (1980-2016)\nMuestra el crecimiento y declive de la industria', 
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(alpha=0.3)

# Subplot 2: Ventas por región
ax2.plot(ventas_regionales_tiempo.index, ventas_regionales_tiempo['Ventas NA'], 
         label='Norteamérica', linewidth=2.5, marker='o', markersize=4, color='#e74c3c')
ax2.plot(ventas_regionales_tiempo.index, ventas_regionales_tiempo['Ventas EU'], 
         label='Europa', linewidth=2.5, marker='s', markersize=4, color='#3498db')
ax2.plot(ventas_regionales_tiempo.index, ventas_regionales_tiempo['Ventas JP'], 
         label='Japón', linewidth=2.5, marker='^', markersize=4, color='#f39c12')
ax2.set_xlabel('Año', fontweight='bold', fontsize=12)
ax2.set_ylabel('Ventas por Región (Millones)', fontweight='bold', fontsize=12)
ax2.set_title('Evolución Regional del Mercado\nCompara el crecimiento de ventas en diferentes regiones', 
              fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizaciones/06_serie_tiempo.png', dpi=300, bbox_inches='tight')
print("   ✅ 06_serie_tiempo.png")
plt.close()

# Análisis por era
print("\n📊 COMPARACIÓN POR ERAS:")
for era in sorted(df['Era'].dropna().unique()):
    df_era = df[df['Era'] == era]
    print(f"\n   {era}:")
    print(f"      Juegos lanzados: {len(df_era):,}")
    print(f"      Ventas totales: {df_era['Ventas Global'].sum():,.0f}M")
    print(f"      Ventas promedio: {df_era['Ventas Global'].mean():.2f}M")
    print(f"      Género líder: {df_era.groupby('Genero')['Ventas Global'].sum().idxmax()}")
    print(f"      Plataforma líder: {df_era.groupby('Plataforma')['Ventas Global'].sum().idxmax()}")

# Gráfica de géneros por era
plt.figure(figsize=(14, 8))
top_generos_era = df.groupby(['Era', 'Genero'])['Ventas Global'].sum().reset_index()
pivot_generos = top_generos_era.pivot_table(values='Ventas Global', index='Genero', columns='Era', fill_value=0)
top8_generos = pivot_generos.sum(axis=1).nlargest(8).index
pivot_generos_top = pivot_generos.loc[top8_generos]

pivot_generos_top.plot(kind='bar', width=0.8, figsize=(14, 7), colormap='Set2', edgecolor='black', linewidth=1)
plt.xlabel('Género', fontweight='bold', fontsize=12)
plt.ylabel('Ventas Totales (Millones)', fontweight='bold', fontsize=12)
plt.title('Evolución de Géneros a través de las Eras\nMuestra qué géneros dominaron en cada época de la industria', 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(title='Era', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizaciones/07_generos_por_era.png', dpi=300, bbox_inches='tight')
print("   ✅ 07_generos_por_era.png")
plt.close()

# ============================================
# 6. ANÁLISIS AVANZADO ADICIONAL
# ============================================
print("\n" + "="*80)
print("📊 ANÁLISIS GEOGRÁFICO Y DE MERCADO")
print("="*80)

# Distribución geográfica
ventas_totales_regiones = {
    'Norteamérica': df['Ventas NA'].sum(),
    'Europa': df['Ventas EU'].sum(),
    'Japón': df['Ventas JP'].sum(),
    'Otros': df['Ventas Otros'].sum()
}

print(f"\n🌍 DISTRIBUCIÓN GEOGRÁFICA:")
for region, ventas in ventas_totales_regiones.items():
    porcentaje = (ventas / df['Ventas Global'].sum()) * 100
    print(f"   {region:15} {ventas:8,.0f}M ({porcentaje:5.1f}%)")

# Gráfica de torta mejorada
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Torta 1: Distribución regional
colors_geo = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#95E1D3']
explode = (0.1, 0, 0, 0)
wedges, texts, autotexts = ax1.pie(ventas_totales_regiones.values(), 
                                     labels=ventas_totales_regiones.keys(),
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=colors_geo,
                                     explode=explode,
                                     shadow=True,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Distribución Geográfica de Ventas\n(1980-2016)', fontsize=14, fontweight='bold', pad=20)

# Torta 2: Distribución por categoría de éxito
colors_cat = {'Fracaso': '#e74c3c', 'Moderado': '#f39c12', 'Éxito': '#3498db', 'Blockbuster': '#2ecc71'}
cat_counts = df['Categoria_Exito'].value_counts()
ax2.pie(cat_counts.values,
        labels=cat_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors_cat[cat] for cat in cat_counts.index],
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Distribución por Nivel de Éxito\n(Cantidad de juegos)', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('visualizaciones/08_distribucion_geografica_exito.png', dpi=300, bbox_inches='tight')
print("   ✅ 08_distribucion_geografica_exito.png")
plt.close()

# Top editoriales con análisis
print(f"\n🏢 TOP 10 EDITORIALES MÁS EXITOSAS:")
top_editoriales = df.groupby('Editorial').agg({
    'Ventas Global': ['sum', 'mean', 'count'],
    'Categoria_Exito': lambda x: (x == 'Blockbuster').sum()
}).round(2)
top_editoriales.columns = ['Ventas_Totales', 'Ventas_Promedio', 'Num_Juegos', 'Blockbusters']
top_editoriales = top_editoriales.sort_values('Ventas_Totales', ascending=False).head(10)

for i, (editorial, row) in enumerate(top_editoriales.iterrows(), 1):
    print(f"   {i:2}. {editorial:20} | {row['Ventas_Totales']:7,.0f}M total | "
          f"{row['Ventas_Promedio']:.2f}M promedio | {int(row['Num_Juegos'])} juegos | "
          f"{int(row['Blockbusters'])} blockbusters")

# ============================================
# 7. GUARDAR TODO
# ============================================
print("\n" + "="*80)
print("💾 GUARDANDO MODELOS Y DATOS")
print("="*80)

# Guardar todo en un diccionario
datos_completos = {
    'clasificador': clf,
    'le_genero': le_genero,
    'le_plataforma': le_plataforma,
    'le_editorial': le_editorial,
    'le_categoria': le_categoria,
    'kmeans': kmeans,
    'scaler': scaler,
    'metricas_clasificacion': {
        'accuracy': accuracy,
        'cross_val': cross_val_score(clf, X, y, cv=5).mean()
    },
    'importancias': importancias_clf.to_dict('records'),
    'categorias': categorias.tolist()
}

with open('modelo_entrenado.pkl', 'wb') as f:
    pickle.dump(datos_completos, f)

print("✅ modelo_entrenado.pkl guardado")

# Guardar dataset procesado
df_modelo.to_csv('datos_procesados.csv', index=False)
print("✅ datos_procesados.csv guardado")

# ============================================
# 8. RESUMEN EJECUTIVO
# ============================================
print("\n" + "="*80)
print("📋 RESUMEN EJECUTIVO")
print("="*80)

print(f"""
🎮 ANÁLISIS COMPLETADO CON ÉXITO

📊 DATOS ANALIZADOS:
   - Período: {df['Año'].min()}-{df['Año'].max()}
   - Total juegos: {len(df):,}
   - Ventas totales: {df['Ventas Global'].sum():,.0f}M copias

🤖 MODELO DE CLASIFICACIÓN:
   - Tipo: Random Forest Classifier
   - Precisión: {accuracy*100:.2f}%
   - Categorías predichas: 4 (Fracaso, Moderado, Éxito, Blockbuster)
   - Factor más importante: {importancias_clf.iloc[0]['Variable']} ({importancias_clf.iloc[0]['Importancia']:.1%})

🔬 CLUSTERING:
   - Clusters identificados: {K_optimo}
   - Agrupa juegos con características similares de mercado

📈 SERIES DE TIEMPO:
   - Año pico: {año_peak} ({ventas_peak:.0f}M ventas)
   - Tendencia: Crecimiento hasta 2008, declive después
   - Mercado dominante: Norteamérica ({ventas_totales_regiones['Norteamérica']:,.0f}M)

📁 ARCHIVOS GENERADOS:
   - 8 visualizaciones en carpeta 'visualizaciones/'
   - modelo_entrenado.pkl (modelos de ML)
   - datos_procesados.csv (dataset limpio)

🚀 SIGUIENTE PASO:
   Ejecuta: streamlit run app_streamlit.py
""")

print("="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)