import streamlit as st
import pandas as pd
import pyreadstat
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt

# --- Carga y combinación de datos ---
@st.cache_data
def cargar_datos():
    df_estudiantes, _ = pyreadstat.read_sav('PISA2022_Estudiantes_Esp.sav')
    df_centros, _ = pyreadstat.read_sav('PISA2022_CentrosEducativos_Esp.sav')
    df = pd.merge(df_estudiantes, df_centros, on='CNTSCHID', how='left')
    columnas_esenciales = ['PV1MATH', 'ESCS', 'ST004D01T']
    df = df.dropna(subset=columnas_esenciales)
    df['GENERO'] = df['ST004D01T'].map({1: 'Hombre', 2: 'Mujer'})
    region_map = {
        72401: 'Andalucía', 72402: 'Aragón', 72403: 'Asturias', 72404: 'Islas Baleares',
        72405: 'Canarias', 72406: 'Cantabria', 72407: 'Castilla y León', 72408: 'Castilla-La Mancha',
        72409: 'Cataluña', 72410: 'Com. Valenciana', 72411: 'Extremadura', 72412: 'Galicia',
        72413: 'Com. de Madrid', 72414: 'Región de Murcia', 72415: 'Navarra', 72416: 'País Vasco',
        72417: 'La Rioja', 72418: 'Ceuta', 72419: 'Melilla'
    }
    df['REGION_NOMBRE'] = df['REGION_x'].map(region_map)
    df['NIVEL_ESCS'] = pd.cut(df['ESCS'], bins=[-4, -1, 1, 4], labels=['Bajo', 'Medio', 'Alto'])
    bullying_cols = [col for col in df.columns if 'ST038Q' in col]
    if bullying_cols:
        df['BULLYING_CAT'] = df[bullying_cols].apply(lambda row: any(row == 1), axis=1)
    elif 'BULLIED' in df.columns:
        df['BULLYING_CAT'] = df['BULLIED'] == 1
    else:
        df['BULLYING_CAT'] = None

    # --- Rellena los NaN de SCHLTYPE con "Público" y mapea valores numéricos ---
    df['SCHLTYPE'] = df['SCHLTYPE'].fillna('Público')
    mapa_tipo = {1: 'Público', 2: 'Concertado', 3: 'Privado'}
    df['SCHLTYPE'] = df['SCHLTYPE'].replace(mapa_tipo)
    df['SCHLTYPE'] = df['SCHLTYPE'].fillna('Público')
    return df

df = cargar_datos()

# --- Título global arriba de la visualización ---
st.title("Dashboard PISA 2022 España")
st.markdown("""
Visualización de los resultados de PISA 2022 por comunidad autónoma, género y variables socioeconómicas.
""")

# --- Filtro por tipo de centro ---
tipos_escuela = df['SCHLTYPE'].unique().tolist()
if 'Todos' not in tipos_escuela:
    tipos_escuela = ['Todos'] + tipos_escuela

tipo_seleccionado = st.selectbox(
    "Selecciona el tipo de centro:",
    tipos_escuela,
    key="filtro_tipo_escuela"
)

if tipo_seleccionado != 'Todos':
    df_filtrado = df[df['SCHLTYPE'] == tipo_seleccionado]
else:
    df_filtrado = df

# --- Muestra el número total de estudiantes ---
num_estudiantes = len(df_filtrado)
st.markdown(f"**Número total de estudiantes en la muestra seleccionada:** {num_estudiantes}")

# --- Pestañas ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Comparación por comunidades",
    "Comparación por género",
    "Análisis por tipo de centro",
    "Análisis socioeconómico"
])

with tab1:
    st.markdown("""
    Visualización de la puntuación media global por comunidad autónoma.
    """)
    # --- Heatmap ---
    st.markdown("### Heatmap: Puntuación media por comunidad autónoma y asignatura")
    df_heat = df_filtrado.groupby('REGION_NOMBRE')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heat, annot=True, cmap='coolwarm', fmt=".1f")
    plt.title('Puntuación media por comunidad autónoma y asignatura (PISA 2022)')
    plt.tight_layout()
    st.pyplot(plt)

    # --- Gráficos de barras por asignatura ---
    st.markdown("### Gráficas de barras con la puntuación media por comunidad autónoma y asignatura")
    asignaturas = ['PV1MATH', 'PV1READ', 'PV1SCIE']
    nombres_asig = {'PV1MATH': 'Matemáticas', 'PV1READ': 'Lectura', 'PV1SCIE': 'Ciencias'}
    for asig in asignaturas:
        df_media = df_filtrado.groupby('REGION_NOMBRE')[asig].mean().reset_index()
        df_media = df_media.sort_values(asig, ascending=True)
        media_global = df_media[asig].mean()
        fig = px.bar(
            df_media,
            x=asig,
            y='REGION_NOMBRE',
            orientation='h',
            title=f"Puntuación media en {nombres_asig[asig]} por comunidad autónoma (PISA 2022)",
            labels={asig: 'Puntuación media', 'REGION_NOMBRE': 'Comunidad Autónoma'}
        )
        fig.add_vline(x=media_global, line_dash="dash", line_color="red",
                      annotation_text=f"Media nacional: {media_global:.1f}", annotation_position="bottom right")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    Visualización de la puntuación media en matemáticas, lectura y ciencias por comunidad autónoma y género.
    """)
    asignaturas = ['PV1MATH', 'PV1READ', 'PV1SCIE']
    nombres_asig = {'PV1MATH': 'Matemáticas', 'PV1READ': 'Lectura', 'PV1SCIE': 'Ciencias'}
    generos = ['Hombre', 'Mujer']
    for asig in asignaturas:
        nombre_asig = nombres_asig[asig]
        df_h = df_filtrado[df_filtrado['GENERO'] == 'Hombre'].groupby('REGION_NOMBRE')[asig].mean().reset_index()
        df_m = df_filtrado[df_filtrado['GENERO'] == 'Mujer'].groupby('REGION_NOMBRE')[asig].mean().reset_index()
        orden_h = df_h.sort_values(asig, ascending=True)['REGION_NOMBRE']
        orden_m = df_m.sort_values(asig, ascending=True)['REGION_NOMBRE']
        df_h = df_h.set_index('REGION_NOMBRE').loc[orden_h].reset_index()
        df_m = df_m.set_index('REGION_NOMBRE').loc[orden_m].reset_index()
        media_global = df_filtrado[asig].mean()
        media_hombres = df_filtrado[df_filtrado['GENERO'] == 'Hombre'][asig].mean()
        media_mujeres = df_filtrado[df_filtrado['GENERO'] == 'Mujer'][asig].mean()
        fig = make_subplots(
            rows=1, cols=2,
            shared_yaxes=False,
            horizontal_spacing=0.15,
            subplot_titles=[f"{nombre_asig} - Hombre", f"{nombre_asig} - Mujer"]
        )
        fig.add_trace(
            go.Bar(
                x=df_h[asig],
                y=df_h['REGION_NOMBRE'],
                orientation='h',
                name='Hombres',
                marker_color='cornflowerblue'
            ),
            row=1, col=1
        )
        fig.add_vline(x=media_hombres, line_dash="dash", line_color="red", row=1, col=1,
                      annotation_text=f"Media hombres: {media_hombres:.1f}", annotation_position="bottom right")
        fig.add_vline(x=media_global, line_dash="dot", line_color="green", row=1, col=1,
                      annotation_text=f"Media global: {media_global:.1f}", annotation_position="top right")
        fig.add_trace(
            go.Bar(
                x=df_m[asig],
                y=df_m['REGION_NOMBRE'],
                orientation='h',
                name='Mujeres',
                marker_color='lightpink'
            ),
            row=1, col=2
        )
        fig.add_vline(x=media_mujeres, line_dash="dash", line_color="red", row=1, col=2,
                      annotation_text=f"Media mujeres: {media_mujeres:.1f}", annotation_position="bottom right")
        fig.add_vline(x=media_global, line_dash="dot", line_color="green", row=1, col=2,
                      annotation_text=f"Media global: {media_global:.1f}", annotation_position="top right")
        fig.update_layout(
            height=600, width=950,
            title_text=f"Puntuación media en {nombre_asig} por comunidad autónoma y género (PISA 2022)",
            showlegend=False,
            margin=dict(t=100, l=120, r=40, b=40)
        )
        fig.update_yaxes(showticklabels=True, row=1, col=1)
        fig.update_yaxes(showticklabels=True, row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""
    Comparación del rendimiento académico por tipo de centro.
    """)
    asignaturas = ['PV1MATH', 'PV1READ', 'PV1SCIE']
    nombres_asig = {'PV1MATH': 'Matemáticas', 'PV1READ': 'Lectura', 'PV1SCIE': 'Ciencias'}
    for asig in asignaturas:
        fig = px.box(
            df,
            x='SCHLTYPE',
            y=asig,
            color='SCHLTYPE',
            title=f"Distribución de puntuaciones en {nombres_asig[asig]} por tipo de centro"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("""
    Análisis del impacto del nivel socioeconómico y el bullying en el rendimiento.
    """)
    # --- Impacto del nivel socioeconómico ---
    st.header("Impacto del nivel socioeconómico (ESCS)")
    df_escs = df_filtrado.groupby('NIVEL_ESCS')[['PV1MATH', 'PV1READ', 'PV1SCIE']].mean().reset_index()
    df_escs = df_escs.melt(id_vars='NIVEL_ESCS', var_name='Asignatura', value_name='Media')
    fig = px.bar(
        df_escs,
        x='NIVEL_ESCS',
        y='Media',
        color='Asignatura',
        barmode='group',
        title="Media de puntuación por asignatura y nivel socioeconómico"
    )
    st.plotly_chart(fig, use_container_width=True)
