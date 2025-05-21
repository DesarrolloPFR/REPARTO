import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import io
import numpy as np
from sklearn.cluster import DBSCAN

st.set_page_config(layout="wide")

tab3, tab4, tab5, tab6 = st.tabs([
    "Desviaciones", "Seguridad de unidades", "Eventos/Incidentes", "Evaluación"])

#----------------------------------------INICIO PESTAÑA 1----------------------------------------#

#----------------------------------------FIN PESTAÑA 1----------------------------------------#


#----------------------------------------INICIO PESTAÑA 2----------------------------------------#
  


#----------------------------------------FIN PESTAÑA 2----------------------------------------#


#----------------------------------------INICIO PESTAÑA 3----------------------------------------#
with tab3:

    ############################################################
    # Cargar y preparar datos del archivo de entregas (general)
    ############################################################
    file_path_entregas = r"guías/PARADAS_UNIDADES_COORD.xlsx"
    try:
        df_entregas = pd.read_excel(file_path_entregas)
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        data_str = ""
        df_entregas = pd.read_csv(io.StringIO(data_str), sep='\t')
        st.info("Utilizando datos de ejemplo por falta de archivo original.")

    # Obtener todas las unidades disponibles (para incluir en la tabla)
    unidades_todas = sorted(df_entregas['unidad'].dropna().unique())

    ############################################################
    # Cargar archivo de paradas inválidas para el mapa y la tabla
    ############################################################
    file_path_invalidas = r"guías/PARADAS_INVALIDAS_MENSUAL.xlsx"
    try:
        df_invalidas = pd.read_excel(file_path_invalidas)
    except Exception as e:
        st.error(f"Error al cargar el archivo de paradas inválidas: {e}")
        st.stop()

    # Normalizar nombres de columnas (elimina espacios y forzar mayúsculas)
    df_invalidas.columns = df_invalidas.columns.str.strip().str.upper()

    # Convertir la columna FECHA a datetime
    df_invalidas['FECHA'] = pd.to_datetime(df_invalidas['FECHA'], errors='coerce')

    # Asegurarse de que las columnas de coordenadas sean numéricas
    df_invalidas['LATITUD_PARADA'] = pd.to_numeric(df_invalidas['LATITUD_PARADA'], errors='coerce')
    df_invalidas['LONGITUD_PARADA'] = pd.to_numeric(df_invalidas['LONGITUD_PARADA'], errors='coerce')

    ############################################################
    # Función para convertir tiempo de espera a minutos
    ############################################################
    def convert_tiempo_espera(t_str):
        try:
            if pd.isna(t_str):
                return 0
            t_str = str(t_str)
            parts = t_str.split(':')
            if len(parts) == 2:
                m_val = int(parts[0])
                s_val = int(parts[1])
                return m_val + s_val/60.0
            elif len(parts) == 3:
                h_val = int(parts[0])
                m_val = int(parts[1])
                s_val = int(parts[2])
                return h_val * 60 + m_val + s_val/60.0
            else:
                return float(t_str)
        except:
            return 0

    # Convertir la columna "TIEMPO_ESPERA" a minutos (usar el nombre normalizado)
    df_invalidas['TIEMPO_ESPERA_MIN'] = df_invalidas['TIEMPO_ESPERA'].apply(convert_tiempo_espera)

    ############################################################
    # Calcular tabla de desempeño (calificaciones) considerando todas las unidades
    # Cada parada inválida descuenta 5 puntos, con mínimo 0 y sin eventos se asigna 100.
    ############################################################
    def calcular_metricas_unidad(unidades_todas, df_invalidas):
        metricas = []
        # Contar paradas inválidas por unidad (agrupando por 'UNIDAD_TRACKING')
        conteo_invalidas = df_invalidas.groupby('UNIDAD_TRACKING').size().reset_index(name='PARADAS INVÁLIDAS')
        for unidad in unidades_todas:
            # Buscar si la unidad (de entregas) tiene paradas inválidas (se asume que la identificación es la misma)
            if unidad in conteo_invalidas['UNIDAD_TRACKING'].values:
                paradas_invalidas = conteo_invalidas[conteo_invalidas['UNIDAD_TRACKING'] == unidad]['PARADAS INVÁLIDAS'].iloc[0]
            else:
                paradas_invalidas = 0
            # Calcular la calificación: 100 - 5 * (número de paradas inválidas), mínimo 0.
            calificacion = max(0, 100 - (paradas_invalidas * 2.5))
            metricas.append({
                'Unidad': unidad,
                'Paradas Inválidas': paradas_invalidas,
                'Calificación': calificacion
            })
        return pd.DataFrame(metricas)

    # Construir la tabla de métricas
    df_metricas = calcular_metricas_unidad(unidades_todas, df_invalidas)
    # Excluir unidad 0 si aparece y ordenar por calificación
    df_metricas = df_metricas[df_metricas['Unidad'] != 0]
    df_metricas = df_metricas.sort_values(by='Calificación', ascending=False)

    # Estilizar la tabla: sin decimales en la columna Calificación y autoajuste
    styled_df = df_metricas.set_index("Unidad").style.format({"Calificación": "{:.1f}"})
    def color_calificacion(val):
        try:
            val_num = float(val)
            if val_num >= 90:
                return 'background-color: green; color: white;'
            elif val_num >= 70:
                return 'background-color: yellow;'
            else:
                return 'background-color: red; color: white;'
        except:
            return ''
    styled_df = styled_df.map(color_calificacion, subset=['Calificación'])
    
    st.subheader("Puntuación")
    st.dataframe(styled_df)

    ############################################################
    # Selector de Unidad para el mapa
    ############################################################
    unidades_map = sorted(df_invalidas['UNIDAD_TRACKING'].dropna().unique())
    if not unidades_map:
        st.warning("No hay unidades con paradas inválidas en el archivo.")
        st.stop()
    
    unidad_seleccionada = st.selectbox("Selecciona la unidad para ver en el mapa", unidades_map)

    ############################################################
    # Mapa de Desviaciones
    ############################################################
    st.subheader("Mapa de Desviaciones")
    # Filtrar datos para la unidad seleccionada
    df_unidad = df_invalidas[df_invalidas['UNIDAD_TRACKING'] == unidad_seleccionada].copy()

    if df_unidad.empty:
        st.info("No se encontraron desviaciones para la unidad seleccionada.")
    else:
        # Asegurarse de que existan coordenadas válidas
        df_cluster = df_unidad.dropna(subset=['LATITUD_PARADA', 'LONGITUD_PARADA']).copy()
        if df_cluster.empty:
            st.info("No se encontraron registros con coordenadas válidas.")
        else:
            # Aplicar clustering para agrupar paradas cercanas (dentro de 350 m)
            coords = df_cluster[['LATITUD_PARADA', 'LONGITUD_PARADA']].values
            coords_rad = np.radians(coords)
            eps = 350 / 6371000  # 350 m en radianes
            db = DBSCAN(eps=eps, min_samples=1, metric='haversine').fit(coords_rad)
            df_cluster['cluster'] = db.labels_

            # Agrupar por clusters, incluyendo las fechas de cada parada
            grouped = df_cluster.groupby('cluster').agg(
                conteo=('cluster', 'count'),
                tiempo_espera_promedio=('TIEMPO_ESPERA_MIN', 'mean'),
                LATITUD=('LATITUD_PARADA', 'mean'),
                LONGITUD=('LONGITUD_PARADA', 'mean'),
                destinos=('DESTINO', lambda x: ', '.join(list(set(x))[:3]) + ('...' if len(set(x)) > 3 else '')),
                fechas=('FECHA', lambda x: ', '.join(x.dropna().dt.strftime("%Y-%m-%d").tolist()))
            ).reset_index()
            grouped['conteo'] = grouped['conteo'].astype(int)

            # Determinar el centro del mapa (promedio de coordenadas)
            center_lat = grouped['LATITUD'].mean()
            center_lon = grouped['LONGITUD'].mean()

            # Crear el mapa
            m_desv = folium.Map(location=[center_lat, center_lon], zoom_start=10)

            # Función para determinar el radio del círculo según el conteo
            def get_radius(conteo):
                return 5 + 2 * conteo

            # Agregar cada cluster al mapa con popup que incluye fechas de cada parada
            for idx, row in grouped.iterrows():
                lat = row['LATITUD']
                lon = row['LONGITUD']
                conteo = int(row['conteo'])
                tiempo_prom = row['tiempo_espera_promedio']
                destinos = row['destinos']
                fechas = row['fechas']
                popup_text = f"""
                    <b>Paradas inválidas:</b> {conteo}<br>
                    <b>Tiempo de espera promedio:</b> {tiempo_prom:.1f} min<br>
                    <b>Destinos:</b> {destinos}<br>
                    <b>Fechas:</b> {fechas}
                """
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=get_radius(conteo),
                    popup=folium.Popup(popup_text, max_width=300),
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6
                ).add_to(m_desv)

            # Mostrar el mapa
            folium_static(m_desv, width=1750, height=800)
#----------------------------------------FIN PESTAÑA 3----------------------------------------#


#----------------------------------------INICIO PESTAÑA 4----------------------------------------#
with tab4:
    st.markdown("### **Seguridad de unidades**")

    # Función para aplicar estilo condicional a 'Puntuación de seguridad'
    def color_puntuacion_seguridad(val):
        if pd.isna(val):
            return ''
        try:
            val_int = int(val)
        except:
            return ''
        if val_int >= 90:
            return 'background-color: green; color: white;'
        elif val_int >= 70:
            return 'background-color: yellow;'
        else:
            return 'background-color: red; color: white;'
    
    # Funciones para cargar datos
    def cargar_informe_unidades():
        return pd.read_excel('seguridad/Informe_Diario_Unidades.xlsx', index_col=None)

    def cargar_informe_mensual_unidades():
        return pd.read_excel('seguridad/Informe_Mensual_Unidades.xlsx', index_col=None)

    # Selector de unidad
    unidades = ["Todas"] + list(cargar_informe_unidades()['Número de unidad'].unique())
    unidad_seleccionada = st.selectbox(
        "Selecciona una unidad",
        options=unidades,
        key="unidad_selector_tab4"
    )

    # Mostrar datos según la unidad seleccionada
    if unidad_seleccionada == "Todas":
        modo_informe = st.radio(
            "Selecciona el tipo de informe de seguridad:",
            ("Informe Diario", "Informe Mensual"),
            key="modo_informe_tab4"
        )
        
        if modo_informe == "Informe Diario":
            df_unidades = cargar_informe_unidades().reset_index(drop=True)
        else:
            df_unidades = cargar_informe_mensual_unidades().reset_index(drop=True)
    
        # Convertir la columna 'Puntuación de seguridad' a numérica y redondear (sin decimales)
        df_unidades['Puntuación de seguridad'] = pd.to_numeric(
            df_unidades['Puntuación de seguridad'], errors='coerce'
        ).round().astype('Int64')
    
        # Convertir y redondear a 1 decimal las columnas 'Distancia total km' y 'Combustible consumido (L)'
        df_unidades['Distancia total km'] = pd.to_numeric(
            df_unidades['Distancia total km'], errors='coerce'
        ).round(1)
    
        df_unidades['Combustible consumido (L)'] = pd.to_numeric(
            df_unidades['Combustible consumido (L)'], errors='coerce'
        ).round(1)
    
        # Columnas predeterminadas y adicionales
        columnas_predeterminadas = [
            'Número de unidad', 'Operador', 'Modelo', 
            'Distancia total km', 'Combustible consumido (L)', 
            'Puntuación de seguridad'
        ]
        columnas_disponibles = list(df_unidades.columns.difference(columnas_predeterminadas))
    
        columnas_seleccionadas = st.multiselect(
            "Selecciona las columnas adicionales para mostrar",
            options=columnas_disponibles,
            default=[]
        )
    
        columnas_finales = columnas_predeterminadas + columnas_seleccionadas
    
        # Aplicar formato a columnas específicas y estilo condicional a 'Puntuación de seguridad'
        df_styled = df_unidades[columnas_finales].set_index("Número de unidad").style.format({
            'Distancia total km': '{:.1f}',
            'Combustible consumido (L)': '{:.1f}'
        }).map(color_puntuacion_seguridad, subset=['Puntuación de seguridad'])
    
        st.write("Informe de Unidades", df_styled)
#----------------------------------------FIN PESTAÑA 4----------------------------------------#


#----------------------------------------INICIO PESTAÑA 5----------------------------------------#
with tab5:
    # --- Tabla de Acumulado por Unidad con clasificación y penalización ---
    try:
        # Cargar todas las unidades desde el archivo Excel
        df_unidades = pd.read_excel('unidades/unidades.xlsx', index_col=None)
        df_unidades.rename(columns={'id': 'Unidad'}, inplace=True)  # Asegurar que la columna sea 'Unidad'

        # Cargar eventos acumulados mensuales
        df_acumulado = pd.read_excel('eventos/eventos_mensual.xlsx', index_col=None)

        # Definir listas para clasificar los eventos
        graves = ["Crash", "Forward Collision Warning"]
        moderados = ["Harsh Brake", "Harsh Turn", "Inattentive Driving", "Mobile Usage"]
        leves = ["Rolling Stop", "Eating", "Drinking", "Obstructed Camera", "No Seat Belt"]

        # Función para obtener la clasificación de un evento
        def get_classification(event_type):
            if event_type in graves:
                return "Grave"
            elif event_type in moderados:
                return "Moderado"
            elif event_type in leves:
                return "Leve"
            else:
                return None

        # Asignar la clasificación a cada evento
        df_acumulado['Clasificación'] = df_acumulado['Tipo de evento'].apply(get_classification)

        # Agrupar por Unidad y Clasificación y contar los eventos
        df_counts = df_acumulado.groupby(['Unidad', 'Clasificación']).size().unstack(fill_value=0)

        # Asegurar que existan las columnas para cada categoría
        for cat in ["Grave", "Moderado", "Leve"]:
            if cat not in df_counts.columns:
                df_counts[cat] = 0

        # Ordenar las columnas en el orden deseado
        df_counts = df_counts[["Grave", "Moderado", "Leve"]]

        # Hacer un merge con todas las unidades y reemplazar NaN con 0
        df_counts = df_unidades.merge(df_counts, on="Unidad", how="left").fillna(0)

        # Calcular totales y penalización
        df_counts['Total eventos'] = df_counts[['Grave', 'Moderado', 'Leve']].sum(axis=1)
        df_counts['Penalización'] = df_counts['Grave'] * 10 + df_counts['Moderado'] * 5 + df_counts['Leve'] * 2.5

        # Calcular la calificación y asegurar que el valor mínimo sea 0
        df_counts['Calificación'] = (100 - df_counts['Penalización']).clip(lower=0)

        # Eliminar la columna de penalización de la visualización
        df_display = df_counts.drop(columns=['Penalización'])

        # Función para aplicar estilo condicional a 'Calificación'
        def color_calificacion(val):
            if pd.isna(val):
                return ''
            try:
                val_int = int(val)
            except:
                return ''
            if val_int >= 90:
                return 'background-color: green; color: white;'
            elif val_int >= 70:
                return 'background-color: yellow;'
            else:
                return 'background-color: red; color: white;'
            
        df_display = df_display.sort_values(by="Calificación", ascending=False)

        # Aplicar formato y estilo a la tabla
        df_styled = df_display.set_index("Unidad").style.format({
            'Grave': '{:.0f}',
            'Moderado': '{:.0f}',
            'Leve': '{:.0f}',
            'Total eventos': '{:.0f}',
            'Calificación': '{:.1f}'
        }).map(color_calificacion, subset=['Calificación'])

        st.markdown("### **Acumulado por Unidad**")
        st.write(df_styled)


    except Exception as e:
        st.write("Error al cargar el acumulado por unidad:", e)
    
    # --- Sección de Eventos de Seguridad ---
    st.markdown("### **Videos de incidentes**")
    
    # Función para cargar incidentes diarios
    def cargar_incidentes_diarios():
        return pd.read_excel('eventos/eventos_diario.xlsx', index_col=None)


    df_eventos = pd.read_excel('eventos/eventos_mensual.xlsx', index_col=None).reset_index(drop=True)
    
    # Verificar que el DataFrame contenga la columna 'Unidad' y que no esté vacío
    if 'Unidad' in df_eventos.columns and not df_eventos.empty:
        unidades = df_eventos['Unidad'].unique()
        unidad_seleccionada = st.selectbox("Selecciona la Unidad", options=unidades)
        
        incidentes_info = df_eventos[df_eventos['Unidad'] == unidad_seleccionada]
        
        if not incidentes_info.empty:
            columnas_mostrar = ['Tipo de evento', 'Operador', 'Hora', 'Unidad']
            st.write(f"Eventos de la Unidad {unidad_seleccionada}",
                     incidentes_info[columnas_mostrar].reset_index(drop=True))

            # Contenedor para videos
            video_container = st.container()
            
            with video_container:
                for idx, row in incidentes_info.iterrows():
                    st.subheader(f"Incidente: {row['Tipo de evento']} ----- Hora: {row['Hora']}")
                    
                    # Video Interior
                    if pd.notna(row.get('video_Interior')) and row.get('video_Interior') != "No video URL":
                        st.write("**Video Interior**")
                        try:
                            st.video(row['video_Interior'], format="video/mp4")
                        except:
                            st.error("Error al cargar video interior")
                    
                    # Video Exterior
                    if pd.notna(row.get('video_Exterior')) and row.get('video_Exterior') != "No video URL":
                        st.write("**Video Exterior**")
                        try:
                            st.video(row['video_Exterior'], format="video/mp4")
                        except:
                            st.error("Error al cargar video exterior")
        else:
            st.write(f"No se encontraron incidentes para la Unidad {unidad_seleccionada}.")
    else:
        st.write("No hay datos disponibles para seleccionar una unidad.")
#----------------------------------------FIN PESTAÑA 5----------------------------------------#


#----------------------------------------INICIO PESTAÑA 6----------------------------------------#
with tab6:
    st.title("Evaluación Total")

    # Crear un DataFrame vacío para almacenar todas las calificaciones
    df_calificaciones_totales = pd.DataFrame(columns=["Unidad", "Calificación Desviaciones", 
                                                      "Calificación Seguridad (Mensual)", "Calificación Eventos", "Promedio Total"])
    
    try:
        # Pestaña 3: Entregas (Desviaciones)
        # Se utiliza 'df_metricas' generado en la pestaña 3 con la columna "Calificación"
        df_cal_entregas = df_metricas[["Unidad", "Calificación"]].copy()
        
        # Pestaña 4: Seguridad Operador (Informe Mensual)
        # Se asume que la función cargar_informe_mensual_unidades() retorna un DataFrame que contiene
        # las columnas "Número de unidad" y "Puntuación de seguridad"
        df_seguridad_mensual = cargar_informe_mensual_unidades().copy()
        df_seguridad_mensual['Puntuación de seguridad'] = pd.to_numeric(
            df_seguridad_mensual['Puntuación de seguridad'], errors='coerce'
        ).round().astype('Int64')
        df_seguridad_mensual = df_seguridad_mensual.rename(columns={"Número de unidad": "Unidad"})
        df_cal_seguridad = df_seguridad_mensual[["Unidad", "Puntuación de seguridad"]].copy()
        
        # Pestaña 5: Eventos
        # Se asume que 'df_counts' es el DataFrame obtenido en la pestaña 5 con la calificación en la columna "Calificación"
        df_cal_eventos = df_counts.reset_index()[["Unidad", "Calificación"]].copy()
        
        # Obtener lista única de todas las unidades
        todas_unidades = sorted(list(set(
            list(df_cal_entregas["Unidad"].unique()) + 
            list(df_cal_seguridad["Unidad"].unique()) + 
            list(df_cal_eventos["Unidad"].unique())
        )))
        
        # Crear DataFrame con todas las unidades
        df_calificaciones_totales = pd.DataFrame({"Unidad": todas_unidades})
        
        # Merge con los DataFrames de cada pestaña
        
        df_calificaciones_totales = df_calificaciones_totales.merge(
            df_cal_entregas, on="Unidad", how="left"
        ).rename(columns={"Calificación": "Calificación Desviaciones"})
        
        df_calificaciones_totales = df_calificaciones_totales.merge(
            df_cal_seguridad, on="Unidad", how="left"
        ).rename(columns={"Puntuación de seguridad": "Calificación Seguridad (Mensual)"})
        
        df_calificaciones_totales = df_calificaciones_totales.merge(
            df_cal_eventos, on="Unidad", how="left"
        ).rename(columns={"Calificación": "Calificación Eventos"})
        
        # Reemplazar None/NaN por 100 en las columnas de calificación
        columnas_cal = ["Calificación Desviaciones", 
                        "Calificación Seguridad (Mensual)", "Calificación Eventos"]
        df_calificaciones_totales[columnas_cal] = df_calificaciones_totales[columnas_cal].fillna(100)
        
        # Calcular el promedio de las calificaciones, ignorando NaN (ahora ya no habrá, pues se rellenó con 100)
        df_calificaciones_totales["Promedio Total"] = df_calificaciones_totales[columnas_cal].mean(axis=1).round(1)
        
        # Eliminar la unidad "0" (comparando como string)
        df_calificaciones_totales = df_calificaciones_totales[df_calificaciones_totales["Unidad"].astype(str) != "0"]
        
        # Función para aplicar estilo a la columna "Promedio Total"
        def color_promedio_total(val):
            if pd.isna(val):
                return ''
            try:
                val_num = float(val)
                if val_num >= 90:
                    return 'background-color: green; color: white;'
                elif val_num >= 70:
                    return 'background-color: yellow;'
                else:
                    return 'background-color: red; color: white;'
            except:
                return ''
            
        df_calificaciones_totales = df_calificaciones_totales.sort_values(by="Promedio Total", ascending=False)

        # Aplicar formato numérico a las columnas y estilo a "Promedio Total"
        df_styled = df_calificaciones_totales.set_index("Unidad").style.format({
            "Calificación Desviaciones": '{:.1f}',
            "Calificación Seguridad (Mensual)": '{:.1f}',
            "Calificación Eventos": '{:.1f}',
            "Promedio Total": '{:.1f}'
        }).map(color_promedio_total, subset=["Promedio Total"])
        
        st.markdown("### **Puntuación por Unidad**")
        st.write(df_styled)
        
    except Exception as e:
        st.error(f"Error al calcular calificaciones totales: {e}")
        st.info("Asegúrese de haber navegado por todas las pestañas anteriores para cargar los datos necesarios.")
#----------------------------------------FIN PESTAÑA 6----------------------------------------#
