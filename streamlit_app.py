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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Pestaña de Rendimiento", "Pestaña de Mantenimiento", "Guías-Entregas", "Seguridad de operador", "Eventos", "Total Evaluación"])

#----------------------------------------INICIO PESTAÑA 1----------------------------------------#
with tab1:
    st.title("Informe de Rendimiento")

    # Leer el archivo Excel
    file_path = r"rendimiento/rendimiento_borrador_1.xlsx"
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    # Preprocesamiento: limpiar columnas monetarias (quitar '$' y comas)
    importe_cols = ["Importe (Promedio al mes)", "Importe"]
    for col in importe_cols:
        df[col] = df[col].replace({'\$': '', ',': ''}, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calcular las variaciones en porcentaje
    df["Variación Litros (%)"] = df["variacion_litros"] * 100
    df["Variación Importe (%)"] = df["variacion_importe"] * 100

    # --- Primera Tabla: Variaciones por Unidad ---
    cols_variation = [
        "Unidad",
        "Litros (Promedio al mes)",
        "Litros",
        "Variación Litros (%)",
        "Importe (Promedio al mes)",
        "Importe",
        "Variación Importe (%)",
    ]
    df_variation = df[cols_variation].set_index("Unidad").sort_values(by="Variación Importe (%)", ascending=True)

    # Diccionario de formateo: todos los valores numéricos a 1 decimal
    format_dict_variation = {
        "Litros (Promedio al mes)": "{:.1f}",
        "Litros": "{:.1f}",
        "Variación Litros (%)": "{:.1f}%",
        "Importe (Promedio al mes)": "{:.1f}",
        "Importe": "{:.1f}",
        "Variación Importe (%)": "{:.1f}%",
        "Calificacion": "{:.1f}"
    }

    # Función para colorear las variaciones según rangos
    def color_variacion(val):
        if val < 10:
            return 'background-color: green; color: white;'
        elif val < 30:
            return 'background-color: yellow;'
        else:
            return 'background-color: red; color: white;'

    styled_variation = (
        df_variation.style
        .format(format_dict_variation)
        .applymap(color_variacion, subset=["Variación Litros (%)", "Variación Importe (%)"])
        .hide(axis="index")
    )

    # --- Segunda Tabla: Calificaciones por Unidad ---
    df_cal = df[["Unidad", "Calificacion"]].set_index("Unidad").sort_values(by="Calificacion", ascending=False)
    format_dict_cal = {"Calificacion": "{:.1f}"}
    def color_cal(val):
        if val >= 90:
            return 'background-color: green; color: white;'
        elif val >= 70:
            return 'background-color: yellow;'
        else:
            return 'background-color: red; color: white;'
    styled_cal = df_cal.style.format(format_dict_cal).applymap(color_cal, subset=["Calificacion"])

    # Mostrar ambas tablas lado a lado
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tabla de Variaciones")
        st.dataframe(styled_variation, height=500)
    with col2:
        st.subheader("Tabla de Calificaciones")
        st.dataframe(styled_cal, height=500)

    # Para las gráficas: asignamos colores según el criterio definido
    def get_color(val):
        if val < 10:
            return 'green'
        elif val < 30:
            return 'yellow'
        else:
            return 'red'
    df_variation["color_litros"] = df_variation["Variación Litros (%)"].apply(get_color)
    df_variation["color_importe"] = df_variation["Variación Importe (%)"].apply(get_color)
    color_map = {"green": "green", "yellow": "yellow", "red": "red"}

    df_variation = df_variation.reset_index()
    # --- Gráfico: Variación en Litros (%) ---
    st.subheader("Gráfico: Variación en Litros (%)")
    fig_litros = px.bar(
        df_variation,
        x="Unidad",
        y="Variación Litros (%)",
        color="color_litros",
        color_discrete_map=color_map,
        hover_data={"Unidad": True, "Variación Litros (%)": True, "color_litros": False}
    )
    fig_litros.update_layout(showlegend=False)
    fig_litros.update_traces(hovertemplate="Unidad: %{x}<br>Variación Litros: %{y:.1f}%<extra></extra>")
    st.plotly_chart(fig_litros, use_container_width=True)

    # --- Gráfico: Variación en Importe (%) ---
    st.subheader("Gráfico: Variación en Importe (%)")
    fig_importe = px.bar(
        df_variation,
        x="Unidad",
        y="Variación Importe (%)",
        color="color_importe",
        color_discrete_map=color_map,
        hover_data={"Unidad": True, "Variación Importe (%)": True, "color_importe": False}
    )
    fig_importe.update_layout(showlegend=False)
    fig_importe.update_traces(hovertemplate="Unidad: %{x}<br>Variación Importe: %{y:.1f}%<extra></extra>")
    st.plotly_chart(fig_importe, use_container_width=True)
#----------------------------------------FIN PESTAÑA 1----------------------------------------#


#----------------------------------------INICIO PESTAÑA 2----------------------------------------#
with tab2:
    st.title("Mantenimiento de Unidades")

    # Leer el archivo Excel de mantenimiento
    file_path_maint = r"mantenimiento/mantenimiento_borrador_1.xlsx"
    try:
        df_maint = pd.read_excel(file_path_maint)
    except Exception as e:
        st.error(f"No se pudo leer el archivo de mantenimiento: {e}")
        st.stop()

    # Preprocesamiento: limpiar la columna de costo (quitar '$', comas y espacios)
    # Primero convertimos a string
    df_maint["Costo (Importe)"] = df_maint["Costo (Importe)"].astype(str)
    # Ahora podemos usar str.replace
    df_maint["Costo (Importe)"] = df_maint["Costo (Importe)"].str.replace(
        r'[\$,\s]', '', regex=True
    )
    # Finalmente convertimos a numérico
    df_maint["Costo (Importe)"] = pd.to_numeric(df_maint["Costo (Importe)"], errors="coerce")

    # Convertir "Fecha de mantenimiento" a datetime (formato dd/mm/yyyy)
    df_maint["Fecha de mantenimiento"] = pd.to_datetime(
        df_maint["Fecha de mantenimiento"], 
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Crear columnas para Año y Cuatrimestre
    df_maint["Año"] = df_maint["Fecha de mantenimiento"].dt.year
    df_maint["Mes"] = df_maint["Fecha de mantenimiento"].dt.month

    def get_cuatrimestre(month):
        if month <= 4:
            return "Ene-Abr"
        elif month <= 8:
            return "May-Ago"
        else:
            return "Sep-Dic"

    df_maint["Cuatrimestre"] = df_maint["Mes"].apply(get_cuatrimestre)
    df_maint["Periodo"] = df_maint["Año"].astype(str) + " - " + df_maint["Cuatrimestre"]

    ##########################
    # Tabla de Resumen Detallada (Ahora al principio)
    st.subheader("Tabla de Resumen por Unidad")
    df_summary = df_maint.pivot_table(
        values="Costo (Importe)",
        index="Unidad",
        columns=["Periodo", "Tipo de servicio"],
        aggfunc="sum",
        fill_value=0
    ).round(1)
    
    # Agregar totales por unidad
    df_summary["Total"] = df_summary.sum(axis=1)
    df_summary = df_summary.sort_values("Total", ascending=False)
    
    # Formatear los números en la tabla
    def format_currency(val):
        return f"${val:,.2f}"
    
    df_summary_formatted = df_summary.applymap(format_currency)
    st.dataframe(df_summary_formatted, use_container_width=True)

    ##########################
    # Nueva gráfica: Distribución del gasto total entre unidades
    st.subheader("Distribución del Gasto Total por Unidad")
    
    # Preparar datos para la gráfica
    df_unit_total = df_maint.groupby("Unidad")["Costo (Importe)"].sum().reset_index()
    df_unit_total = df_unit_total.sort_values("Costo (Importe)", ascending=True)  # Ordenar de menor a mayor
    
    # Calcular el porcentaje del total
    total_cost = df_unit_total["Costo (Importe)"].sum()
    df_unit_total["Porcentaje"] = (df_unit_total["Costo (Importe)"] / total_cost * 100).round(1)
    
    # Crear gráfica de barras horizontal
    fig_unit_dist = px.bar(
        df_unit_total,
        x="Costo (Importe)",
        y="Unidad",
        orientation='h',
        text=df_unit_total["Porcentaje"].apply(lambda x: f'{x:.1f}%'),
        title="Distribución del Gasto Total por Unidad"
    )
    
    # Personalizar la gráfica
    fig_unit_dist.update_traces(
        textposition='outside',
        hovertemplate="Unidad: %{y}<br>Costo: $%{x:,.2f}<br>Porcentaje: %{text}<extra></extra>"
    )
    fig_unit_dist.update_layout(
        xaxis_title="Costo Total ($)",
        yaxis_title="",
        showlegend=False
    )
    
    st.plotly_chart(fig_unit_dist, use_container_width=True)

    ##########################
    # Crear dos columnas para las gráficas de distribución
    col1, col2 = st.columns(2)

    # Gráfica 1: Distribución de Costos por Periodo (En la primera columna)
    with col1:
        df_period_distribution = df_maint.groupby("Periodo")["Costo (Importe)"].sum().reset_index()
        df_period_distribution["Costo (Importe)"] = df_period_distribution["Costo (Importe)"].round(1)
        
        st.subheader("Distribución por Periodo")
        fig_pie_period = px.pie(
            df_period_distribution,
            names="Periodo",
            values="Costo (Importe)",
            title="Distribución de Costos por Periodo",
            hole=0.4
        )
        fig_pie_period.update_traces(
            textinfo='percent+label', 
            hovertemplate="Periodo: %{label}<br>Costo: %{value:,.1f}<extra></extra>"
        )
        st.plotly_chart(fig_pie_period, use_container_width=True)

    # Gráfica 2: Distribución de Costos por Tipo de Servicio (En la segunda columna)
    with col2:
        df_type_distribution = df_maint.groupby("Tipo de servicio")["Costo (Importe)"].sum().reset_index()
        df_type_distribution["Costo (Importe)"] = df_type_distribution["Costo (Importe)"].round(1)
        
        st.subheader("Distribución por Tipo")
        fig_pie_type = px.pie(
            df_type_distribution,
            names="Tipo de servicio",
            values="Costo (Importe)",
            title="Distribución de Costos por Tipo de Servicio",
            hole=0.4
        )
        fig_pie_type.update_traces(
            textinfo='percent+label', 
            hovertemplate="Tipo: %{label}<br>Costo: %{value:,.1f}<extra></extra>"
        )
        st.plotly_chart(fig_pie_type, use_container_width=True)
#----------------------------------------FIN PESTAÑA 2----------------------------------------#


#----------------------------------------INICIO PESTAÑA 3----------------------------------------#
with tab3:
    st.header("Mapa de Entregas")

    ############################################################
    # Cargar y preparar datos del archivo de entregas (general) para el mapa
    ############################################################
    file_path_entregas = r"guías/PARADAS_UNIDADES_COORD.xlsx"
    try:
        df_entregas = pd.read_excel(file_path_entregas)
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        data_str = ""
        df_entregas = pd.read_csv(io.StringIO(data_str), sep='\t')
        st.info("Utilizando datos de ejemplo por falta de archivo original.")
    
    # Convertir columnas de coordenadas a numérico en df_entregas
    for col in ['LATITUD', 'LONGITUD', 'LAT_PARADA', 'LON_PARADA']:
        df_entregas[col] = pd.to_numeric(df_entregas[col], errors='coerce')
    
    # Asegurar que 'tiempo_espera' sea string y convertirlo a minutos en df_entregas
    df_entregas['tiempo_espera'] = df_entregas['tiempo_espera'].astype(str)
    def convert_to_minutes(time_str):
        try:
            if pd.isna(time_str) or time_str in ['', 'nan']:
                return 0
            if isinstance(time_str, pd.Timedelta):
                return time_str.total_seconds() / 60
            if isinstance(time_str, pd.Timestamp):
                return time_str.hour * 60 + time_str.minute + time_str.second / 60
            parts = time_str.split(':')
            if len(parts) == 3:
                horas = int(parts[0])
                minutos = int(parts[1])
                segundos = int(parts[2])
                return horas * 60 + minutos + segundos / 60
            elif len(parts) == 2:
                minutos = int(parts[0])
                segundos = int(parts[1])
                return minutos + segundos / 60
            else:
                return 0
        except:
            return 0
    df_entregas['tiempo_espera_minutos'] = df_entregas['tiempo_espera'].apply(convert_to_minutes)
    
    ############################################################
    # Cargar y preparar datos para la TABLA de Guías-Entregas (nuevo archivo)
    # Este archivo no contiene coordenadas, ya que se usa exclusivamente para las calificaciones.
    ############################################################
    file_path_tabla = r"guías/ACUMULADO_PARADAS_MENSUAL.xlsx"
    try:
        df_tabla = pd.read_excel(file_path_tabla)
    except Exception as e:
        st.error(f"Error al cargar el archivo para la tabla: {e}")
        data_str = ""
        df_tabla = pd.read_csv(io.StringIO(data_str), sep='\t')
        st.info("Utilizando datos de ejemplo por falta de archivo original para la tabla.")
    
    # No se realizan conversiones de coordenadas en df_tabla, ya que no las contiene.
    
    # Asegurar que 'tiempo_espera' sea string y convertirlo a minutos en df_tabla
    df_tabla['tiempo_espera'] = df_tabla['tiempo_espera'].astype(str)
    df_tabla['tiempo_espera_minutos'] = df_tabla['tiempo_espera'].apply(convert_to_minutes)
    
    ############################################################
    # Función para determinar las coordenadas a usar según el status.
    ############################################################
    def get_coordinates(row):
        status = str(row['parada_status']).upper()
        if "ENTREGADO" in status:
            if pd.notna(row['LAT_PARADA']) and pd.notna(row['LON_PARADA']):
                return row['LAT_PARADA'], row['LON_PARADA']
            else:
                return row['LATITUD'], row['LONGITUD']
        elif "NO ENTREGADO" in status or "PFR" in status:
            return row['LATITUD'], row['LONGITUD']
        elif "INVÁLIDA" in status:
            return row['LAT_PARADA'], row['LON_PARADA']
        else:
            return row['LATITUD'], row['LONGITUD']
    
    ############################################################
    # Tabla de Desempeño por Unidad (con datos de ACUMULADO_PARADAS_MENSUAL)
    ############################################################
    def calcular_metricas_unidad(df):
        unidades = sorted(df['unidad (Entrega)'].dropna().unique())
        metricas = []
        for unidad in unidades:
            df_unidad = df[df['unidad (Entrega)'] == unidad]
            
            total_paradas = len(df_unidad)
            entregados = df_unidad[df_unidad['parada_status'] == 'ENTREGADO'].shape[0]
            no_entregados = df_unidad[df_unidad['parada_status'].str.contains('NO ENTREGADO', na=False)].shape[0]
            paradas_invalidas = df_unidad[df_unidad['parada_status'].str.contains('INVÁLIDA', na=False)].shape[0]
            paradas_pfr = df_unidad[df_unidad['parada_status'].str.contains('PFR', na=False)].shape[0]
            otros = total_paradas - entregados - no_entregados - paradas_invalidas - paradas_pfr
            
            # Esperas largas (más de 20 minutos)
            esperas_largas = df_unidad[df_unidad['tiempo_espera_minutos'] > 20].shape[0]
            
            # Asignaciones incorrectas
            df_unidad_check = df_unidad.dropna(subset=['unidad (Asignada)'])
            asignaciones_incorrectas = df_unidad_check[
                df_unidad_check['unidad (Asignada)'] != df_unidad_check['unidad (Entrega)']
            ].shape[0]
            
            # Cálculo de penalizaciones según la fórmula solicitada
            if total_paradas > 0:
                penalty_no_entregados = (no_entregados / total_paradas) * 30   # 30% máximo
                penalty_invalidas = (paradas_invalidas / total_paradas) * 20    # 20% máximo
                penalty_esperas = (esperas_largas / total_paradas) * 10         # 10% máximo
                calificacion = max(0, 100 - (penalty_no_entregados + penalty_invalidas + penalty_esperas))
            else:
                calificacion = 0
            
            metricas.append({
                'Unidad': unidad,
                'Entregados': entregados,
                'No Entregados': no_entregados,
                'Paradas Inválidas': paradas_invalidas,
                'Paradas PFR': paradas_pfr,
                'Esperas Largas': esperas_largas,
                'Asignaciones Incorrectas': asignaciones_incorrectas,
                'Calificación': calificacion
            })
        return pd.DataFrame(metricas)
    
    # Se construye la tabla usando df_tabla (archivo nuevo)
    df_metricas = calcular_metricas_unidad(df_tabla)

    df_metricas = df_metricas[df_metricas['Unidad'] != 0]


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
    styled_df = styled_df.applymap(color_calificacion, subset=['Calificación'])
    st.dataframe(styled_df, width=1200)
    
    ############################################################
    # Selector de Unidad para definir "unidad_seleccionada"
    ############################################################
    unidades = sorted(df_entregas['unidad (Entrega)'].dropna().unique())
    unidad_seleccionada = st.selectbox("Selecciona la unidad", unidades)
    
    ############################################################
    # Selector del tipo de visualización (Mapa General / Mapa de Desviaciones)
    ############################################################
    tipo_mapa = st.radio(
        "Selecciona tipo de visualización",
        ("Mapa General", "Mapa de Desviaciones"),
        key="tipo_mapa",
        horizontal=True
    )
    
    ############################################################
    # Mapa General de Entregas (usa df_entregas)
    ############################################################
    if tipo_mapa == "Mapa General":
        st.subheader("Mapa de Entregas - General")
        df_unidad = df_entregas[df_entregas['unidad (Entrega)'] == unidad_seleccionada].copy()
        df_unidad['coords'] = df_unidad.apply(get_coordinates, axis=1)
        df_unidad['coord_lat'] = df_unidad['coords'].apply(lambda x: x[0] if isinstance(x, tuple) and pd.notna(x[0]) else None)
        df_unidad['coord_lon'] = df_unidad['coords'].apply(lambda x: x[1] if isinstance(x, tuple) and pd.notna(x[1]) else None)
        
        # Determinar el centro del mapa (Guadalajara por defecto)
        center_lat, center_lon = 20.6596988, -103.3496092
        entregados = df_unidad[(df_unidad['parada_status'].str.upper().str.contains("ENTREGADO")) &
                               (df_unidad['coord_lat'].notna()) &
                               (df_unidad['coord_lon'].notna())]
        if not entregados.empty:
            center_lat = entregados.iloc[0]['coord_lat']
            center_lon = entregados.iloc[0]['coord_lon']
        else:
            any_reg = df_unidad[df_unidad['coords'].apply(lambda x: pd.notna(x[0]) and pd.notna(x[1]))]
            if not any_reg.empty:
                center_lat = any_reg.iloc[0]['coords'][0]
                center_lon = any_reg.iloc[0]['coords'][1]
        
        df_filtrado = df_unidad[df_unidad['coords'].apply(lambda x: pd.notna(x[0]) and pd.notna(x[1]))]
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        for idx, row in df_filtrado.iterrows():
            lat, lon = row['coords']
            status = str(row['parada_status']).upper()
            hora_entrega = row['hora_entrega']
            nombre_comercial = row.get('NOMBRE_COMERCIAL', 'Sin nombre')
            # Selección de color según el estado
            if "ENTREGADO" in status:
                color = 'green'
            elif "INVÁLIDA" in status:
                color = 'red'
            elif "NO ENTREGADO" in status or "PFR" in status:
                color = 'orange'

            if "ENTREGADO" in status:
                popup_text = f"""
                    <b>{nombre_comercial}</b><br>
                    Clave Cliente: {row['CLAVE_CLIENTE_HIJO']}<br>
                    Unidad que entregó: {row['unidad (Entrega)']}<br>
                    Unidad asignada: {row.get('unidad (Asignada)', 'N/A')}<br>
                    Estado: {row['parada_status']}<br>
                    Hora de entrega: {hora_entrega}<br>
                    Tiempo de espera: {row['tiempo_espera_minutos']:.1f} min
                """
            elif "INVÁLIDA" in status:
                popup_text = f"""
                    <b>PARADA INVÁLIDA</b><br>
                    Tiempo de espera: {row['tiempo_espera_minutos']:.1f} min
                """
            else:
                popup_text = f"""
                    <b>{nombre_comercial}</b><br>
                    Clave Cliente: {row['CLAVE_CLIENTE_HIJO']}<br>
                    Estado: {row['parada_status']}<br>
                    Tiempo de espera: {row['tiempo_espera_minutos']:.1f} min
                """
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_text, max_width=250),
                icon=folium.Icon(color=color, prefix='fa', icon='circle')
            ).add_to(m)
        folium_static(m, width=1000)
    
    ############################################################
    # Mapa de Desviaciones
    ############################################################
    else:
        st.subheader("Mapa de Desviaciones")
        
        # Seleccionar periodo
        periodo_invalidas = st.radio("Selecciona periodo de desviaciones", ["Día Anterior", "Acumulado Mensual"], key="periodo_invalidas")
        if periodo_invalidas == "Día Anterior":
            file_invalidas = r"guías/PARADAS_INVALIDAS.xlsx"
        else:
            file_invalidas = r"guías/ACUMULADO_INVALIDAS.xlsx"
        try:
            df_invalidas = pd.read_excel(file_invalidas)
        except Exception as e:
            st.error(f"Error al cargar el archivo {file_invalidas}: {e}")
            st.stop()
        
        for col in ['LAT_PARADA', 'LON_PARADA']:
            df_invalidas[col] = pd.to_numeric(df_invalidas[col], errors='coerce')
        df_invalidas['FECHA'] = pd.to_datetime(df_invalidas['FECHA'], errors='coerce')
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
                    return h_val*60 + m_val + s_val/60.0
                else:
                    return float(t_str)
            except:
                return 0
        tiempo_espera_col = 'tiempo_espera'
        df_invalidas['tiempo_espera_min'] = df_invalidas[tiempo_espera_col].apply(convert_tiempo_espera)
        
        # Determinar la columna de unidad
        if 'unidad' in df_invalidas.columns:
            unidad_col = 'unidad'
        elif 'unidad (Entrega)' in df_invalidas.columns:
            unidad_col = 'unidad (Entrega)'
        else:
            st.error("No se encontró la columna de unidad en el archivo de desviaciones.")
            st.stop()
        
        df_invalidas = df_invalidas[df_invalidas[unidad_col] == unidad_seleccionada].copy()
        if df_invalidas.empty:
            st.info("No se encontraron desviaciones para la unidad y el periodo seleccionados.")
        else:
            df_cluster = df_invalidas.dropna(subset=['LAT_PARADA', 'LON_PARADA']).copy()
            if df_cluster.empty:
                st.info("No se encontraron registros con coordenadas válidas.")
            else:
                coords = df_cluster[['LAT_PARADA', 'LON_PARADA']].values
                coords_rad = np.radians(coords)
                eps = 350 / 6371000  # 350 m en radianes
                db = DBSCAN(eps=eps, min_samples=1, metric='haversine').fit(coords_rad)
                df_cluster['cluster'] = db.labels_
                
                grouped = df_cluster.groupby('cluster').agg(
                    conteo=('cluster', 'count'),
                    tiempo_espera_promedio=('tiempo_espera_min', 'mean'),
                    LAT_PARADA=('LAT_PARADA', 'mean'),
                    LON_PARADA=('LON_PARADA', 'mean')
                ).reset_index()
                grouped['conteo'] = grouped['conteo'].astype(int)
                
                center_lat = grouped['LAT_PARADA'].mean()
                center_lon = grouped['LON_PARADA'].mean()
                
                m_desv = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                def get_radius(conteo):
                    return 5 + 2 * conteo
                for idx, row in grouped.iterrows():
                    lat = row['LAT_PARADA']
                    lon = row['LON_PARADA']
                    conteo = int(row['conteo'])
                    tiempo_prom = row['tiempo_espera_promedio']
                    popup_text = f"""
                        <b>Conteo:</b> {conteo}<br>
                        <b>Tiempo de espera promedio:</b> {tiempo_prom:.1f} min
                    """
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=get_radius(conteo),
                        popup=folium.Popup(popup_text, max_width=250),
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.6
                    ).add_to(m_desv)
                folium_static(m_desv, width=1000)

#----------------------------------------FIN PESTAÑA 3----------------------------------------#


#----------------------------------------INICIO PESTAÑA 4----------------------------------------#
with tab4:
    st.header("Informe de Unidades")

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
        st.subheader("Información Seguridad de Unidades")
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
        }).applymap(color_puntuacion_seguridad, subset=['Puntuación de seguridad'])
    
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
        df_counts['Penalización'] = df_counts['Grave'] * 5 + df_counts['Moderado'] * 3 + df_counts['Leve'] * 1

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
            'Calificación': '{:.0f}'
        }).applymap(color_calificacion, subset=['Calificación'])

        st.markdown("### **Acumulado por Unidad**")
        st.write(df_styled)


    except Exception as e:
        st.write("Error al cargar el acumulado por unidad:", e)
    
    # --- Sección de Eventos de Seguridad ---
    st.header("Eventos de seguridad")
    st.markdown("### **Eventos de seguridad:**")
    
    # Función para cargar incidentes diarios
    def cargar_incidentes_diarios():
        return pd.read_excel('eventos/eventos_diario.xlsx', index_col=None)
    
    # Seleccionar el tipo de evento mediante un radio button
    tipo_eventos = st.radio(
        "Selecciona el tipo de evento",
        ("Día Anterior", "Acumulado Mensual"),
        key="tipo_eventos"
    )
    
    # Cargar el DataFrame correspondiente según el tipo seleccionado
    if tipo_eventos == "Día Anterior":
        df_eventos = cargar_incidentes_diarios().reset_index(drop=True)
    else:
        df_eventos = pd.read_excel('eventos/eventos_mensual.xlsx', index_col=None).reset_index(drop=True)
    
    # Verificar que el DataFrame contenga la columna 'Unidad' y que no esté vacío
    if 'Unidad' in df_eventos.columns and not df_eventos.empty:
        # Obtener las unidades disponibles y permitir su selección
        unidades = df_eventos['Unidad'].unique()
        unidad_seleccionada = st.selectbox("Selecciona la Unidad", options=unidades)
    
        # Filtrar los incidentes para la unidad seleccionada
        incidentes_info = df_eventos[df_eventos['Unidad'] == unidad_seleccionada]
    
        # Verificar si hay incidentes para la unidad seleccionada
        if not incidentes_info.empty:
            # Seleccionar solo las columnas necesarias para mostrar en la tabla
            columnas_mostrar = ['Tipo de evento', 'Operador', 'Hora', 'Unidad']
            st.write(f"{tipo_eventos} de la Unidad {unidad_seleccionada}",
                     incidentes_info[columnas_mostrar].reset_index(drop=True))
    
            # Recorrer cada incidente para mostrar la información y los videos correspondientes
            for _, row in incidentes_info.iterrows():
                st.subheader(f"Incidente: {row['Tipo de evento']} ----- Hora: {row['Hora']}")
    
                # Video Interior
                if pd.notna(row.get('video_Interior')) and row.get('video_Interior') != "No video URL":
                    st.write("**Video Interior**")
                    video_url_interior = row['video_Interior']
                    st.markdown(
                        f'<video width="640" height="360" controls><source src="{video_url_interior}" type="video/mp4"></video>',
                        unsafe_allow_html=True
                    )
                else:
                    st.write("**No hay video interior disponible.**")
    
                # Video Exterior
                if pd.notna(row.get('video_Exterior')) and row.get('video_Exterior') != "No video URL":
                    st.write("**Video Exterior**")
                    video_url_exterior = row['video_Exterior']
                    st.markdown(
                        f'<video width="640" height="360" controls><source src="{video_url_exterior}" type="video/mp4"></video>',
                        unsafe_allow_html=True
                    )
                else:
                    st.write("**No hay video exterior disponible.**")
        else:
            st.write(f"No se encontraron incidentes para la Unidad {unidad_seleccionada}.")
    else:
        st.write("No hay datos disponibles para seleccionar una unidad.")
#----------------------------------------FIN PESTAÑA 5----------------------------------------#


#----------------------------------------INICIO PESTAÑA 6----------------------------------------#
with tab6:
    st.title("Evaluación Total")

    # Crear un DataFrame vacío para almacenar todas las calificaciones
    df_calificaciones_totales = pd.DataFrame(columns=["Unidad", "Calificación Rendimiento", "Calificación Entregas", 
                                                      "Calificación Seguridad (Mensual)", "Calificación Eventos", "Promedio Total"])
    
    try:
        # Pestaña 1: Rendimiento
        df_cal_rendimiento = df[["Unidad", "Calificacion"]].sort_values(by="Calificacion", ascending=False).copy()
        
        # Pestaña 3: Entregas
        df_cal_entregas = df_metricas[["Unidad", "Calificación"]].copy()
        
        # Pestaña 4: Seguridad Operador (Informe Mensual)
        df_seguridad_mensual = cargar_informe_mensual_unidades().copy()
        df_seguridad_mensual['Puntuación de seguridad'] = pd.to_numeric(
            df_seguridad_mensual['Puntuación de seguridad'], errors='coerce'
        ).round().astype('Int64')
        df_seguridad_mensual = df_seguridad_mensual.rename(columns={"Número de unidad": "Unidad"})
        df_cal_seguridad = df_seguridad_mensual[["Unidad", "Puntuación de seguridad"]].copy()
        
        # Pestaña 5: Eventos
        df_cal_eventos = df_counts.reset_index()[["Unidad", "Calificación"]].copy()
        
        # Obtener lista única de todas las unidades
        todas_unidades = sorted(list(set(
            list(df_cal_rendimiento["Unidad"].unique()) + 
            list(df_cal_entregas["Unidad"].unique()) + 
            list(df_cal_seguridad["Unidad"].unique()) + 
            list(df_cal_eventos["Unidad"].unique())
        )))
        
        # Crear DataFrame con todas las unidades
        df_calificaciones_totales = pd.DataFrame({"Unidad": todas_unidades})
        
        # Merge con los DataFrames de cada pestaña
        df_calificaciones_totales = df_calificaciones_totales.merge(
            df_cal_rendimiento, on="Unidad", how="left"
        ).rename(columns={"Calificacion": "Calificación Rendimiento"})
        
        df_calificaciones_totales = df_calificaciones_totales.merge(
            df_cal_entregas, on="Unidad", how="left"
        ).rename(columns={"Calificación": "Calificación Entregas"})
        
        df_calificaciones_totales = df_calificaciones_totales.merge(
            df_cal_seguridad, on="Unidad", how="left"
        ).rename(columns={"Puntuación de seguridad": "Calificación Seguridad"})
        
        df_calificaciones_totales = df_calificaciones_totales.merge(
            df_cal_eventos, on="Unidad", how="left"
        ).rename(columns={"Calificación": "Calificación Eventos"})
        
        # Calcular el promedio de las calificaciones, ignorando NaN
        df_calificaciones_totales["Promedio Total"] = df_calificaciones_totales[[
            "Calificación Rendimiento", "Calificación Entregas", 
            "Calificación Seguridad", "Calificación Eventos"
        ]].mean(axis=1, skipna=True).round(1)
        
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
            "Calificación Rendimiento": '{:.1f}',
            "Calificación Entregas": '{:.1f}',
            "Calificación Seguridad": '{:.1f}',
            "Calificación Eventos": '{:.1f}',
            "Promedio Total": '{:.1f}'
        }).applymap(color_promedio_total, subset=["Promedio Total"])
        
        # Mostrar la tabla directamente usando st.write, como en el ejemplo que proporcionaste
        st.markdown("### **Calificaciones por Unidad**")
        st.write(df_styled)
        
    except Exception as e:
        st.error(f"Error al calcular calificaciones totales: {e}")
        st.info("Asegúrese de haber navegado por todas las pestañas anteriores para cargar los datos necesarios.")
#----------------------------------------FIN PESTAÑA 6----------------------------------------#
