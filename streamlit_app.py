import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Crear 5 pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pestaña de Rendimiento", "Pestaña de Mantenimiento", "Pestaña 3", "Pestaña 4", "Pestaña 5"
])


#----------------------------------------INICIO PESTAÑA 1----------------------------------------#
with tab1:
    st.title("Informe de Rendimiento")
    st.markdown("Esta sección presenta dos tablas (variaciones y calificaciones) junto con gráficos de variación.")

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
    df_variation = df[cols_variation].sort_values(by="Variación Importe (%)", ascending=True)

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
    df_cal = df[["Unidad", "Calificacion"]].sort_values(by="Calificacion", ascending=False)
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
    st.markdown(
        """
        Esta sección presenta la distribución de mantenimientos organizados por periodo y por tipo de servicio, 
        junto con la distribución del gasto entre las unidades.
        """
    )

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


with tab3:
    st.header("Contenido de la Pestaña 3")
    st.write("Contenido para la tercera pestaña.")

with tab4:
    st.header("Contenido de la Pestaña 4")
    st.write("Información que corresponde a la cuarta pestaña.")

with tab5:
    st.header("Contenido de la Pestaña 5")
    st.write("Finalmente, el contenido de la quinta pestaña.")
