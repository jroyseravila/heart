import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import time

# Configuración de la página
st.set_page_config(
    page_title="Predicción Cardíaca 💓",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para estilos
st.markdown("""
    <style>
        /* Fondo de la página */
        .main {
            background-color: #f5f5f5;
            color: #4a4a4a;
            font-family: 'Arial', sans-serif;
        }

        /* Títulos */
        h1, h2, h3 {
            color: #4b6584;
            font-family: 'Arial Black', sans-serif;
        }

        /* Botones */
        .stButton button {
            background-color: #38ada9;
            color: white;
            border-radius: 12px;
            font-size: 16px;
            font-weight: bold;
            border: 2px solid #38ada9;
        }
        .stButton button:hover {
            background-color: #3c6382;
            border-color: #3c6382;
        }

        /* Encabezado lateral */
        .css-1aumxhk {
            background-color: #82ccdd;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Ruta del modelo
MODEL_PATH = os.path.join("modelos", "corazon_m.pkl")

# Cargar el modelo de predicción
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    st.sidebar.success("Modelo cargado correctamente.")
except FileNotFoundError:
    st.sidebar.error(f"No se encontró el modelo en: {MODEL_PATH}")
    model = None

# Función para realizar la predicción
def realizar_prediccion(input_data):
    try:
        if model is None:
            raise ValueError("El modelo no está cargado.")
        
        # Convertir a formato adecuado para el modelo
        input_data = np.array(input_data).reshape(1, -1)
        
        # Predicción
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
        # Convertir a porcentaje
        probabilities = probabilities[0] * 100  # Ajustar a porcentaje
        return prediction[0], probabilities
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")
        return None, None

# Función para mostrar gráficos interactivos
def mostrar_grafico_probabilidades(probabilities):
    labels = ['Sin Riesgo', 'Riesgo']
    fig = go.Figure(data=[go.Pie(labels=labels, values=probabilities, hole=.3)])
    fig.update_layout(title="Distribución de Probabilidades (%)", title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

# Menú interactivo
with st.sidebar:
    st.markdown("### 🧭 Menú Principal")
    selected = option_menu(
        menu_title="Navegación",
        options=["Inicio", "Predicción", "Acerca de"],
        icons=["house", "activity", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Página de Inicio
if selected == "Inicio":
    st.image("assets/welcome_banner.png", use_container_width=True)
    st.title("💓 Plataforma de Predicción Cardíaca")
    st.markdown("### Explora nuestras funcionalidades:")
    st.markdown("""
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div style="background-color: #82ccdd; padding: 20px; border-radius: 10px; width: 30%; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);">
            <h3>📊 Predicción</h3>
            <p>Realiza predicciones personalizadas sobre tu salud cardíaca.</p>
        </div>
        <div style="background-color: #60a3bc; padding: 20px; border-radius: 10px; width: 30%; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);">
            <h3>🔎 Análisis</h3>
            <p>Visualiza resultados detallados y gráficas interactivas.</p>
        </div>
        <div style="background-color: #3c6382; padding: 20px; border-radius: 10px; width: 30%; box-shadow: 2px 2px 10px rgba(0,0,0,0.2); color: white;">
            <h3>📚 Información</h3>
            <p>Aprende cómo cuidar tu salud y prevenir riesgos cardíacos.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Predicción de Salud Cardíaca
elif selected == "Predicción":
    st.header("Predicción Individual de Salud Cardíaca")
    st.markdown("Introduce los datos clínicos para predecir el riesgo cardíaco.")
    
    # Explicación interactiva
    with st.expander("¿Qué significan estos datos?"):
        st.write("""
        - **Edad:** Factores de riesgo aumentan con la edad.
        - **Género:** Los hombres tienen mayor riesgo cardíaco hasta la menopausia.
        - **Presión Arterial:** Altos niveles pueden indicar hipertensión.
        - **Colesterol:** Factor clave para enfermedades cardíacas.
        - **Frecuencia Cardíaca Máxima:** Puede indicar problemas cardiovasculares.
        """)

    # Recopilar datos del usuario
    user_data = {
        "Edad": st.slider("Edad (años)", 20, 90, 50),
        "Género": st.radio("Género", ["Masculino", "Femenino"]),
        "Presión Arterial": st.slider("Presión Arterial en Reposo (mmHg)", 80, 200, 120),
        "Colesterol": st.slider("Colesterol (mg/dl)", 100, 400, 200),
        "Frecuencia Cardíaca": st.slider("Frecuencia Máxima (lpm)", 60, 220, 150),
    }
    
    # Procesar entrada
    input_data = [
        user_data["Edad"],
        1 if user_data["Género"] == "Masculino" else 0,
        user_data["Presión Arterial"],
        user_data["Colesterol"],
        user_data["Frecuencia Cardíaca"],
    ]

    if st.button("Evaluar Riesgo"):
        with st.spinner('Procesando...'):
            for i in range(101):
                time.sleep(0.02)
                st.progress(i)
        prediction, probabilities = realizar_prediccion(input_data)
        
        if prediction is not None:
            st.success(f"Resultado: {'Riesgo Cardíaco' if prediction == 1 else 'Sin Riesgo'}")
            st.write(f"Probabilidad de Sin Riesgo: {probabilities[0]:.2f}%")
            st.write(f"Probabilidad de Riesgo: {probabilities[1]:.2f}%")
            mostrar_grafico_probabilidades(probabilities)


