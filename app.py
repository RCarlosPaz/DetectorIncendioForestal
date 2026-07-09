import streamlit as st

import torch

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision.models as models

from PIL import Image

import io



# --- Configuración del dispositivo (CPU para Streamlit Cloud si no hay GPU) ---

device = torch.device('cpu') # En Streamlit Cloud, es más probable que uses CPU



# --- Definición de la arquitectura del modelo (DEBE SER LA MISMA QUE SE ENTRENÓ) ---

@st.cache_resource # Cachea el modelo para que no se cargue en cada ejecución

def load_model():

    # Cargar la estructura de ResNet18

    model = models.resnet18(weights=None) # No necesitamos los pesos pre-entrenados de ImageNet aquí



    # Congelar las capas base (aunque no entrenaremos, es buena práctica mantener la misma estructura)

    for param in model.parameters():

        param.requires_grad = False



    # Reemplazar la última capa de clasificación

    num_caracteristicas = model.fc.in_features

    model.fc = nn.Linear(num_caracteristicas, 2) # 2 clases (fire, nofire)



    # Cargar los pesos entrenados

    # Asegúrate de que 'resnet18_biclase.pth' esté en el mismo directorio que app.py

    try:

        model.load_state_dict(torch.load('resnet18_biclase.pth', map_location=device))

    except FileNotFoundError:

        st.error("Error: El archivo del modelo 'resnet18_biclase.pth' no se encontró. ¡Asegúrate de subirlo a tu repositorio de GitHub!")

        st.stop()



    model.eval() # Poner el modelo en modo evaluación

    model = model.to(device)

    return model



model = load_model()



# --- Transformaciones para una sola imagen de inferencia ---

transforme_inferencia = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



# --- Mapeo de clases (DEBE COINCIDIR CON EL ORDEN DE ENTRENAMIENTO) ---

clases = ['fire', 'nofire'] # Asegúrate de que esto coincida con train_dataset.classes



# --- Interfaz de Streamlit ---

st.set_page_config(page_title="Clasificador de Fuego", page_icon="🔥")



st.title("🔥 Clasificador de Fuego (Biclase)")

st.markdown("Una aplicación para detectar fuego en imágenes usando un modelo ResNet18 pre-entrenado.")



# Barra lateral

st.sidebar.header("Acerca de")

st.sidebar.markdown("Esta aplicación utiliza un modelo de redes neuronales convolucionales (CNN) basado en ResNet18 para clasificar imágenes. El modelo ha sido entrenado para distinguir entre imágenes con **fuego** y sin **fuego**.")

st.sidebar.markdown("\n\n**Cómo usar:**\n1. Sube una imagen en la sección principal.\n2. La aplicación mostrará la imagen y la predicción del modelo.\n")

st.sidebar.markdown("--- desarrollado con ❤️ y PyTorch")



st.header("Sube tu imagen aquí")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:

    # Leer la imagen subida

    image_data = uploaded_file.getvalue()

    image = Image.open(io.BytesIO(image_data)).convert("RGB")



    # Crear columnas para mostrar la imagen y la predicción

    col1, col2 = st.columns(2)



    with col1:

        st.subheader("Imagen Subida")

        st.image(image, caption='Imagen para analizar', use_column_width=True)



    # Preprocesar la imagen

    input_tensor = transforme_inferencia(image)

    input_batch = input_tensor.unsqueeze(0) # Añadir una dimensión de batch

    input_batch = input_batch.to(device)



    # Realizar la predicción

    with torch.no_grad():

        output = model(input_batch)

        probabilities = torch.softmax(output, dim=1)

        _, predicted_idx = torch.max(probabilities, 1)



    # Obtener el nombre de la clase y la confianza

    predicted_class = clases[predicted_idx.item()]

    confidence = probabilities[0][predicted_idx.item()].item() * 100



    with col2:

        st.subheader("Resultado de la Predicción")

        if predicted_class == 'fire':

            st.success(f"Predicción: **¡Fuego detectado!**")

            st.balloons()

        else:

            st.info(f"Predicción: **No se detectó fuego.**")

        st.write(f"Confianza: {confidence:.2f}%")



    st.markdown("--- ")
