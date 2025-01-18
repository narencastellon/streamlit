# Cargar las librerias

import streamlit as st 
import base64
from PIL import Image

# Vamos aprender

#1. Cargar imagenes
#2. Cargar videos de Youtube
#3. cargar audio 

# Cargar imagenes

col1, col2 = st.columns(2)

with col1:
    st.write("Display Imagen")
    st.image("forecast accuracy.jpeg")
    st.write("Imagen de Cortesi: Naren Castello")

with col2:
    # Cargar imagen de una URL

    st.write("Imagen URL")
    st.image("https://evotic.es/wp-content/uploads/2022/10/dashboard_ventas.png")
    st.write("Cortesia: evotic.es")

# Agregar multiples imagenes
    
st.write("Multiples Imagenes")

varios = ["forecast accuracy.jpeg", "Cerveza_Forecasting.png", "https://evotic.es/wp-content/uploads/2022/10/dashboard_ventas.png"]

st.image(varios)

# imagen de background

def add_local_backgound_image_(image):
    # Opening Image and converting to base64
    with open(image, "rb") as image:
        encoded_string = base64.b64encode(image.read())
    
    st.write("Image Courtesy: unplash")

    # Embedding image in markdown to background
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:files/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

st.write("Background Image")

# Calling Image in function
#add_local_backgound_image_('leopardo.png')

# Agregar audio

sample_audio = open("Rey_Ruiz.mp4", "rb")

audio_byte = sample_audio.read()

# Display audio

st.audio(sample_audio)
st.write("Cortesia : https://www.youtube.com/watch?v=NNRpdg-dOao&list=RDNNRpdg-dOao&start_radio=1")

# Display video 

sample_video = open("post para linkedin webinar redes sociales gris obscuro gradiente-2.mp4", "rb").read()

st.video(sample_video)

# display videos de Youtube

st.video("https://www.youtube.com/watch?v=v_MgbNr-blA")
st.write("Cortesia: Naren Castellon Channel")

# Agregar animacion

# Globos
# Primera animacion
st.balloons()

# animacion snowflake
st.snow()

# Agregar Emojis

emojis = """:rain_cloud: :coffee: :love_hotel: 	🦸‍♂️🦸‍♀️ :couple_with_heart: 💍"""

# Displaying Shortcodes
st.title(emojis)

st.write("HEMOS LLEGADO AL FINAL ESTE TUTORIAL PARA MOSTRAR MULTIMEDIA")


