import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns

# Llamamos al estilo CSS
st.set_page_config(layout="wide")   
with open("style.css") as s:
    st.markdown(f"<style>{s.read()}</style>", unsafe_allow_html=True)


penguins_df = pd.read_csv('./data/penguins.csv')
st.write(penguins_df.head())


st.title("Palmer's Penguins")

st.markdown('¡Utiliza esta aplicación Streamlit para crear tu propio diagrama de dispersión sobre pingüinos!')



penguin_file = st.file_uploader("Select Your Local Penguins CSV (default provided)")
if penguin_file is not None:
    penguins_df = pd.read_csv(penguin_file)
else:
    penguins_df = pd.read_csv("./data/penguins.csv")


selected_x_var = st.selectbox('¿Cuál quieres que sea la variable x?',
                              ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])

selected_y_var = st.selectbox('¿Qué pasa con el y?',
                              ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])

alt_chart = (
    alt.Chart(penguins_df, title="Scatterplot of Palmer's Penguins")
    .mark_circle()
    .encode(
        x=selected_x_var,
        y=selected_y_var,
        color="species",
)
    .interactive()
)
st.altair_chart(alt_chart, use_container_width=True)