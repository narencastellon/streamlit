import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time

# plot
# =========================================================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
import pydeck as pdk


# Llamamos al estilo CSS
st.set_page_config(layout="wide")   
with open("style1.css") as s:
    st.markdown(f"<style>{s.read()}</style>", unsafe_allow_html=True)

 
    
# Cargamos los datos
trees_df = pd.read_csv("https://raw.githubusercontent.com/tylerjrichards/streamlit_apps/main/trees_app/trees.csv")
trees_df.set_index("date", inplace= True)
df_dbh_grouped = pd.DataFrame(trees_df.groupby(['dbh']).count()['tree_id'])



# Usaremos las librerias Plotly, Altair, Seaborn, y Matplotlib
st.write("Datos", trees_df.head())


# Grafico con usado las libreria de directa con Streamlit

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Grafico de Barra")
    st.bar_chart(df_dbh_grouped )

with col2:
    st.header("Grafico de Linea")
    st.line_chart(trees_df["dbh"].tail(500))

with col3:
    st.header("Area")    
    st.area_chart(df_dbh_grouped)

# Grafico usando plotly

st.header("Grafico usando Plotly")

fig = px.histogram(trees_df['dbh'])
st.plotly_chart(fig)


# Graficos seaborn
trees_df = pd.read_csv("https://raw.githubusercontent.com/tylerjrichards/streamlit_apps/main/trees_app/trees.csv")
trees_df["date"] = pd.to_datetime(trees_df['date']) #format='%d/%m/%Y'
trees_df['age'] = (pd.to_datetime('today') - pd.to_datetime(trees_df['date'])).dt.days


trees_df["year"] = trees_df["date"].dt.year

st.subheader('Seaborn Chart')
fig_sb, ax_sb = plt.subplots()
ax_sb = sns.histplot(trees_df['age'])
plt.xlabel('Age (Days)')
st.pyplot(fig_sb)

fig = plt.figure(figsize=(18, 5))
sns.countplot(x = "caretaker", data = trees_df)
st.pyplot(fig)

# Graficos usando matplotlib 
st.subheader('Matploblib Chart')
fig_mpl, ax_mpl = plt.subplots()
ax_mpl = plt.hist(trees_df['age'])
plt.xlabel('Age (Days)')
st.pyplot(fig_mpl)

## grafico de violin

agua = pd.read_csv("https://raw.githubusercontent.com/narencastellon/Serie-de-tiempo-con-Machine-Learning/main/Data/aguacate.csv", parse_dates= ["Date"])

st.header("Gráfico de Violin")
agua["year"] =  agua["Date"].dt.year
fig = plt.figure(figsize=(18, 5))
sns.violinplot(x = "year", y = "AveragePrice", data = agua)
st.pyplot(fig)

st.header("Gráfico de Correlacion Mapa de Calor")
fig = plt.figure(figsize=(18, 6))  # Adjust the figure size if needed
sns.heatmap(agua.corr(numeric_only = True), annot=True, fmt=".2f", cmap="YlGnBu")
st.pyplot(fig)

# mapas usando la funcion map

trees_df = trees_df.dropna(subset=['longitude', 'latitude'])
trees_df = trees_df.sample(n = 1000)
st.map(trees_df)

# Grafico con Altair

st.title('SF Trees')
st.write(
    """Esta aplicación analiza árboles en San Francisco utilizando un conjunto de datos proporcionado amablemente por SF DPW"""
)

df_caretaker = trees_df.groupby(['caretaker']).count()['tree_id'].reset_index()
df_caretaker.columns = ['caretaker', 'tree_count']
fig = alt.Chart(df_caretaker).mark_bar().encode(x = 'caretaker', y ='tree_count')
st.altair_chart(fig)

# Visualzando pydeck
st.header("Visualizando con Pydeck")
sf_initial_view = pdk.ViewState(
     latitude=37.77,
     longitude=-122.4
     )
st.pydeck_chart(pdk.Deck(
     initial_view_state=sf_initial_view
     ))

# agregando estilo
sf_initial_view = pdk.ViewState(
     latitude=37.77,
     longitude=-122.4,
     zoom=9
     )
st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=sf_initial_view,
     ))

# 
trees_df.dropna(how='any', inplace=True)
sf_initial_view = pdk.ViewState(
     latitude=37.77,
     longitude=-122.4,
     zoom=11
     )
sp_layer = pdk.Layer(
     'ScatterplotLayer',
     data = trees_df,
     get_position = ['longitude', 'latitude'],
     get_radius=30)
st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=sf_initial_view,
     layers = [sp_layer]
))

# 
trees_df.dropna(how='any', inplace=True)
sf_initial_view = pdk.ViewState(
     latitude=37.77,
     longitude=-122.4,
     zoom=11,
     pitch=30
     )
hx_layer = pdk.Layer(
     'HexagonLayer',
     data = trees_df,
     get_position = ['longitude', 'latitude'],
     radius=100,
     extruded=True)
st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=sf_initial_view,
     layers = [hx_layer]
))

# 
import graphviz as graphviz
st.title('Graphviz')
# Creating graph object
st.graphviz_chart('''
                  digraph {
                  "Training Data" -> "ML Algorithm"
                  "ML Algorithm" -> "Model"
                  "Model" -> "Result Forecasting"
                  "New Data" -> "Model"}
''')

# Otra forma de visualizar
st.title('Otra forma de Visualizar con Graphviz')
# Create a graphlib graph object
graph = graphviz.Digraph()
graph.edge('Training Data', 'ML Algorithm')
graph.edge('ML Algorithm', 'Model')
graph.edge('Model', 'Result Forecasting')
graph.edge('New Data', 'Model')
st.graphviz_chart(graph)