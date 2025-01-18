import streamlit as st 
import pandas as pd 
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

# Carga base de datos

df =  pd.read_csv("https://raw.githubusercontent.com/narencastellon/Serie-de-tiempo-con-Machine-Learning/main/Data/Adidas%20US%20Sales%20Data.csv", sep = ";")

#Eliminamos el signo de dólar y el espacio de la coma.
df['Price per Unit'] = df['Price per Unit'].str.replace('$', '')
df['Total Sales'] = df['Total Sales'].str.replace('$', '').str.replace(',', '')
df['Operating Profit'] = df['Operating Profit'].str.replace('$', '').str.replace(',', '')
df["Units Sold"]=df["Units Sold"].str.replace(',', '')

# Quita el signo % y divide por 100.
df['Operating Margin']=df['Operating Margin'].str[:-1].astype(float)
df['Operating Margin'] = df['Operating Margin'] / 100

df['Invoice Date']=pd.to_datetime(df['Invoice Date']) #Cambiar el tipo de datos de la fecha de la factura a fecha y hora

df[['Price per Unit', 'Units Sold', 'Total Sales','Operating Profit']] = df[['Price per Unit', 'Units Sold', 'Total Sales','Operating Profit']].astype("float")

st.header("Nos muestra la data")

# Cargamos la base de datos

st.dataframe(df)fghiop

# Crear 3 columnas

col1, col2, col3 = st.columns(3)

# Display data in the first column
with col1:
    st.header("Column 1")
    st.dataframe(df[["Price per Unit", "Units Sold"]].style.highlight_min(axis=0))  # Display specific columns

with col2:
    st.header("columna 2")
    st.dataframe(df.describe().T)

with col3:
    st.header("columnas 3")
    st.dataframe(df.head())

# Funcion Table

st.header("Función Table")
#st.table(df)

# Función Metric
#Defining Columns
c1, c2, c3 = st.columns(3)
# Defining Metrics
c1.metric("Rainfall", "100 cm", "10 cm")
c2.metric(label="Population", value="123 Billions", delta="1 Billions", delta_color="inverse")
c3.metric(label="Customers", value=100, delta=10, delta_color="off")
st.metric(label="Speed", value=None, delta=0)
st.metric("Trees", "91456", "-1132649")

# json

#Defining Nested JSON
st.json
(
    { "Books" : [{"BookName" : "Python Testing with Selenium",
                "BookID" : "1",
                "Publisher" : "Apress",
                "Year" : "2021",
                "Edition" : "First",
                "Language" : "Python",
},
{
"BookName": "Beginners Guide to Streamlit with Python",
"BookID" : "2",
"Publisher" : "Apress",
"Year" : "2022",
"Edition" : "First",
"Language" : "Python"
}]}
)


# Write 

st.write('Aquí están nuestros datos', df.head(), 'Los datos están en formato de DataFrame.\n', "\n `Write` es una súper función")

# Defining random Values for Chart
data = pd.DataFrame(np.random.randn(10, 2),columns=['a', 'b'])

col1, col2 = st.columns(2)

with col1:
    st.header("Grafico barras")
    chart = alt.Chart(data).mark_bar().encode(x='a', y='b',tooltip=['a', 'b'])
    st.write(chart)

st.header("Grafico - Usando las función Magic")
chart, ax = plt.subplots()
ax.hist(df["Sales Method"], bins=15)
# Magic chart
"chart", chart