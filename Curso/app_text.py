# instalar el framework streamlit

# pip install streamlit

# como vamos a ejecutar nuestra aplicacion 
# streamlit run app_text.py
# Cargamos la libreria

import streamlit as st
import pandas as pd
# Crear titulos 

st.title("Nuestra primera aplicación")

#st.title("Nuestra primera aplicación1", anchor="Apress")

# Crear en cabezado "Hearders"
#st.Hearders("encabezado")
st.header("""Nuestro primer encabezado""")

# sub encabezado
st.subheader("Sub encabezado")

#Caption
st.caption("Este es nuestro primer Caption")

# Markdown

st.markdown("# Hi,\n# ***People*** \t!!!!!!!!!")

st.markdown("Streamlit es un marco de Python de código abierto para que los científicos de datos y los ingenieros de IA/ML entreguen aplicaciones de datos dinámicos con solo unas pocas líneas de código. Crea e implementa potentes aplicaciones de datos en minutos. ¡Empecemos!")

st.markdown("# Este es un ejemplo de Markdown usando el Framework de Streamlit")

st.markdown(" Este es un ejemplo de Markdown usando el Framework de Streamlit")

st.markdown("### Este es un ejemplo de Markdown usando el Framework de Streamlit")

# usar latex

st.markdown("## Estamos agregando ecuaciones matematicas usando la función de Latex")
st.latex(r'''cos2\theta = 1 - 2sin^2\theta''')
st.latex("""(a+b)^2 = a^2 + b^2 + 2ab""")
st.latex(r'''\frac{\partial u}{\partial t} = h^2 \left( \frac{\partial^2 u}{\partial x^2}
+ \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right)''')


# Codigos 

# Displaying Python Code
st.subheader("""Codigo de Python""")

code = '''def hello(): print("Hello, Streamlit!")'''

st.code(code, language='python')

# Displaying Java Code
st.subheader("""Java Code""")
st.code("""public class GFG {public static void main(String args[])
{System.out.println("Hello World");}
}""", language='javascript')


# JavaScript
st.subheader("""JavaScript Code""")
st.code(""" <p id="demo"></p>
<script>
try {adddlert("Welcome guest!");
}
catch(err) {document.getElementById("demo").innerHTML = err.message;
}
</script> """)

# Carga tabla a nuestra aplicación 

st.header("Mostrando DataFrame")

df = pd.read_csv("https://raw.githubusercontent.com/narencastellon/Serie-de-tiempo-con-Machine-Learning/main/Data/Adidas%20US%20Sales%20Data.csv", sep = ";")

st.dataframe(df.head())

