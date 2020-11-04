from sklearn import tree
import pandas 

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
'font.size': 20,
'figure.figsize': [12, 8],
'figure.autolayout': True,
'font.family': 'serif',
'font.sans-serif': ['Palatino']})

# En primer lugar, a partir de la variable Sales creen una nueva variable binaria que se llame High, que sea Yes si Sales es mayor a 8 o No si es menor que 8.  A partir de ahí tienen que  usar  árboles  de  decisión  para  estimar High con  clasificación  y Sales con  regresión(importante: tomar la precaución de sacar una al estimar la otra, ya que si no tendrán errores artificialmente bajos)

#"Sales","CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"

# (a)  Separar los datos en dos particiones, una para datos de entrenamiento/validación (osea, de desarrollo) y otra para test.
def data_loading():
    df = pandas.read_csv("./Carseats.csv", header='infer')
    df = df.to_numpy()
    sales = df[:,0]
    print(sales[:10])
    print(type(sales))
    print(sales.shape)
    
    high = np.heaviside(sales, 0)
    print(sales[:10], high[:10] )
    
    
data_loading()
    

def split_data():
    pass

# (b)  Entrenar un árbol de decisión para clasificación de la variableHigh. Hacer un plot delárbol (usarscikit-learn) e interpretar los resultados.

def train_tree_high():
    pass
# (c)  Entrenar un árbol de decisión para regresión de la variableSales.  Hacer un plot delárbol (usarscikit-learn) e interpretar los resultados.

def  train_tree_Sales():
    pass

# (d)  ¿Cuál es el error de test que obtienen en cada caso?  Comparar con el error de entrenamiento y determinar si tienen overfitting o no.



# (e)  Para el árbol de regresión, usarcross-validationpara determinar el nivel óptimo de com-plejidad del árbol.  Busquen cómo usar enscikit-learnla técnica depruningparamejorar la tasa de error de test.

# (f)  Para el caso de regresión,  usar el abordaje tipobaggingpara mejorar el error de test.Comparar con el abordaje de un único árbol de decisión.  Buscar enscikit-learncómo determinar el orden de importancia de los atributos.

# (g)  Usarrandom forestspara mejorar los resultados dados. Comparar el error de test con los abordajes anteriores.  ¿Cambia el orden de la importancia de los atributos?  Hacer unplot con el error de test en función del del hiperparámetromax_featuresque limitael número de atributos a incluir en cada split.  Hacer otro plot equivalente en funcióndemax_depth.1


# (h)  Hacer la misma regresión usandoAdaBoosty comparar errores de test con lo obtenidoconRandom Foresten el punto anterior.2