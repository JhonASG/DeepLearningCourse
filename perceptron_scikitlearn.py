from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the iris dataset
iris = load_iris()

#Leer el dataset en forma de DataFrame
data_df = pd.DataFrame(iris.data, columns = iris.feature_names)
data_df['target'] = iris.target

#Tomar solo ciertas columnas
#columns = ['sepal length (cm)', 'sepal width (cm)', 'target']
#data_df = data_df[columns]

#Filtrar solo vetosa y versicolor
data_df = data_df[data_df['target'].isin([0, 1])] #isin solo toma los elementos que contengan los valores de la lista

#data_df.head(5) ver de arriba hacia abajo
#data_df.tail(5) ver de abajo hacia arriba
print(data_df.head(5))

#Visualización de los datos
plt.figure()

#Datos de iris setosa
setosa_data = data_df[data_df['target'] == 0]

#Datos de iris versicolor
versicolor_data = data_df[data_df['target'] == 1]

#Gráfica de los datos
plt.scatter(setosa_data['sepal length (cm)'], setosa_data['petal length (cm)'], label='Setosa', color='cyan', s = 50)
plt.scatter(versicolor_data['sepal length (cm)'], versicolor_data['petal length (cm)'], label='Versicolor', color='magenta', s = 50)

#Personalizar la gráfica
plt.title('Sepal length vs Petal length')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Petal length (cm)')
plt.grid(True)
plt.legend()

#Mostrar la gráfica
plt.show()