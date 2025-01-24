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

#Preparar los datos
X = data_df[['sepal length (cm)', 'petal length (cm)']].values
y = data_df['target'].values

#Dividir los datos en conjuntos de entrenamiento y prueba
#random_state = 42 es una semilla para que los datos se dividan de manera aleatoria
#Pero si el valor de random_state es el mismo, los datos se dividirán de la misma manera
#Si se cambia el valor de random_state, los datos se dividirán de manera diferente o si no se pone random_state.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) #20% de los datos se usarán para pruebas y el otro 80% para entrenamiento

#Inicializar y entrenar el perceptrón
perceptron = Perceptron(max_iter = 10, eta0 = 0.09) #max_iter es el número de iteraciones y eta0 es la tasa de aprendizaje
perceptron.fit(X_train, y_train) #Entrenar el perceptrón con los datos de entrenamiento para ajustar los pesos y el bias

#Obtener los pesos y bias del perceptrón
weights = perceptron.coef_[0]
bias = perceptron.intercept_[0]

print(f'Pesos: {weights}')
print(f"Bias: {bias}")

#Predecir en el conjunto de prueba
y_pred = perceptron.predict(X_test)

print(y_pred)
print(y_test)

#Probar el modelo en un dato de prueba
new_data = np.array([4.9, 1.4]).reshape(1, -1) #reshape(1, -1) es para convertir el dato en un arreglo de una sola fila y n columnas
print(f'Predicción: {perceptron.predict(new_data)}')

#Calcular la precisión o exactitud del modelo
accuracy = accuracy_score(y_pred, y_test)
print(f'Precisión: {accuracy:.2f}')

#Dibujar la linea de separación
x_values = np.linspace(4, 7.5, 100) #Generar 100 valores entre 4 y 7.5 para la línea de separación respecto al valor mínimo y máximo de sepal length (cm)
y_values = -(weights[0] * x_values / weights[1]) - bias / weights[1] #Ecuación de la línea de separación (Ecuación de la recta)
plt.plot(x_values, y_values, '-r', label = 'Linea de separación')

#Mostrar la gráfica
plt.show()