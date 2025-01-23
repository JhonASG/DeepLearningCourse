import matplotlib.pyplot as plt
import numpy as np

#inputs compuerta logica AND
inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

#outputs compuerta logica AND
outputs = np.array([0, 0, 0, 1])
#outputs compuerta logica OR
#outputs = np.array([0, 1, 1, 1])
#outputs compuerta logica XOR - Esta compuerta no puede ser solucionada con un perceptrón
#outputs = np.array([0, 1, 1, 0])

#Visualización de los datos de entrada
plt.scatter(inputs[:, 0], inputs[:, 1], c = outputs, cmap = 'cool', marker = 'o', label = 'Datos de entrada', s = 150)
name_graph = 'Compuerta lógica AND'

#Configuración del gráfico
plt.title(name_graph)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)

learning_rate = 0.05 #Definir la taza de aprendizaje -> And: 0.05 - OR: 0.5 - XOR: 0.9
epochs = 20 #Epocas de entrenamiento
error_min = 0 #Error mínimo

#Función de activación
def activation_function(z):
    return 1 if z > 0 else 0

def predict(inputs, weights, bias):
    #Calcular la suma ponderada de la red neuronal
    z = np.dot(inputs, weights) + bias
    return activation_function(z) #La predicción de la red neuronal

#Función para entrenar el perceptrón
def train_perceptron(inputs, outputs, learning_rate, epochs):
    #Inicializar los pesos y el bias aleatoriamente.
    weights = np.random.rand(2) #Termina retornando dos números aleatorios
    bias = np.random.rand() #Termina retornando un número aleatorio

    #Iteraciones de entrenamiento
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}", end = ' ')
        total_error = 0 #Error total

        for input, label in zip(inputs, outputs):
            #Calcular la suma ponderada de la red neuronal
            z = np.dot(input, weights) + bias
            y_pred = activation_function(z) #La predicción de la red neuronal

            #Encontrar el error
            error = label - y_pred
            total_error += abs(error)

            #Actualizar los pesos y el bias
            delta_bias = learning_rate * error
            delta_weights = delta_bias * input

            weights = weights + delta_weights
            bias = bias + delta_bias
        
        average_error = total_error / len(inputs)
        print(f"Error promedio: {average_error}")
        if average_error <= error_min: break

    return weights, bias #Retornar los pesos y el bias ajustado.

weights_adjusted, bias_adjusted = train_perceptron(inputs, outputs, learning_rate, epochs)
print(f"Pesos: {weights_adjusted} Bias: {bias_adjusted}")

print(predict([0, 0], weights_adjusted, bias_adjusted)) #Predicción para [0, 0]
print(predict([0, 1], weights_adjusted, bias_adjusted)) #Predicción para [0, 1]
print(predict([1, 0], weights_adjusted, bias_adjusted)) #Predicción para [1, 0]
print(predict([1, 1], weights_adjusted, bias_adjusted)) #Predicción para [1, 1]

#Dibujar la linea de separación de los datos de entrada
x_values = np.linspace(0, 1.1, 100)
y_values = (-((weights_adjusted[0] / weights_adjusted[1]) * x_values) - (bias_adjusted / weights_adjusted[1]))
plt.plot(x_values, y_values, label = 'Separación lineal', color = 'red')

#Mostrar el gráfico
plt.legend()
plt.show()