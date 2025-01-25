import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential # crear el modelo de la red capa x capa de forma lineal
from tensorflow.keras.layers import Input, Dense # Input -> Define la forma de los datos de entrada, Dense -> Referencia al uso de capas densamente conectadas.

#Crear un rango de números para las tablas
numbers = np.arange(1, 101) #Genera un array de 1 a 100

#Generar las tablas de sumar, restar, multiplicar y dividir
data = {
    'Num_1': np.repeat(numbers, len(numbers)),
    'Num_2': np.tile(numbers, len(numbers)),
    "addition": np.repeat(numbers, len(numbers)) + np.tile(numbers, len(numbers)),
    "subtraction": np.repeat(numbers, len(numbers)) - np.tile(numbers, len(numbers)),
    "multiplication": np.repeat(numbers, len(numbers)) * np.tile(numbers, len(numbers)),
    "division": np.round(np.repeat(numbers, len(numbers)) / np.tile(numbers, len(numbers)), 2),
}

#Crear un DataFrame a partir de los datos
df = pd.DataFrame(data)

#Transformar en una única columna de resultados y una de operaciones (df.melt -> no se indica df en los atributos)
#(pd.melt -> se indica df en los atributos)
df = pd.melt(
    df,
    id_vars = ['Num_1', 'Num_2'],
    value_vars = ['addition', 'subtraction', 'multiplication', 'division'],
    var_name = 'Operation',
    value_name = 'Result'
)

#Añadir etiquetas descriptivas a las operaciones
operations_labes = {
    "addition": 0,
    "subtraction": 1,
    "multiplication": 2,
    "division": 3,
}

#Agregar columna con etiquetas descriptivas
df['Operation_label'] = df['Operation'].map(operations_labes)
#print(df.head(-1))

#A explorar el dataset que hemos creado
elements_operation = df['Operation'].value_counts()
stats_data = df.describe()

#print(elements_operation)
#print(stats_data)

#A separar las características de la variable objetivo
x = df[['Num_1', 'Num_2', 'Result']].values #Entradas de la red neuronal
y = df['Operation_label'].values #Etiquetas de salida

#print(x.shape, y.shape)
#print(x)
#print(y)

#Dividir el dataset en entranamiento y validación (sklearn -> train_test_split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Construir la arquitectura de la red neuronal
model = Sequential([
    Input(shape = (3,)), #Entradas de la red neuronal (es una tupla de 3 elementos)
    Dense(16, activation = 'relu'), #Primera capa densamente conectada
    Dense(16, activation = 'relu'), #Segunda capa densamente conectada
    Dense(4, activation = 'softmax') #Capa de salida
])

#model.summary()

#Compilar el modelo
#Optimizer ajustar los pesos para minimizar el error
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#Entrenar el modelo
#batch_size -> número de ejemplos que se procesan antes de actualizar los pesos
#validation_data -> datos de validación
history = model.fit(
    x_train,
    y_train,
    epochs = 50,
    batch_size = 32,
    validation_data = (x_test, y_test)
)

#Graficamos la información de los datos de perdida de entrenamiento y validación
arrayTrainDataLoss = np.arange(0, len(history.history['loss']))
arrayTestDataLoss = np.arange(0, len(history.history['val_loss']))

plt.style.use('ggplot')
plt.figure(figsize = (6, 4))

plt.plot(arrayTrainDataLoss, history.history['loss'], label = 'Training data loss')
plt.plot(arrayTestDataLoss, history.history['val_loss'], label = 'Validation data loss')

plt.title('Training and validation data loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()

#Graficamos la información de los datos de precisión de entrenamiento y validación
arrayTrainDataAccuracy = np.arange(0, len(history.history['accuracy']))
arrayTestDataAccuracy = np.arange(0, len(history.history['val_accuracy']))

plt.style.use('ggplot')
plt.figure(figsize = (6, 4))

plt.plot(arrayTrainDataAccuracy, history.history['accuracy'], label = 'Training data accuracy')
plt.plot(arrayTestDataAccuracy, history.history['val_accuracy'], label = 'Validation data accuracy')

plt.title('Training and validation data accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#Evaluar el modelo con los datos de prueba y de entrenamiento
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)

#A hacer predicciones con el modelo entrenado
#Datos para predecir
data_pred = np.array([
    [99, 5, 19.8],
    [10, 10, 0],
    [2, 2, 4],
    [1000, -2, 998],
    [-10, 2, -8]
])

#Predecir con el modelo entrenado
predictions = model.predict(data_pred)
predictions = np.round(predictions)
print(predictions)

#Obtener las etiquetas de las predicciones
predicted_classes = np.argmax(predictions, axis = 1)
print(predicted_classes)

#Mapa de etiquetas para las operaciones
map_tags_operations = {
    0: 'Addition',
    1: 'Subtraction',
    2: 'Multiplication',
    3: 'Division'
}

#Mostrar las predicciones
for i, (inputs, pred_class) in enumerate(zip(data_pred, predicted_classes)):
    op = map_tags_operations[pred_class]
    print(f'Para los números {inputs[0]} y {inputs[1]} con resultado {inputs[2]} -> {op}')