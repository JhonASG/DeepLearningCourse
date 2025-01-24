import numpy as np
import matplotlib.pyplot as plt

#Function de x^2
def f(x):
    return x**2

#Función de x^2 cos(x) - x
def fcos(x):
    return x**2 * np.cos(x) - x

#Derivada de x^2
def df(x):
    return 2*x

#Derivada de x^2 cos(x) - x
def dfcos(x):
    return 2*x*np.cos(x) - x**2 * np.sin(x) - 1

#Ecuación de la recta tangente - slope = pendiente
def tangent_line(x, x0, y0, slope):
    return slope * (x - x0) + y0

#Función del descenso de gradiente
def gradient_descent(x_point, learning_rate, num_iterations):
    points = [x_point]

    for _ in range (num_iterations):
        gradient = df(x_point)
        x_point = x_point - learning_rate * gradient
        points.append(x_point)
    
    return points

#Obtener los valores de x
x = np.linspace(-10, 10, 100)
f_x = f(x)

#Punto inicial
x_ini = -6.5
f_x_ini = f(x_ini)
slope = df(x_ini) #Pendiente se obtiene con la derivada

#Linea tangente al punto inicial
tangent = tangent_line(x, x_ini, f_x_ini, slope)

#Gradiente descendente
points = gradient_descent(x_ini, learning_rate = 0.125, num_iterations = 80)
last_point = points[-1]
last_point_f = f(last_point)
print(f'last point: {last_point}, f(last point): {last_point_f}')

#Obtener los valores para la función x^2 cos(x) - x
x_cos = np.linspace(-10, 10, 100)
f_cosx = fcos(x_cos)

#Punto inicial
x_ini_cos = 7.25
f_x_ini_cos = fcos(x_ini_cos)
slope_cos = dfcos(x_ini_cos) #Pendiente se obtiene con la derivada

#Linea tangente al punto inicial
tangent_cos = tangent_line(x_cos, x_ini_cos, f_x_ini_cos, slope_cos)

#Gradiente descendente
points_cos = gradient_descent(x_ini_cos, learning_rate = 0.05, num_iterations = 80)
last_point_cos = points_cos[-1]
last_point_f_cos = fcos(last_point_cos)
print(f'slope cos: {slope_cos}')
print(f'last point cos: {last_point}, f(last point) cos: {last_point_f}')
#Obtener los valores para la función x^2 cos(x) - x

#Visualización de la gráfica para x^2
plt.plot(x, f_x, label='f(x) = x^2')
plt.plot(x_ini, f_x_ini, 'o', label='Punto inicial', markersize = 12)
plt.plot(x, tangent, 'g--', label='Recta Tangente') #g-- indica que es una linea verde entrecortada.
plt.plot(points, [f(x) for x in points], 'ro-', label = 'Descenso de gradiente') #[f(x) for x in points] es para obtener los valores de x^2 para los puntos obtenidos en gradient_descent
plt.plot(last_point, last_point_f, 'bo', label='Punto final', markersize = 2) #Graficar el último punto obtenido en gradient_descent
plt.title('Descenso de gradiente')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(-10, 10)
plt.ylim(-10, 100)
plt.legend()
plt.grid(True)
plt.show()

#Visualización de la gráfica para x^2 cos(x) - x
plt.plot(x_cos, f_cosx, label='f(x) = x^2 cos(x) - x')
plt.plot(x_ini_cos, f_x_ini_cos, 'o', label = 'Punto inicial', markersize = 12)
plt.plot(x_cos, tangent_cos, 'g--', label='Recta Tangente')
plt.plot(points_cos, [fcos(x) for x in points_cos], 'ro-', label = 'Descenso de gradiente')
plt.plot(last_point_cos, last_point_f_cos, 'bo', label='Punto final', markersize = 2)
plt.title('Descenso de gradiente')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(-120, 100)
plt.legend()
plt.grid(True)
plt.show()