# -*- coding: utf-8 -*-
"""Red Neuronal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1i3KldikmDMgmYz3nYqrM11UZEBPoieWH
"""

import tensorflow as tf
import numpy as np

#Haremos que una red neuronal averigue la forma de callular la conversió de grados celsius a fahrenheit
celsius = np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100],dtype=float)

#keras para simplificar redes neuronales
#CAPA DENSA: conexiones de una neurona a todas las demás
#Solo una neurona y
capa=tf.keras.layers.Dense(units=1,input_shape=[1])
#Entrenamiento con modelo secuencial
modelo = tf.keras.Sequential([capa])

#Agregando con más capas
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1,oculta2,salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),#Poco a poco mejorar con taza de aprendizaje
    loss = 'mean_squared_error'
)

print("Comenzando entrenamiento....")
#(Datos de entrada, resultados esperados, cuántas vueltas intentar,)
historial = modelo.fit(celsius, fahrenheit, epochs =1000,verbose=False)
print("Ya entrenó")

#La funcion de perdida dice que tan mal estan los resultados en cada vuelta que dio
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("Ahora la predicción")
resultado = modelo.predict([100.0])
print("El resultado es" + str(resultado)+"fahrenheit")

print("Variables internas del modelo")
print(capa.get_weights())

print("Variables internas del modelo")
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())