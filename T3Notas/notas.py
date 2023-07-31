import numpy as np
import tensorflow as tf
from tensorflow import keras

# ingresar los datos
notas = [4.5, 3.0, 2.5, 4.2, 4.0, 3.5, 4.3, 3.8, 4.5, 4.0]
asistencia = [12, 18, 7, 15, 3, 19, 15, 2, 18, 15]
gana_semestre = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]

# convertir las listas de datos en un arreglo de NumPy
notas = np.array(notas)
asistencia = np.array(asistencia)
gana_semestre = np.array(gana_semestre)

# combinar las características en una matriz de características
X = np.column_stack((notas, asistencia))
y = gana_semestre

# definir el modelo de red neuronal
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ajustar el modelo a los datos
model.fit(X, y, epochs=100)

# hacer predicciones
nota = float(input("Ingrese la nota del alumno: "))
asistencias = int(input("Ingrese la cantidad de asistencias del alumno: "))
prediction = model.predict(np.array([[nota, asistencias]]))

# mostrar el resultado
if prediction > 0.5:
    print("El alumno ha ganado el semestre.")
else:
    print("El alumno ha perdido el semestre.")

