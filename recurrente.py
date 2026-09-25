# ==========================================
# RED NEURONAL RECURRENTE (RNN)
# Predicción de secuencias numéricas
# ==========================================

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ==========================================
# 1. GENERAR DATOS
# ==========================================

X = []
y = []

# Crear secuencias:
# [1,2,3,4,5] -> 6
# [2,3,4,5,6] -> 7
# ...

for i in range(1, 100):

    secuencia = [i, i+1, i+2, i+3, i+4]

    X.append(secuencia)
    y.append(i+5)

# Convertir a arreglos NumPy
X = np.array(X)
y = np.array(y)

print("Forma original de X:", X.shape)
print("Forma original de y:", y.shape)

# ==========================================
# 2. ADAPTAR DATOS PARA LA RNN
# ==========================================

# Las RNN esperan:
# (muestras, pasos_de_tiempo, características)

X = X.reshape((X.shape[0], X.shape[1], 1))

print("\nForma después del reshape:")
print(X.shape)

# ==========================================
# 3. CREAR EL MODELO
# ==========================================

model = Sequential()

# Capa RNN
model.add(
    SimpleRNN(
        units=20,
        activation='tanh',
        input_shape=(5,1)
    )
)

# Capa de salida
model.add(Dense(1))

# Mostrar arquitectura
model.summary()

# ==========================================
# 4. COMPILAR MODELO
# ==========================================

model.compile(
    optimizer='adam',
    loss='mse'
)

# ==========================================
# 5. ENTRENAR
# ==========================================

history = model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)

# ==========================================
# 6. VISUALIZAR ERROR
# ==========================================

plt.figure(figsize=(8,4))

plt.plot(history.history['loss'])

plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Loss (MSE)")
plt.grid(True)

plt.show()

# ==========================================
# 7. REALIZAR PREDICCIONES
# ==========================================

print("\n==============================")
print("PRUEBAS DE LA RNN")
print("==============================")

# Prueba 1
prueba1 = np.array([[101,102,103,104,105]])
prueba1 = prueba1.reshape((1,5,1))

pred1 = model.predict(prueba1)

print("\nSecuencia:")
print([101,102,103,104,105])

print("Predicción:")
print(pred1[0][0])

print("Valor esperado:")
print(106)

# ------------------------------------------

# Prueba 2

prueba2 = np.array([[201,202,203,204,205]])
prueba2 = prueba2.reshape((1,5,1))

pred2 = model.predict(prueba2)

print("\nSecuencia:")
print([201,202,203,204,205])

print("Predicción:")
print(pred2[0][0])

print("Valor esperado:")
print(206)

# ==========================================
# 8. PRUEBA INTERACTIVA
# ==========================================

print("\n==============================")
print("PRUEBA PERSONALIZADA")
print("==============================")

entrada = input(
    "Escribe 5 números separados por comas (ej: 10,11,12,13,14): "
)

valores = [float(x) for x in entrada.split(",")]

datos = np.array([valores])
datos = datos.reshape((1,5,1))

prediccion = model.predict(datos)

print("\nLa RNN predice:")

print(round(float(prediccion[0][0]), 2))
