import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential

# 1. Cargar una imagen real desde la computadora
nombre_archivo = "imagenes/mio.jpg" 

if not os.path.exists(nombre_archivo):
    print(f"❌ ERROR: No se encontró la imagen '{nombre_archivo}' en esta carpeta.")
    print("👉 Por favor, guarda una imagen en la misma carpeta o cambia el nombre en la línea 9.")
    imagen_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(imagen_bgr, "Pon tu foto!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
else:
    imagen_bgr = cv2.imread(nombre_archivo)

# Formato flotante y dimensiones para Keras (1, Alto, Ancho, Canales)
imagen_procesar = imagen_bgr.astype(np.float32)
imagen_keras = np.expand_dims(imagen_procesar, axis=0)

# 2. Definir las matrices de los filtros (Efecto Neón Eléctrico)
# 👁️ ¡Los chicos pueden cambiar estos números para experimentar!
filtro_rojo = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

filtro_verde = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

filtro_azul = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# 3. Preparar la estructura tridimensional de pesos que requiere Keras
# Formato Keras Conv2D: (Alto_Filtro, Ancho_Filtro, Canales_Entrada, Filtros_Salida)
pesos_convolucion = np.zeros((3, 3, 3, 3), dtype=np.float32)

# Mapeamos cada filtro a su canal correspondiente en el espacio BGR de OpenCV
pesos_convolucion[:, :, 0, 0] = filtro_azul   # Canal 0 (Azul)
pesos_convolucion[:, :, 1, 1] = filtro_verde  # Canal 1 (Verde)
pesos_convolucion[:, :, 2, 2] = filtro_rojo   # Canal 2 (Rojo)

# Sesgos individuales para cada canal de color (pueden sumarle brillo aquí)
sesgos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# 4. Construir el modelo Keras con la Capa Convolucional oficial
modelo = Sequential()
modelo.add(Conv2D(
    filters=3, 
    kernel_size=(3, 3), 
    input_shape=(imagen_bgr.shape[0], imagen_bgr.shape[1], 3),
    padding='same', 
    use_bias=True
))

# Inyectamos de forma manual nuestros núcleos y sesgos en la red neuronal
modelo.layers[0].set_weights([pesos_convolucion, sesgos])

# 5. Ejecutar la Convolución (Predicción de la red)
resultado_keras = modelo.predict(imagen_keras)

# Aseguramos que los valores sigan entre 0 y 255 tras el cálculo matemático
resultado_bgr = np.clip(resultado_keras[0], 0, 255).astype(np.uint8)

# 6. Graficar y comparar los resultados usando Matplotlib
plt.figure(figsize=(12, 6))

imagen_rgb_orig = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
imagen_rgb_proc = cv2.cvtColor(resultado_bgr, cv2.COLOR_BGR2RGB)

plt.subplot(1, 2, 1)
plt.imshow(imagen_rgb_orig)
plt.title("Fotografía Original de Entrada")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagen_rgb_proc)
plt.title("Procesado con Capa Keras Conv2D\n(Efecto Neón)")
plt.axis('off')

plt.tight_layout()
plt.show()
