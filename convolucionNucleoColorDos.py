import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 1. Cargar una imagen real desde la computadora
# RUTA DE LA IMAGEN: Coloca aquí el nombre de la foto que quieras procesar.
nombre_archivo = "imagenes/mio.jpg" 

# Comprobamos si el archivo existe antes de cargarlo
if not os.path.exists(nombre_archivo):
    print(f"❌ ERROR: No se encontró la imagen '{nombre_archivo}' en esta carpeta.")
    print("👉 Por favor, guarda una imagen en la misma carpeta o cambia el nombre en la línea 7.")
    # Si no hay foto real, creamos una de prueba automáticamente para que el código no falle
    imagen_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(imagen_bgr, "Pon tu foto!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
else:
    # Cargamos la imagen real usando OpenCV
    imagen_bgr = cv2.imread(nombre_archivo)

# Convertimos la imagen a tipo flotante (float32) para poder hacer las operaciones matemáticas
imagen_procesar = imagen_bgr.astype(np.float32)

# 2. Definir los filtros matemáticos y el sesgo para cada canal de color
# 👁️ ¡Aquí los chicos pueden experimentar cambiando los números de cada matriz!
# 2. Definir los filtros matemáticos con Efecto Neón Eléctrico
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

# Deja los sesgos en 0 para mantener el fondo oscuro y que resalte el neón
sesgo_azul = 0.0
sesgo_verde = 0.0
sesgo_rojo = 0.0

# CONTROL DEL SESGO INDEPENDIENTE: Suma o resta luz (pueden experimentar con estos valores)
sesgo_azul = 0.0
sesgo_verde = 0.0
sesgo_rojo = 0.0

# 3. Aplicar la convolución canal por canal usando OpenCV
# OpenCV separa en orden BGR (Azul, Verde, Rojo)
canal_azul, canal_verde, canal_rojo = cv2.split(imagen_procesar)

# Convolucionamos cada canal de forma independiente agregando su propio sesgo
res_azul = cv2.filter2D(canal_azul, -1, filtro_azul) + sesgo_azul
res_verde = cv2.filter2D(canal_verde, -1, filtro_verde) + sesgo_verde
res_rojo = cv2.filter2D(canal_rojo, -1, filtro_rojo) + sesgo_rojo

# 4. Asegurar que los números se mantengan en los límites válidos de color (0 a 255)
res_azul = np.clip(res_azul, 0, 255)
res_verde = np.clip(res_verde, 0, 255)
res_rojo = np.clip(res_rojo, 0, 255)

# 🔄 ¡LA MAGIA AQUÍ!: Volvemos a fusionar los 3 canales procesados en una sola imagen a color
resultado_bgr = cv2.merge([res_azul, res_verde, res_rojo])
resultado_bgr = resultado_bgr.astype(np.uint8) # Convertimos de vuelta a formato de imagen estándar

# 5. Graficar y comparar los resultados usando Matplotlib
plt.figure(figsize=(12, 6))

# Convertimos ambas imágenes de BGR a RGB para que Matplotlib muestre los colores correctamente
imagen_rgb_original = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
imagen_rgb_procesada = cv2.cvtColor(resultado_bgr, cv2.COLOR_BGR2RGB)

# Subplot 1: Mostrar la fotografía original
plt.subplot(1, 2, 1)
plt.imshow(imagen_rgb_original)
plt.title("Fotografía Original de Entrada")
plt.axis('off')

# Subplot 2: Mostrar el resultado final procesado a todo color
plt.subplot(1, 2, 2)
plt.imshow(imagen_rgb_procesada)
plt.title("Imagen Convolucionada a Color\n(Canales fusionados de nuevo)")
plt.axis('off')

plt.tight_layout()
plt.show()
