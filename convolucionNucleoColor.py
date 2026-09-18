import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 1. Cargar una imagen real desde la computadora
# RUTA DE LA IMAGEN: Coloca aquí el nombre de la foto que quieras procesar.
# Tip: Asegúrate de guardar la foto dentro de la misma carpeta "Convolucion" donde está este script.
nombre_archivo = "imagenes/mio.jpg" 

# Comprobamos si el archivo existe antes de cargarlo
if not os.path.exists(nombre_archivo):
    print(f"❌ ERROR: No se encontró la imagen '{nombre_archivo}' en esta carpeta.")
    print("👉 Por favor, guarda una imagen en la misma carpeta o cambia el nombre en la línea 8.")
    # Si no hay foto real, creamos una de prueba automáticamente para que el código no falle
    imagen_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(imagen_bgr, "Pon tu foto!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
else:
    # Cargamos la imagen real usando OpenCV
    imagen_bgr = cv2.imread(nombre_archivo)

# Convertimos la imagen a tipo flotante (float32) para poder hacer las operaciones matemáticas
imagen_procesar = imagen_bgr.astype(np.float32)

# 2. Definir los filtros matemáticos y el sesgo para que los chicos experimenten
# 👁️ ¡Aquí es donde los chicos pueden cambiar los números para ver efectos diferentes!
filtro_rojo = np.eye(3, dtype=np.float32) # Filtro Identidad (deja pasar el canal igual)

filtro_verde = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32) # Filtro de Desenfoque (Box Blur)

filtro_azul = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32) # Filtro Laplaciano (Detección de bordes)

# CONTROL DEL SESGO: Suma o resta luz al mapa final
sesgo = 0.0  

# 3. Aplicar la convolución canal por canal usando OpenCV
# OpenCV lee por defecto en orden BGR (Azul, Verde, Rojo)
canal_azul, canal_verde, canal_rojo = cv2.split(imagen_procesar)

# Convolucionamos cada canal de forma independiente con su matriz
res_rojo = cv2.filter2D(canal_rojo, -1, filtro_rojo)
res_verde = cv2.filter2D(canal_verde, -1, filtro_verde)
res_azul = cv2.filter2D(canal_azul, -1, filtro_azul)

# 4. Combinación lineal matemática completa: rojo + verde + azul + sesgo
resultado_final = res_rojo + res_verde + res_azul + sesgo

# Clip_by_value opcional: asegura que los píxeles no se desborden de los límites normales (0-255)
resultado_final = np.clip(resultado_final, 0, 255)

# 5. Graficar y comparar los resultados usando Matplotlib
plt.figure(figsize=(12, 6))

# Convertimos la imagen original de BGR a RGB para que Matplotlib muestre los colores reales correctamente
imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

# Subplot 1: Mostrar la fotografía original
plt.subplot(1, 2, 1)
plt.imshow(imagen_rgb)
plt.title("Fotografía Original de Entrada")
plt.axis('off')

# Subplot 2: Mostrar el Mapa de Características procesado
plt.subplot(1, 2, 2)
plt.imshow(resultado_final, cmap='gray')
plt.title("Mapa de Características Resultante\n(rojo + verde + azul + sesgo)")
plt.axis('off')

plt.tight_layout()
plt.show()
