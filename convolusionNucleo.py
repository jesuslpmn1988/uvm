import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. Crear una imagen completamente NEGRA (Valores en 0)
# Tamaño: 300x300 píxeles, con 3 canales de color (RGB)
altura, anchura = 300, 300
imagen_negra = np.zeros((altura, anchura, 3), dtype=np.float32)

# Dibujamos figuras para probar los filtros en la oscuridad
# Un cuadrado en el canal Rojo (BGR en OpenCV: el tercer elemento es Rojo)
cv2.rectangle(imagen_negra, (30, 30), (120, 120), (0, 0, 255), -1)
# Un círculo en el canal Verde
cv2.circle(imagen_negra, (200, 80), 45, (0, 255, 0), -1)
# Una línea gruesa en el canal Azul
cv2.line(imagen_negra, (50, 200), (250, 250), (255, 0, 0), 10)

# 2. Definir los filtros matemáticos para cada canal de forma correcta
filtro_rojo = np.eye(3, dtype=np.float32)

filtro_verde = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

filtro_azul = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# 3. Aplicar la convolución canal por canal usando OpenCV
# Separamos los canales independientes (Azul, Verde, Rojo)
canal_azul, canal_verde, canal_rojo = cv2.split(imagen_negra)

# Convolucionamos cada canal con su respectiva matriz
res_rojo = cv2.filter2D(canal_rojo, -1, filtro_rojo)
res_verde = cv2.filter2D(canal_verde, -1, filtro_verde)
res_azul = cv2.filter2D(canal_azul, -1, filtro_azul)

# 4. Combinación lineal matemática: rojo + verde + azul + sesgo
sesgo = 0.0
resultado_final = res_rojo + res_verde + res_azul + sesgo

# 5. Graficar y comparar los resultados usando Matplotlib
plt.figure(figsize=(12, 5))

# Convertimos de BGR (OpenCV) a RGB para que Matplotlib muestre bien los colores
imagen_rgb = cv2.cvtColor(imagen_negra.astype(np.uint8), cv2.COLOR_BGR2RGB)

# Subplot 1: Imagen Original
plt.subplot(1, 2, 1)
plt.imshow(imagen_rgb)
plt.title("Imagen Original (Fondo Negro)")
plt.axis('off')

# Subplot 2: Resultado de la Convolución
plt.subplot(1, 2, 2)
plt.imshow(resultado_final, cmap='gray')
plt.title("Mapa de Características Resultante\n(rojo + verde + azul + sesgo)")
plt.axis('off')

plt.tight_layout()
plt.show()
