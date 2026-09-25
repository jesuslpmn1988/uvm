import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam

# ==========================================
# PARAMETROS
# ==========================================

LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 3000

# ==========================================
# CARGAR MNIST
# ==========================================

(X_train, _), (_, _) = mnist.load_data()

X_train = X_train.astype("float32")

# Normalizar [-1,1]
X_train = (X_train - 127.5) / 127.5

# (60000,28,28,1)
X_train = np.expand_dims(X_train, axis=-1)

print("Dataset cargado")
print(X_train.shape)

# ==========================================
# GENERADOR
# ==========================================

def crear_generador():

    model = Sequential()

    model.add(Dense(128, input_dim=LATENT_DIM))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dense(28 * 28, activation="tanh"))

    model.add(Reshape((28, 28, 1)))

    return model

# ==========================================
# DISCRIMINADOR
# ==========================================

def crear_discriminador():

    model = Sequential()

    model.add(Flatten(input_shape=(28, 28, 1)))

    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(0.0002),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ==========================================
# CONSTRUIR MODELOS
# ==========================================

generador = crear_generador()
discriminador = crear_discriminador()

# Congelar discriminador
discriminador.trainable = False

gan = Sequential([
    generador,
    discriminador
])

gan.compile(
    optimizer=Adam(0.0002),
    loss="binary_crossentropy"
)

print("\nGenerador:")
generador.summary()

print("\nDiscriminador:")
discriminador.summary()

# ==========================================
# GUARDAR IMAGENES
# ==========================================

def guardar_imagenes(epoca):

    ruido = np.random.normal(
        0,
        1,
        (25, LATENT_DIM)
    )

    imagenes = generador.predict(
        ruido,
        verbose=0
    )

    imagenes = 0.5 * imagenes + 0.5

    fig, axs = plt.subplots(
        5,
        5,
        figsize=(6,6)
    )

    contador = 0

    for i in range(5):
        for j in range(5):

            axs[i,j].imshow(
                imagenes[contador,:,:,0],
                cmap="gray"
            )

            axs[i,j].axis("off")

            contador += 1

    plt.suptitle(f"Epoca {epoca}")

    plt.tight_layout()

    plt.savefig(
        f"gan_epoca_{epoca}.png"
    )

    plt.close()

# ==========================================
# ENTRENAMIENTO
# ==========================================

for epoca in range(EPOCHS):

    # -----------------------------
    # IMAGENES REALES
    # -----------------------------

    idx = np.random.randint(
        0,
        X_train.shape[0],
        BATCH_SIZE
    )

    imagenes_reales = X_train[idx]

    # -----------------------------
    # IMAGENES FALSAS
    # -----------------------------

    ruido = np.random.normal(
        0,
        1,
        (BATCH_SIZE, LATENT_DIM)
    )

    imagenes_falsas = generador.predict(
        ruido,
        verbose=0
    )

    # Etiquetas

    reales = np.ones((BATCH_SIZE,1))
    falsas = np.zeros((BATCH_SIZE,1))

    # -----------------------------
    # ENTRENAR DISCRIMINADOR
    # -----------------------------

    perdida_real = discriminador.train_on_batch(
        imagenes_reales,
        reales
    )

    perdida_falsa = discriminador.train_on_batch(
        imagenes_falsas,
        falsas
    )

    perdida_d = 0.5 * np.add(
        perdida_real,
        perdida_falsa
    )

    # -----------------------------
    # ENTRENAR GENERADOR
    # -----------------------------

    ruido = np.random.normal(
        0,
        1,
        (BATCH_SIZE, LATENT_DIM)
    )

    objetivo = np.ones(
        (BATCH_SIZE,1)
    )

    perdida_g = gan.train_on_batch(
        ruido,
        objetivo
    )

    # -----------------------------
    # MOSTRAR AVANCE
    # -----------------------------

    if epoca % 500 == 0:

        print("\n--------------------------------")
        print("Epoca:", epoca)
        print("Loss Discriminador:",
              round(float(perdida_d[0]),4))
        print("Accuracy Discriminador:",
              round(float(perdida_d[1])*100,2),
              "%")
        print("Loss Generador:",
              round(float(perdida_g),4))
        print("--------------------------------")

        guardar_imagenes(epoca)

print("\nEntrenamiento finalizado.")
