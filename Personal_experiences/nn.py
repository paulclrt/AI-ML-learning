import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Charger le dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données
x_train, x_test = x_train / 255.0, x_test / 255.0

# Construire le modèle
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Transformer l'image 28x28 en un vecteur de 784 éléments
    layers.Dense(128, activation='relu'),  # Couche cachée avec 128 neurones et activation ReLU
    layers.Dropout(0.2),                   # Dropout pour éviter le surapprentissage
    layers.Dense(10, activation='softmax') # Couche de sortie avec 10 neurones (classes)
])


# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Évaluer le modèle sur les données de test
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Précision sur les données de test : {test_acc:.2f}')

# Visualiser l'évolution de la précision et de la perte
plt.plot(history.history['accuracy'], label='Précision d’entraînement')
plt.plot(history.history['val_accuracy'], label='Précision de validation')
plt.xlabel('Épochs')
plt.ylabel('Précision')
plt.legend()
plt.show()

# Sauvegarder le modèle dans un fichier ou un dossier
model.save('mon_modele.h5')  # Sauvegarde au format HDF5
# OU
# model.save('mon_modele')  # Sauvegarde au format TensorFlow (dossier)
