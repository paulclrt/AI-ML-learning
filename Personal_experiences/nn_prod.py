from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Charger le modèle
model = keras.models.load_model('mon_modele.h5')


# Fonction pour charger, normaliser et afficher la prédiction
def normalize_and_show_pred(image_path):
    # Charger et prétraiter l'image
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Faire des prédictions
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label]

    # Afficher l'image et la prédiction
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f'Prédiction: {predicted_label} (Confiance: {confidence:.2%})')
    plt.axis('off')
    plt.show()

    # Afficher la distribution complète des probabilités
    print("Probabilités par classe :")
    for i, prob in enumerate(predictions[0]):
        print(f'Classe {i}: {prob:.2%}')

# Charger les données
image_paths = ["mydata/1.png", "mydata/2.png", "mydata/3.png", "mydata/inc.png"]

# Faire des prédictions
for data in image_paths:
    normalize_and_show_pred(data)
