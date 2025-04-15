from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from test import extract_watermark


image_path = "C:/Users/zyr/Desktop/PAr/output/watermarked_image.png"  # Chemin de l'image après l'attaque
# 1. Chemin de l'image après l'attaque (rotation)
image_path_1 = "C:/Users/zyr/Desktop/PAr/output/watermarked_image_rotated.png"  # Image après rotation

# 2. Chemin de l'image après l'attaque (ajustement de la luminosité)
image_path_2 = "C:/Users/zyr/Desktop/PAr/output/watermarked_image_brightness.png"  # Image après ajustement de la luminosité

# 3. Chemin de l'image après l'attaque (flou)
image_path_3 = "C:/Users/zyr/Desktop/PAr/output/watermarked_image_blurred.png"  # Image après flou

# 4. Chemin de l'image après l'attaque (bruit)
image_path_4 = "C:/Users/zyr/Desktop/PAr/output/watermarked_image_with_noise.png"  # Image après ajout de bruit

# 5. Chemin de l'image après l'attaque (compression)
image_path_5 = "C:/Users/zyr/Desktop/PAr/output/watermarked_image_compressed.png"  # Image après compression


# Charger l'image
watermarked_image = Image.open(image_path_3)     #### A modifier ici ! remplacez par le chemin souhaite ####

# Convertir l'image PIL en tableau NumPy
watermarked_image_array = np.array(watermarked_image)

# Extraire le filigrane (supposons que la fonction extract_watermark soit définie)
extracted_watermark = extract_watermark(watermarked_image_array, key=None)

# Créer une nouvelle figure
plt.figure(figsize=(8, 8))

# Afficher l'image du filigrane extrait
plt.imshow(extracted_watermark, cmap='gray', vmin=0, vmax=1)  # Afficher en niveau de gris, s'assurer que l'échelle est entre [0,1]
plt.title("Extracted Watermark")  # Ajouter un titre
plt.axis('off')  # Désactiver les axes

# Afficher l'image
plt.show()

# Sauvegarder l'image du filigrane extrait
# cv2.imwrite("C:/Users/zyr/Desktop/PAr/output/extracted_watermark.png", extracted_watermark)

cv2.imwrite("output/extracted_watermark.png", extracted_watermark)



import matplotlib.pyplot as plt
import cv2
import os

# Liste des chemins d'image, les images sont situées dans le répertoire 'output'
image_paths = [
    "C:/Users/zyr/Desktop/PAr/output/blurred-1.png",
    "C:/Users/zyr/Desktop/PAr/output/blurred-1.2.png",
    "C:/Users/zyr/Desktop/PAr/output/blurred-2.png",    
    "C:/Users/zyr/Desktop/PAr/output/blurred-5.png",
    "C:/Users/zyr/Desktop/PAr/output/brightness-1.2.png",
    "C:/Users/zyr/Desktop/PAr/output/brightness-1.5.png",
    "C:/Users/zyr/Desktop/PAr/output/brightness-1.79.png",
    "C:/Users/zyr/Desktop/PAr/output/brightness-2.png",
    "C:/Users/zyr/Desktop/PAr/output/compressed-50.png",
    "C:/Users/zyr/Desktop/PAr/output/compressed-80.png",
    "C:/Users/zyr/Desktop/PAr/output/compressed-85.png",
    "C:/Users/zyr/Desktop/PAr/output/compressed-90.png",
    "C:/Users/zyr/Desktop/PAr/output/noise-0.01.png",
    "C:/Users/zyr/Desktop/PAr/output/noise-0.02.png",
    "C:/Users/zyr/Desktop/PAr/output/noise-0.03.png",
    "C:/Users/zyr/Desktop/PAr/output/noise-0.05.png",
    "C:/Users/zyr/Desktop/PAr/output/rotation-1.png",
    "C:/Users/zyr/Desktop/PAr/output/rotation-10.png",
    "C:/Users/zyr/Desktop/PAr/output/rotation-20.png",
    "C:/Users/zyr/Desktop/PAr/output/rotation-45.png"

]



# Créer une grille 2x4 pour afficher ces images
fig, axes = plt.subplots(5, 4, figsize=(15, 12))

# Charger et afficher les images
for i, ax in enumerate(axes.flat):
    if i < len(image_paths):  # Vérifier que nous ne dépassons pas la liste des images
        # Lire l'image
        img = cv2.imread(image_paths[i])

        # OpenCV lit l'image en format BGR, la convertir en format RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Afficher l'image sur le graphique courant
        ax.imshow(img_rgb)
        ax.axis('off')  # Désactiver les axes

        # Définir le titre pour chaque sous-graphique
        title = os.path.basename(image_paths[i]).split(".")[0]  # Obtenir le nom du fichier comme titre
        title = title.replace(".", " ")  # Remplacer les points dans le nom par des espaces
        ax.set_title(title)
    else:
        ax.axis('off')  # Si pas d'image, désactiver le sous-graphique

# Ajuster automatiquement la mise en page pour éviter le chevauchement des titres et des images
plt.tight_layout()
# plt.axis('off') 

# Afficher l'image finale
plt.show()

# Sauvegarder l'image composite
fig.savefig("C:/Users/zyr/Desktop/PAr/output/attacked_images_grid_2.png")

import matplotlib.pyplot as plt
import cv2
import os

# Spécifier le répertoire contenant les images
image_dir = "C:/Users/zyr/Desktop/PAr/output/attacked_images/"

# Obtenir tous les fichiers image dans ce répertoire (supposé être au format PNG)
image_paths = [
    os.path.join(image_dir, "watermarked_image_blurred.png"),
    os.path.join(image_dir, "watermarked_image_brightness.png"),
    os.path.join(image_dir, "watermarked_image_compressed.png"),
    os.path.join(image_dir, "watermarked_image_rotated.png"),
    os.path.join(image_dir, "watermarked_image_with_noise.png")
]

# Créer une grille 2x3 pour afficher ces images
fig, axes = plt.subplots(2, 3, figsize=(15, 12))

# Charger et afficher les images
for i, ax in enumerate(axes.flat):
    if i < len(image_paths):  # Vérifier que nous ne dépassons pas la liste des images
        # Lire l'image
        img = cv2.imread(image_paths[i])

        # OpenCV lit l'image en format BGR, la convertir en format RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Afficher l'image sur le graphique courant
        ax.imshow(img_rgb)
        ax.axis('off')  # Désactiver les axes

        # Définir le titre pour chaque sous-graphique, extraire le type d'image (par exemple, flou, rotation)
        title = os.path.basename(image_paths[i]).split(".")[0]  # Obtenir le nom du fichier
        title = title.split("_")[-1]  # Prendre la dernière partie du nom (par exemple "blurred", "rotated")
        ax.set_title(title)

    else:
        ax.axis('off')  # Si pas d'image, désactiver le sous-graphique

# Ajuster automatiquement la mise en page pour éviter le chevauchement des titres et des images
plt.tight_layout()

# Afficher l'image finale
plt.show()

# Sauvegarder l'image composite
fig.savefig("C:/Users/zyr/Desktop/PAr/output/attacked_images_grid_4.png")
