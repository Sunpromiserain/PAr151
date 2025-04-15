from PIL import Image
from distortions import apply_single_distortion

# Charger l'image
image_path = "C:/Users/zyr/Desktop/PAr/output/watermarked_image.png"
image = Image.open(image_path)

# Appliquer la distorsion de rotation
distorted_image = apply_single_distortion(
    image=image,
    distortion_type="rotation",  # Choisir la distorsion de rotation
    strength=10,                  # Angle de rotation
    distortion_seed=42            # Graine aléatoire
)

# Afficher l'image distordue
# distorted_image.show()

# Sauvegarder l'image distordue
distorted_image.save("C:/Users/zyr/Desktop/PAr/output/watermarked_image_rotated.png")

# Appliquer la distorsion de luminosité
distorted_image = apply_single_distortion(
    image=image,
    distortion_type="brightness",  # Choisir l'ajustement de luminosité
    strength=1.8,                  # Coefficient d'augmentation de luminosité
    distortion_seed=42             # Graine aléatoire
)

# Afficher l'image distordue
# distorted_image.show()

# Sauvegarder l'image distordue
distorted_image.save("C:/Users/zyr/Desktop/PAr/output/watermarked_image_brightness.png")

# Appliquer la distorsion de flou
distorted_image = apply_single_distortion(
    image=image,
    distortion_type="blurring",  # Choisir la distorsion de flou
    strength=1.5,                  # Intensité du flou (minimum 1)
    distortion_seed=42           # Graine aléatoire
)

# Afficher l'image distordue
# distorted_image.show()

# Sauvegarder l'image distordue
distorted_image.save("C:/Users/zyr/Desktop/PAr/output/watermarked_image_blurred.png")

# Appliquer la distorsion de bruit
distorted_image = apply_single_distortion(
    image=image,
    distortion_type="noise",  # Choisir la distorsion de bruit
    strength=0.03,            # Intensité du bruit
    distortion_seed=42        # Graine aléatoire
)

# Afficher l'image distordue
# distorted_image.show()

# Sauvegarder l'image distordue
distorted_image.save("C:/Users/zyr/Desktop/PAr/output/watermarked_image_with_noise.png")

# Appliquer la distorsion de compression
distorted_image = apply_single_distortion(
    image=image,
    distortion_type="compression",  # Choisir la distorsion de compression
    strength=85,                    # Qualité de compression (0-100)
    distortion_seed=42              # Graine aléatoire
)

# Afficher l'image distordue
# distorted_image.show()

# Sauvegarder l'image distordue
distorted_image.save("C:/Users/zyr/Desktop/PAr/output/watermarked_image_compressed.png")
