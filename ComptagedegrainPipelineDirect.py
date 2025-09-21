import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Chemin du dossier contenant les images
folder_path = r"D:\cours IFI\Traitement D'image\Images\Proj1.2"

# Lister toutes les images du dossier
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        img_path = os.path.join(folder_path, filename)

        # Charger l'image en niveaux de gris
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Impossible de lire {filename}")
            continue

        # --- Seuillage Otsu ---
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Inversion si nécessaire
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # --- Morphologie (ouverture) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # --- Composantes connexes ---
        num_labels, labels = cv2.connectedComponents(opened)

        # --- Résultat final ---
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for label in range(1, num_labels):  # 0 = fond
            mask = np.uint8(labels == label) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, str(label), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 255), 1)

        # --- Résultats ---
        print(f"✅ {filename} : {num_labels - 1} grains de riz détectés")

        # --- Affichage avec Matplotlib ---
        titles = ["Image originale", "Binaire (Otsu)", "Nettoyée (Ouverture)", "Résultat final"]
        images = [img, binary, opened, output]

        plt.figure(figsize=(14, 6))
        for i, (im, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 4, i + 1)
            if len(im.shape) == 2:  # image en niveaux de gris
                plt.imshow(im, cmap="gray")
            else:  # image couleur BGR → RGB
                plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis("off")
        plt.suptitle(f"{filename} → {num_labels - 1} grains détectés", fontsize=14)
        plt.tight_layout()
        plt.show()
