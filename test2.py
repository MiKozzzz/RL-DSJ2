import cv2
import os


def convert_jpg_to_png(input_folder, output_folder):
    # Tworzy folder wyjściowy jeśli nie istnieje
    os.makedirs(output_folder, exist_ok=True)

    # Iteracja po wszystkich plikach w folderze
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            jpg_path = os.path.join(input_folder, filename)
            png_name = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_folder, png_name)

            # Wczytanie JPG
            img = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)

            # Zapis PNG
            cv2.imwrite(png_path, img)

            print(f"✔ Przekonwertowano: {filename} → {png_name}")

    print("\nKonwersja zakończona!")


# ---- UŻYCIE ----
convert_jpg_to_png(
    input_folder="cyfry",
    output_folder="cyfry"
)
