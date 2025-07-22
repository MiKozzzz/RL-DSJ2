import os
import time
import cv2
import numpy as np

def Binaryzacja(image_path):
    # Wczytanie obrazu w skali szarości
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Sprawdzenie, czy obraz został poprawnie wczytany
    if img is None:
        print(f"Nie udało się wczytać obrazu: {image_path}")
        return False

    # Binaryzacja obrazu
    _, binary_image = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

    # Zapisanie binaryzowanego obrazu z powrotem
    cv2.imwrite(image_path, binary_image)
    return True


def Dzialanie_w_folderach(folder_path, funkcja):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # Możesz dostosować rozszerzenie plików do przetwarzania
            image_path = os.path.join(folder_path, filename)
            if funkcja(image_path):
                print(f"Przetworzono i zapisano: {filename}")
            else:
                print(f"Nie udało się przetworzyć: {filename}")


def okreslone_kolory_reszta_szara(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Sprawdzenie, czy obraz został poprawnie wczytany
    if image is None:
        print(f"Nie udało się wczytać obrazu: {image_path}")
        return False


    # Lista wartości pikseli do zachowania
    value_list = [76, 85, 73, 109, 41, 55, 65, 51, 107, 230, 91]

    # 91 Cien
    # 76 kask
    # 85 Gorna narta (Uwaga tlo)
    # 73 rekawiczka
    # 109 dolna narta
    # 41 nogawka tyl ciemny
    # 55 nogawka tyl jasniejszy
    # 65 nogawka przod jasniejszy
    # 51 tylna reka
    # 107 przod reka
    # 230 plastron klata

    # Wartość zastępcza
    replacement_value = 191

    # Konwersja listy wartości i wartości zastępczej na numpy array
    value_list_np = np.array(value_list)
    replacement_value_np = np.uint8(replacement_value)

    # Tworzenie maski dla pikseli, które mają pozostać niezmienione
    mask = np.isin(image, value_list_np)

    # Tworzenie nowego obrazu z wartością zastępczą
    filtered_image = np.full(image.shape, replacement_value_np, dtype=np.uint8)

    # Nakładanie oryginalnych pikseli na maskę
    filtered_image[mask] = image[mask]

    # Zapisanie przetworzonego obrazu
    cv2.imwrite(image_path, filtered_image)

    # print(filtered_image[0])
    # cv2.imshow("filtr", filtered_image)
    # # time.sleep(0.5)
    # cv2.imshow("zwykly", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return True


# Ścieżka do folderu z obrazami
folder_path = "C:/RL_DSJ2/obraz"

# Binaryzacja obrazów w folderze
# Dzialanie_w_folderach(folder_path, Binaryzacja)

# FILTROWANIE obrazów w folderze
Dzialanie_w_folderach(folder_path, okreslone_kolory_reszta_szara)


# okreslone_kolory_reszta_szara('kat_zawodnika.png')

# img = cv2.imread('Wszystkie_nauka/60_1.png', cv2.IMREAD_GRAYSCALE)

# Kask
# lower_threshold = 75
# upper_threshold = 77

# Narta dolna i rekaw
# lower_threshold = 100
# upper_threshold = 110

# lista_zaw = [[54, 56], [64, 66], [50, 52], [106, 108], [229, 231]]
# img_zaw = np.zeros((100, 100))
# for asd in lista_zaw:
#     lower_threshold = asd[0]
#     upper_threshold = asd[1]
#
#     # Binaryzacja obrazu przy użyciu dolnego progu
#     _, lower_binary = cv2.threshold(img, lower_threshold, 255, cv2.THRESH_BINARY)
#
#     # Binaryzacja obrazu przy użyciu górnego progu
#     _, upper_binary = cv2.threshold(img, upper_threshold, 255, cv2.THRESH_BINARY_INV)
#
#     # Połączenie obu progowych obrazów przy użyciu operacji AND
#     combined_binary = cv2.bitwise_and(lower_binary, upper_binary)
#
#     img_zaw = img_zaw + combined_binary
#
# print(img[0])
# cv2.imshow("img", img)
# # time.sleep(0.5)
# cv2.imshow("binary", img_zaw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


