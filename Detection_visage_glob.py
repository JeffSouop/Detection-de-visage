import os
import cv2
import time
import torch
import numpy as np
import face_recognition

# Appel des differents directory

input_dir = "path of data base imagee"
output_dir = "path of good image"
trash_dir = "path of trash image"


############################################################################# Algorithme utilisant la bibliotheque facial-recognition

def detect_faces(input_dir, output_dir, trash_dir):
    start_time = time.time()  # Mesurer le temps d'exécution
    num_saved = 0  # Compteur d'images sauvegardées
    num_trashed = 0  # Compteur d'images mises à la corbeille
    # Parcourir les images dans le dossier d'entrée
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(input_dir, filename))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)

            if len(face_locations) > 0:
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, img)
                num_saved += 1
            else:
                trash_path = os.path.join(trash_dir, filename)
                cv2.imwrite(trash_path, img)
                num_trashed += 1

    end_time = time.time()
    # print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")
    # print(f"Nombre d'images sauvegardées : {num_saved}")
    # print(f"Nombre d'images mises à la corbeille : {num_trashed}")


############################################################################# Algorithme utilisant Haarcascade

def detect_images_haarcascade(input_dir, output_dir, trash_dir):
    # Chargement du classificateur de cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Initialiser le compteur d'images
    detectable_count = 0
    non_detectable_count = 0

    # Mesurer le temps d'exécution
    start_time = time.time()

    # Parcourir les images dans le dossier d'entrée
    for filename in os.listdir(input_dir):
        # Vérifier que le fichier est une image
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Charger l'image
            image = cv2.imread(os.path.join(input_dir, filename))

            # Convertir l'image en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Détection de visages
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Vérifier si un visage a été détecté ou si l'image est toute noire
            if len(faces) > 0:
                cv2.imwrite(os.path.join(output_dir, filename), image)
                detectable_count += 1
            elif cv2.mean(gray)[0] < 10:
                cv2.imwrite(os.path.join(trash_dir, filename), image)
                non_detectable_count += 1
            else:
                cv2.imwrite(os.path.join(trash_dir, filename), image)

    # Mesurer le temps d'exécution
    end_time = time.time()
    execution_time = end_time - start_time

    # Afficher le nombre d'images détectées et le temps d'exécution
    # print(f"{detectable_count} images ont été détectées et sauvegardées dans {output_dir}")
    # print(f"{non_detectable_count} images n'ont pas été détectées et ont été déplacées dans {trash_dir}")
    # print(f"Le temps d'exécution est de {execution_time:.2f} secondes.")


#################################################################################### Algorithme utilisant YOLOv5

def detect_faces_in_images(input_dir, output_dir, trash_dir):
    # Charger le modèle YOLOv5 pré-entrainé
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

    # Initialiser les compteurs
    nb_saved_images = 0
    nb_trash_images = 0

    # Mesurer le temps d'exécution
    start_time = time.time()

    # Parcourir les images dans le dossier d'entrée
    for filename in os.listdir(input_dir):
        # Vérifier que le fichier est une image
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Charger l'image
            img = cv2.imread(os.path.join(input_dir, filename))

            # Convertir l'image de BGR en RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Détecter les visages dans l'image
            results = model(img)

            # Récupérer les coordonnées des boîtes englobantes des visages détectés
            boxes = results.xyxy[0][:, :4].tolist()

            # Dessiner les boîtes englobantes en niveaux de gris
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            for box in boxes:
                x1, y1, x2, y2 = [int(x) for x in box]
                cv2.rectangle(gray_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Convertir l'image de niveaux de gris en couleurs BGR
            color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            # Enregistrer l'image dans le dossier de sauvegarde si des visages ont été détectés
            if len(boxes) > 0:
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, color_img)
                nb_saved_images += 1
            # Enregistrer l'image dans le dossier de corbeille sinon
            else:
                trash_path = os.path.join(trash_dir, filename)
                cv2.imwrite(trash_path, color_img)
                nb_trash_images += 1

    # Afficher les résultats
    # print("Temps d'exécution : %.2f secondes" % (time.time() - start_time))
    # print("Nombre d'images sauvegardées : %d" % nb_saved_images)
    # print("Nombre d'images dans la corbeille : %d" % nb_trash_images)


#####################################################################################Algorithme utilisant Deep Neural Network

def detect_faces(input_dir, output_dir, trash_dir):
    start_time = time.time()

    # Charger le modèle de détection de visages de OpenCV
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # Vérifier si le dossier de sortie existe et le créer s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Vérifier si le dossier corbeille existe et le créer s'il n'existe pas
    if not os.path.exists(trash_dir):
        os.makedirs(trash_dir)

    # Parcourir toutes les images dans le dossier d'entrée
    save_count = 0
    trash_count = 0
    for filename in os.listdir(input_dir):
        # Vérifier que le fichier est une image
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Charger l'image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Récupérer les dimensions de l'image et créer un blob d'entrée pour le modèle
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # Passer le blob dans le réseau de neurones pour détecter les visages
            net.setInput(blob)
            detections = net.forward()

            # Vérifier s'il y a eu des détections de visages dans l'image
            has_face = False
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Si la confiance de la détection est suffisamment élevée
                if confidence > 0.5:
                    has_face = True

                    # Récupérer les coordonnées du rectangle entourant le visage
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Dessiner le rectangle autour du visage
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

            if has_face:
                output_path = os.path.join(output_dir, filename)
                save_count += 1
            else:
                output_path = os.path.join(trash_dir, filename)
                trash_count += 1
            cv2.imwrite(output_path, image)
    end_time = time.time()
    execution_time = end_time - start_time

    # print(f"{save_count} images ont été détectées et sauvegardées dans {output_dir}")
    # print(f"{trash_count} images n'ont pas été détectées et ont été déplacées dans {trash_dir}")
    # print(f"Le temps d'exécution est de {execution_time:.2f} secondes.")


print("1- Facial Recognition\n 2- Haarcascade\n 3- YOLOv5\n 4- Deep Neural Network")
choix = int(input("Veuillez faire un choix d'algorithme de detection:"))

for i in range(1, 4):
    if choix == 1:
        detect_faces(input_dir, output_dir, trash_dir)
    else:
        if choix == 2:
            detect_images_haarcascade(input_dir, output_dir, trash_dir)
        else:
            if choix == 3:
                detect_faces_in_images(input_dir, output_dir, trash_dir)
            else:
                if choix == 4:
                    detect_faces(input_dir, output_dir, trash_dir)
