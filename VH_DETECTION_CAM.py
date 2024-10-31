"""
import numpy as np
import cv2
import torch
import supervision as sv

def empty(a):
    pass

def resize_final_img(x, y, *argv):
    images = cv2.resize(argv[0], (x, y))
    for i in argv[1:]:
        resize = cv2.resize(i, (x, y))
        images = np.concatenate((images, resize), axis=1)
    return images

# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

# Intentar abrir la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")
    exit()

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 300, 300)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

cv2.namedWindow('F')
cv2.resizeWindow('F', 700, 600)

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: No se puede leer el frame de la cámara")
        break

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_img, lower, upper)
    kernel = np.ones((3, 3), 'uint8')

    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    final_img = resize_final_img(300, 300, mask, d_img)

    # Detección de personas usando YOLOv5
    results = model(img)
    detections = sv.Detections.from_yolov5(results)
    person_detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
    person_count = len(person_detections)

    # Mostrar cantidad de personas detectadas
    cv2.putText(final_img, f'Personas: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('F', final_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

#mejoras detectadas, capturar imagen con PIL o crear una alerta con Tkinter


import numpy as np
import cv2
import torch
import supervision as sv
from datetime import datetime

def empty(a):
    pass

# Cargar modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

# Permite abrir camara donde 0 es la camara 1 , se puede ingresar otros valors como 1, para una segunda camara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se puede acceder a la cámara")
    exit()

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 300, 300)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)

cv2.namedWindow('F')
cv2.resizeWindow('F', 700, 600)

last_person_count = -1

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: No se puede leer el frame de la cámara")
        break

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_img, lower, upper)
    kernel = np.ones((3, 3), 'uint8')

    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Detección de personas usando YOLOv5
    results = model(img)
    detections = sv.Detections.from_yolov5(results)
    person_detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
    person_count = len(person_detections)

    # Añadir un rectángulo negro y el conteo de personas en texto amarillo
    img_height, img_width, _ = img.shape
    cv2.rectangle(img, (0, 0), (300, 50), (0, 0, 0), -1)
    cv2.putText(img, f'Personas: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    #imprime la cantidad de personas en consola
    if person_count != last_person_count:
        date_actual = datetime.now()
        print(f"{date_actual} | Personas : {person_count}")
        last_person_count = person_count

    cv2.imshow('F', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
