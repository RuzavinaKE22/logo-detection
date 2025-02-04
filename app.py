import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO

model_path = r'C:\Users\ruzks\VK_intership\ML Developer_02.2025\logo-detection\yolo11n.pt'

model = YOLO(model_path)

# Функция для обработки изображений
def detect_image(image):
    results = model(image)  # Выполнение детекции
    return results

# Функция для обработки видео
def detect_video(video_file):
    # Чтение видео с помощью OpenCV
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Выполнение детекции на каждом кадре
        results = detect_image(frame)
        # Сохраняем обработанный кадр с результатами
        img = results.render()[0]
        frames.append(img)
    cap.release()
    return frames

# Заголовок веб-приложения
st.title('Логотипы на изображениях и видео')

# Загрузка изображения или видео
uploaded_file = st.file_uploader("Загрузите изображение или видео", type=['jpg', 'png', 'mp4'])

if uploaded_file is not None:
    # Если это изображение
    if uploaded_file.type in ['image/jpeg', 'image/png']:
        # Преобразование файла в изображение
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_container_width=True)

        # Детекция
        results = detect_image(np.array(image))
        # Получаем первый результат из списка (YOLOv8 возвращает список результатов)
        result = results[0]
        # Отображаем обработанное изображение
        st.image(result.plot(), caption='Результаты детекции', use_container_width=True)


    # Если это видео
    elif uploaded_file.type == 'video/mp4':
        # Детекция на видео
        frames = detect_video(uploaded_file)
        # Выводим первое изображение с результатами
        st.image(frames[0], caption='Результат на первом кадре видео', use_container_width=True)

        # Для вывода видео (если хотите показать видео в реальном времени)
        st.video(uploaded_file)
