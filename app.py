import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import tempfile
from ultralytics import YOLO

model_path = r'C:\Users\ruzks\VK_intership\ML Developer_02.2025\logo-detection\best.pt'
model = YOLO(model_path)

def detect_image(image):
    results = model(image)
    return results

def detect_video(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    cap = cv2.VideoCapture(temp_video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_image(frame)
        if results:
            frame = results[0].plot()
            frames.append(frame)

    cap.release()
    return frames

st.title('Детекция логотипов на изображениях и видео')

uploaded_file = st.file_uploader("Загрузите изображение или видео", type=['jpg', 'png', 'mp4'])

if uploaded_file is not None:
    if uploaded_file.type in ['image/jpeg', 'image/png']:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_container_width=True)

        results = detect_image(np.array(image))

        if results:
            result = results[0]
            st.image(result.plot(), caption='Результаты детекции', use_container_width=True)
        else:
            st.write("Объекты не найдены.")

    elif uploaded_file.type == 'video/mp4':
        st.write("Обработка видео, пожалуйста, подождите...")
        frames = detect_video(uploaded_file)

        if frames:
            st.image(frames[0], caption='Результат на первом кадре видео', use_container_width=True)
            st.video(uploaded_file)
        else:
            st.write("Не удалось обработать видео. Возможно, оно повреждено.")
