import streamlit as st
import numpy as np
import tensorflow as tf
import os
from streamlit_drawable_canvas import st_canvas

st.title("Распознавание рукописных цифр")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="#ffffff",
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

model = None

if canvas_result is not None and canvas_result.image_data is not None:
    if not os.path.isfile("./mnist_model.h5"):
        raise FileNotFoundError("Не найден файл с обученной моделью.")
    else:
        model = tf.keras.models.load_model("./mnist_model.h5")
    
    image = canvas_result.image_data

    if image is not None:
        image = image.astype("float32")
        image = np.mean(image, axis=-1)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image = tf.image.resize(image, [28, 28])
        image = tf.expand_dims(image, axis=0)
        image = image / 255.0

        prediction = np.argmax(model.predict(image))

        st.write(f"Предсказанная цифра: {prediction}")