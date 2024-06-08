import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from streamlit_drawable_canvas import st_canvas

# Флаг для отслеживания того, было ли обучение модели
model_trained = False

# Функция для создания модели CNN
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Функция для обучения модели
def train_model():
    global model_trained
    model = create_model()
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    st.write(f"Точность модели на тестовых данных: {test_acc * 100:.2f}%")
    model.save("mnist_model.h5")
    model_trained = True
    return model

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование данных и нормализация
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Streamlit приложение
st.title("Распознавание рукописных цифр")

# Создание холста для рисования
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Цвет кисти
    stroke_width=20,  # Толщина кисти
    stroke_color="#ffffff",  # Цвет обводки
    background_color="#000000",  # Цвет фона
    height=150,  # Высота холста
    width=150,  # Ширина холста,
    drawing_mode="freedraw",  # Режим рисования
    key="canvas",
)

# Обработка нарисованного изображения
if canvas_result is not None and canvas_result.image_data is not None:
    # Проверяем, была ли обучена модель
    if not os.path.isfile("mnist_model.h5"):
        model = create_model()
        train_model(model, X_train, y_train, X_test, y_test)
        model.save("mnist_model.h5")
    else:
        model = tf.keras.models.load_model("mnist_model.h5")
    
    # Преобразование изображения в формат, подходящий для модели
    image = canvas_result.image_data.astype("float32")

    # Конвертация изображения в градации серого
    image = np.mean(image, axis=-1)

    # Убедимся, что изображение имеет три измерения (высота, ширина, каналы)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    # Изменение размера изображения до 28x28
    image = tf.image.resize(image, [28, 28])

    # Добавление батч-измерения для совместимости с моделью
    image = tf.expand_dims(image, axis=0)

    # Нормализация изображения
    image = image / 255.0

    # Предсказание
    prediction = np.argmax(model.predict(image))

    # Вывод предсказанной цифры
    st.write(f"Предсказанная цифра: {prediction}")
