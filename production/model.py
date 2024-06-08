import tensorflow as tf 
import os

model= None
if not os.path.isfile("mnist_model.h5"):
    raise FileNotFoundError("Не найден файл с обученной моделью.")
else:
     model = tf.keras.models.load_model("mnist_model.h5")

def predict(image):
    if image is None:
        raise ValueError
    model.predict(image)