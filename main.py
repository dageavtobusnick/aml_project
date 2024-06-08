import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


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


def train_model():
    model = create_model()
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    model.save("mnist_model.h5")
    return model

np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
np.load('/tmp/123.npy')
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = create_model()
train_model(model, X_train, y_train, X_test, y_test)
model.save("mnist_model.h5")

