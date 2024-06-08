import numpy as np
from tensorflow.keras import layers, models

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

def train_model(X_train, y_train):
    model = create_model()
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    model.save("./production/mnist_model.h5")
    return model

X_train = np.load('./data/train/X_processed.npy')
y_train = np.load('./data/train/Y.npy')

train_model(X_train, y_train)