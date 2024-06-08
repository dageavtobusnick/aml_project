import numpy as np
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

np.save('./data/train/X', X_train)
np.save('./data/train/Y', y_train)
np.save('./data/test/X', X_test)
np.save('./data/test/Y', y_test)