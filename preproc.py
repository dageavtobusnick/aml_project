import numpy as np

X_train = np.load('./data/train/X.npy')
X_test = np.load('./data/test/X.npy')

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

np.save('./data/train/X_processed', X_train)
np.save('./data/test/X_processed', X_test)