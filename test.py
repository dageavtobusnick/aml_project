import numpy as np
from tensorflow.keras import models

X_test = np.load('./data/test/X_processed.npy')
y_test = np.load('./data/test/Y.npy')

model = models.load_model("./production/mnist_model.h5")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(test_loss)
print(test_acc)