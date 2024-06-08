import numpy as np
import sys
import os
from tensorflow.keras.datasets import mnist

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

os.makedirs('./data', exist_ok=True)
os.makedirs('./data/train', exist_ok=True)
os.makedirs('./data/test', exist_ok=True)
np.save('./data/train/X', X_train)
np.save('./data/train/Y', y_train)
np.save('./data/test/X', X_test)
np.save('./data/test/Y', y_test)