import unittest
from production.model import predict
import numpy as np
import sys
from tensorflow.keras import models

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
class TestPredict(unittest.TestCase):

    def test_predict_None(self):
        image = None
        with self.assertRaises(ValueError):
            predict(image)

    def test_predict_test_data(self):
        X_test = np.load('../data/test/X_processed.npy')
        y_test = np.load('../data/test/Y.npy')

        model = models.load_model("./mnist_model.h5")
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"loss:{test_loss}")
        print(f"accuracy:{test_acc}")
        self.assertLess(test_loss, 0.04)
        self.assertGreater(test_acc, 0.96)


if __name__ == '__main__':
    unittest.main()