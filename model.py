from datetime import datetime
import os
import shutil
import unittest

import numpy as np
import torch
from learning_model import FLModel
from preprocess import get_test_loader
from preprocess import UserRoundData

from sklearn import tree


class ParameterServer(object):
    def __init__(self, init_model_path, testworkdir):
        self.rounds_model_path = {}
        self.current_models = []
        self.init_model_path = init_model_path

        self.testworkdir = testworkdir
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

    def receive_models_info(self, model):
        self.current_models.append(model)

    def ensemble(self, data):
        pred = np.empty((len(data[:, 1]), len(self.current_models)))
        for i, model in enumerate(self.current_models):
            pred[:, i] = model.predict(data)
        return np.array([max(pred[i, :].tolist(), key=pred[i, :].tolist().count) for i in range(len(pred[:, 0]))],
                        dtype='int64')


class RFTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    TEST_BASE_DIR = '/tmp/'

    def setUp(self):

        self.testbase = self.TEST_BASE_DIR
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        # torch.manual_seed(self.seed)

        if not os.path.exists(self.init_model_path):
            torch.save(FLModel().state_dict(), self.init_model_path)

        self.ps = ParameterServer(init_model_path=self.init_model_path,
                                  testworkdir=self.testworkdir)

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.urd = UserRoundData()
        self.n_users = self.urd.n_users

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    def test_random_forest(self):
        model = None
        for u in range(0, self.n_users):
            # decision tree
            model = tree.DecisionTreeClassifier()
            x, y = self.urd.round_data(user_idx=u)
            model.fit(x, y)
            self.ps.receive_models_info(model=model)

        if model is not None:
            self.save_testdata_prediction(model=model)

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray,)):
            predition = predition.reshape(-1).tolist()

        with open(os.path.join(self.RESULT_DIR, 'result.txt'), 'w') as fout:
            fout.writelines(os.linesep.join([str(n) for n in predition]))

    def save_testdata_prediction(self, model):
        loader = get_test_loader(batch_size=1000)
        prediction = []
        for data in loader:
            # prediction
            pred = self.ps.ensemble(data)
            prediction.extend(pred.reshape(-1).tolist())
        self.save_prediction(prediction)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(RFTestSuit('test_random_forest'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    main()