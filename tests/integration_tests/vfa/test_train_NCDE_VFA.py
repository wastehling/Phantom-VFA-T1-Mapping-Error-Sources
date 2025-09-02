import torch
import types
import os

from scripts.train_qmri import train


def get_args():
    args = types.SimpleNamespace()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.parameterfile = os.getcwd().split(sep='/NeuralCDE-QMRI')[0] + \
                         '/NeuralCDE-QMRI/tests/integration_tests/vfa/train_parameterfile_VFA.yml'
    return args


class TestTrainNCDE:
    def test_train(self):
        args = get_args()
        train(args)