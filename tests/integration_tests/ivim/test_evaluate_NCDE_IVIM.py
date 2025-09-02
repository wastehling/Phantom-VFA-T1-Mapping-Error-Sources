import torch
import types
import os

from scripts.evaluate_qmri import evaluate


def get_args():
    args = types.SimpleNamespace()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.parameterfile = os.getcwd().split(sep='/NeuralCDE-QMRI')[0] + \
                         '/NeuralCDE-QMRI/tests/integration_tests/ivim/evaluate_parameterfile_NCDE_IVIM_invivo.yml'
    return args


class TestEvaluateNCDE:
    def test_evaluate(self):
        args = get_args()
        evaluate(args)
