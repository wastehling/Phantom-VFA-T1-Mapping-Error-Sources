import argparse
import logging

import torch
import yaml

from utils.data_loading.load_data import load_test_data
from utils.setup import setup_logger, setup_target_folder, copy_parameterfile, log_git_hash, log_target_folder
from models.create_model import load_model
from utils.evaluation import evaluation
from utils.data_processing.preprocess_data import preprocess_data
from utils.save_data import save_metric_dict


def evaluate(args):
    """
    Main function for evaluation of model
    Args:
        args: object containing arguments, used for command line execution of script

    """

    # unpack arguments
    device = args.device
    path_to_parameterfile = args.parameterfile

    # load experiment parameters
    with open(path_to_parameterfile) as file:
        parameters = yaml.load(file, Loader=yaml.Loader)

    # setup target folder
    target_folder = setup_target_folder(parameters, 'evaluate')
    copy_parameterfile(path_to_parameterfile, target_folder)

    # setup logger
    logger = setup_logger(target_folder)
    log_git_hash()
    log_target_folder(target_folder)

    # load data
    data = {}
    data = load_test_data(data, parameters)

    # preprocess data
    data = preprocess_data(data, parameters['preprocessing'])

    # initialize model
    if 'model_path' in parameters['test'].keys():
        model, device = load_model(parameters['test']['model_path'], device=device)
    else:
        model = None

    # evaluation
    data, metric_dict = evaluation(model, parameters, data, device, target_folder)

    # log metric dict
    save_metric_dict(metric_dict, target_folder)

    # log target folder
    log_target_folder(target_folder)
    logging.info('finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration
    parser.add_argument('--parameterfile', type=str, default='../parameters/evaluate_parameterfile.yml')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args)
