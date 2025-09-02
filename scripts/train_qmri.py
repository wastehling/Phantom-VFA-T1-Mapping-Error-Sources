import argparse

import torch
import yaml

from utils.data_loading.load_data import load_train_data
from utils.data_loading.dataloaders import create_train_dataloaders, create_val_dataloaders
from utils.save_data import save_loss_history
from utils.setup import (setup_logger, setup_target_folder, copy_parameterfile, log_git_hash, log_target_folder,
                         log_device, log_weighted_training)
from models.create_model import create_model
from models.FunctionCreator import FunctionCreator
from utils.data_processing.preprocess_data import preprocess_data, get_weights
from utils.data_modification.modify_data import flatten_data_structure
from utils.metrics.train_metrics import create_metric_list


def train(args):
    """
    Main function for training of model
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
    target_folder = setup_target_folder(parameters, 'train')
    copy_parameterfile(path_to_parameterfile, target_folder)

    # setup logger
    logger = setup_logger(target_folder)
    log_device(device)
    log_git_hash()
    log_target_folder(target_folder)

    # load data
    data = {}
    data = load_train_data(data, parameters)

    # preprocess data
    data = preprocess_data(data, parameters['preprocessing'])
    train_set, validation_set, test_set = flatten_data_structure(data)

    train_data_parameters = parameters['data']['in_vivo'].get('train', [])
    weighted = any(bool(dataset.get('weight')) for dataset in train_data_parameters)
    log_weighted_training(weighted)

    if weighted:
        weights = get_weights(parameters['data']['in_vivo']['train'])
    else:
        weights = []

    # create dataloaders
    train_loader = create_train_dataloaders(train_set, parameters['training']['batch_size'],
                                            parameters['training']['steps_per_epoch'],
                                            parameters['qmri_fit']['model'], 
                                            weights, 
                                            weighted=weighted)
    
    val_loader = create_val_dataloaders(validation_set, parameters['validation']['batch_size'],
                                        parameters['qmri_fit']['model'])

    # initialize model
    model = create_model(parameters, device)

    # initialize metrics
    metric_list = create_metric_list(parameters['training']['metrics'])

    # train model
    train_function = FunctionCreator.create(parameters['model']['train_function'])
    model, metric_dict = train_function(model, parameters, train_loader, val_loader, metric_list, target_folder, device)

    # save model
    if isinstance(model, list):
        for idx, model_el in enumerate(model):
            torch.save(model_el, target_folder + f'/model_{idx}.pt')
    else:
        torch.save(model, target_folder + '/model.pt')

    # save loss history
    save_loss_history(target_folder, metric_dict)

    # log target folder
    log_target_folder(target_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration
    parser.add_argument('--parameterfile', type=str, default='/home/rnga/dkuppens/NeuralCDE-QMRI/parameters/train_parameterfile.yml')
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args)
