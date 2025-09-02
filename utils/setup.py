import os
import sys
import logging
import datetime
import shutil
import subprocess


def setup_logger(target_folder):
    """
    set up logger
    Args:
        target_folder: where to set up logfile

    Returns: logger
    """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler(target_folder+'/logfile.txt'),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger('logger')
    logger.addHandler(logging.StreamHandler)
    return logger


def setup_target_folder(parameters, type):
    """
    setup target folder
    Args:
        parameters: parameters dict
        type: type of script calling this function, used for sorting of folders

    Returns: str of target folder

    """
    supplied_target_folder = parameters['experiment']['target_folder']

    # change path
    if '/home/rnga/' not in supplied_target_folder:
        user = os.getcwd().split('/')[3]
        supplied_target_folder = f'/home/rnga/{user}/my-scratch/{supplied_target_folder}'

    current_datetime_str = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    target_folder = supplied_target_folder + type + '/' + current_datetime_str + '_' + \
                    parameters['experiment']['name']
    os.makedirs(target_folder, exist_ok=True)
    return target_folder


def copy_parameterfile(src, dst):
    """
    copy paramfile from src to dst
    Args:
        src: str of source path
        dst: str of dst path

    Returns:
    """
    shutil.copyfile(src, dst + '/parameterfile.yml')


def log_git_hash():
    """
    log git hash
    Returns: None

    """
    logging.info('current git hash: ' + subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())


def log_target_folder(target_folder):
    """
    log target folder
    Args:
        target_folder: str of target folder

    Returns:

    """
    logging.info(f'target directory is: {target_folder}')

def log_device(device):
    """
    log device
    Returns: None

    """
    logging.info(f'using device: {device}')

def log_weighted_training(weight):
    """
    log weighted training
    Returns: None

    """
    logging.info(f'using weighted training: {weight}')