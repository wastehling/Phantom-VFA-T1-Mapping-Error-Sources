import logging
import os

import numpy as np
import torchio as tio
import torch

from utils.data_modification.morphological_operation import morphological_closing, largest_connected_component_3d
from qmri.data_loading.dce.load_dce_data import load_t1_map, load_aif_image
from qmri.data_loading.DataLoaderCreator import DataLoaderCreator
from qmri.simulation.create_simulated_data import create_structured_simulation_data, create_unstructured_simulation_data
from utils.data_modification.modify_data import slice_and_multiply


def load_train_data(data, parameters):
    """
    loads training data according to parameters
    Args:
        data: dict containing data
        parameters: parameters dict

    Returns:
        data: dict containing loaded train data

    """
    data['in_vivo'] = {}
    data['simulation'] = {}

    if ('in_vivo' in parameters['data'].keys() and 'train' in parameters['data']['in_vivo'].keys() and
            parameters['data']['in_vivo']['train']):
        data['in_vivo']['train'] = load_invivo_data(parameters['data']['in_vivo']['train'])
        logging.info(f"loaded {len(data['in_vivo']['train'])} studies of in vivo train data")

    if ('in_vivo' in parameters['data'].keys() and 'validation' in parameters['data']['in_vivo'].keys() and
            parameters['data']['in_vivo']['validation']):
        data['in_vivo']['validation'] = load_invivo_data(parameters['data']['in_vivo']['validation'])
        logging.info(f"loaded {len(data['in_vivo']['validation'])} studies of in vivo validation data")

    if ('simulation' in parameters['data'].keys() and 'train' in parameters['data']['simulation'].keys() and
            parameters['data']['simulation']['train']):
        data['simulation']['train'] = load_simulation_data(parameters['data']['simulation']['train'])
        logging.info(f"loaded {len(data['simulation']['train'])} studies of simulation train data")

    if ('simulation' in parameters['data'].keys() and 'validation' in parameters['data']['simulation'].keys() and
            parameters['data']['simulation']['validation']):
        data['simulation']['validation'] = load_simulation_data(parameters['data']['simulation']['validation'])
        logging.info(f"loaded {len(data['simulation']['validation'])} studies of in simulation val data")

    if data['in_vivo'] == {} and data['simulation'] == {}:
        raise ValueError("No data loaded, please check your parameters")

    return data


def load_test_data(data, parameters):
    """
    loads test data according to parameters
    Args:
        data: dict containing data
        parameters: parameters dict

    Returns:
        data: dict containing loaded test data

    """
    data['in_vivo'] = {}
    data['simulation'] = {}

    if ('in_vivo' in parameters['data'].keys() and 'test' in parameters['data']['in_vivo'].keys() and
            parameters['data']['in_vivo']['test']):
        data['in_vivo']['test'] = load_invivo_data(parameters['data']['in_vivo']['test'])
        logging.info(f"loaded {len(data['in_vivo']['test'])} studies of in vivo test data")

    if ('simulation' in parameters['data'].keys() and 'test' in parameters['data']['simulation'].keys() and
            parameters['data']['simulation']['test']):
        data['simulation']['test'] = load_simulation_data(parameters['data']['simulation']['test'])
        logging.info(f"loaded {len(data['simulation']['test'])} studies of simulation test data")

    if data['in_vivo'] == {} and data['simulation'] == {}:
        raise ValueError("No data loaded, please check your parameters")
    return data


def load_simulation_data(simulation_parameters):
    """
    loads simulation data according to simulation_parameters
    Args:
        simulation_parameters: simulation parameters

    Returns:
        data_per_study: dict with data organized per study

    """
    # determine amount of simulated samples required
    data_per_study = {}
    for study_simulation_parameters in simulation_parameters:

        samples = study_simulation_parameters['samples']

        if 'seed' in study_simulation_parameters.keys():
            seed = study_simulation_parameters['seed']
        elif 'seed_nr' in study_simulation_parameters.keys():
            seed = study_simulation_parameters['seed_nr']
        else:
            seed = None

        if 'structured' in study_simulation_parameters.keys() and study_simulation_parameters['structured'] is not None:
            simulation_set = create_structured_simulation_data(study_simulation_parameters, samples, seed)
        else:
            simulation_set = create_unstructured_simulation_data(study_simulation_parameters, samples, seed)

        data_per_study[study_simulation_parameters['name']] = simulation_set
    return data_per_study


def load_invivo_data(data_parameters):
    """
    loads dataset according to data_parameters
    Args:
        data_parameters: list of studies to load, each study is a dict containing necessary info to load images

    Returns:
        data_per_study: dict with data organized per study

    """
    data_per_study = {}
    for study_parameters in data_parameters:
        qmri_method = study_parameters['qmri']
        study_name = study_parameters['name']
        logging.info(f'started loading data for study {study_name}')

        # load data on a per study basis
        data_per_study[study_name] = load_study(study_parameters['path'], study_parameters['patient_ids'], qmri_method,
                                                study_parameters)
        if 'slice_and_multiply' in study_parameters.keys():
            new_data_per_study = {}
            dict_of_subjects = data_per_study[study_name]
            logging.info(f"started slicing and multiplying data for study {study_name}")
            if study_name not in new_data_per_study.keys():
                new_data_per_study = {}
            for subject_id in dict_of_subjects.keys():
                logging.info(f"started slicing and multiplying data for subject {subject_id} of study {study_name}")
                new_data_per_study.update(slice_and_multiply(dict_of_subjects[subject_id], subject_id,
                                                             study_parameters['slice_and_multiply']['multiplications'],
                                                             study_parameters['slice_and_multiply']['slice_factor']))
                logging.info(f"subject {subject_id} from study {study_name} was sliced and multiplied")
            data_per_study[study_name] = new_data_per_study
        logging.info(f'finished loading data for study {study_name}')

    return data_per_study


def load_study(study_path, subject_ids, qmri_method, parameters):
    """
    loads subjects for a study
    Args:
        study_path: path where to look for subject files
        subject_ids: ids of subjects to load
        qmri_method: qmri method, determines what data to load
        parameters: parameters dict specifying what data to load

    Returns:
        list_of_subjects: list containing all subjects for a specified study who meet the subject_ids criteria

    """

    # change path
    if '/home/rnga/' not in study_path:
        user = os.getcwd().split('/')[3]
        study_path = f'/home/rnga/{user}/{study_path}'

    # initialize data structures
    dict_of_subjects = {}

    loader = DataLoaderCreator.create(parameters['dataloader'])

    # for every subject id
    for idx, subject_id in enumerate(subject_ids):
        study_subject_path = study_path + f'P{subject_id}'
        if subject_id in dict_of_subjects.keys():
            subject_id = f'{subject_id}_{idx}'
        if qmri_method == 'IVIM':
            subject = loader(study_subject_path)
            if 'load_slices' in parameters.keys():
                signals = subject.get('signals').numpy()
                signals = slice_data(signals, parameters['slice_axis'], parameters['load_slices'])
                subject.add_image(tio.Image(tensor=torch.Tensor(signals.astype('float64'))), 'signals')
            if 'mask' in parameters.keys() and parameters['mask']['load_mask']:
                if 'permutation' in parameters['mask'].keys():
                    loaded_mask = load_mask(study_subject_path, parameters['mask']['permutation'])
                else:
                    loaded_mask = load_mask(study_subject_path)
                nonzero_mask = make_nonzero_mask(subject.get('signals').numpy())
                valid_mask = np.logical_and(loaded_mask, nonzero_mask)
            else:
                valid_mask = make_nonzero_mask(subject.get('signals').numpy())
            subject.add_image(tio.LabelMap(tensor=torch.Tensor(valid_mask)), 'valid_mask')
            dict_of_subjects[subject_id] = subject
        elif qmri_method == 'VFA':
            if 'xvals' in parameters.keys():
                subject = loader(study_subject_path, np.array(parameters['xvals'][idx]))
            else:
                subject = loader(study_subject_path)
            if 'load_slices' in parameters.keys():
                signals = subject.get('signals').numpy()
                signals = slice_data(signals, parameters['slice_axis'], parameters['load_slices'])
                subject.add_image(tio.Image(tensor=torch.Tensor(signals.astype('float64'))), 'signals')
            if 'mask' in parameters.keys() and parameters['mask']['load_mask']:
                if 'permutation' in parameters['mask'].keys():
                    loaded_mask = load_mask(study_subject_path, parameters['mask']['permutation'])
                else:
                    loaded_mask = load_mask(study_subject_path)
                nonzero_mask = make_nonzero_mask(subject.get('signals').numpy())
                valid_mask = np.logical_and(loaded_mask, nonzero_mask)
            else:
                valid_mask = make_nonzero_mask(subject.get('signals').numpy())
            subject.add_image(tio.LabelMap(tensor=torch.Tensor(valid_mask)), 'valid_mask')
            if parameters['load_non_predicted_parameters']:
                TR = parameters['non_predicted_parameters']['TR'][idx]
                subject.add_image(load_TR(subject, TR), 'TR_image')
            dict_of_subjects[subject_id] = subject
        elif qmri_method == 'DCE':
            subject = loader(study_subject_path, parameters['temporal_resolution'])
            if 'load_slices' in parameters.keys():
                signals = subject.get('signals').numpy()
                signals = slice_data(signals, parameters['slice_axis'], parameters['load_slices'])
                subject.add_image(tio.Image(tensor=torch.Tensor(signals.astype('float64'))), 'signals')
            if parameters['load_t1_map']:
                t1_map = load_t1_map(study_subject_path)
                if 'flip_t1_map' in parameters.keys() and parameters['flip_t1_map']:
                    t1_map = tio.Image(tensor=torch.tensor(np.flip(t1_map.numpy(), axis=parameters['flip_t1_map_axis']).copy()))
                if 'load_slices' in parameters.keys():
                    t1_map = tio.Image(tensor=slice_data(t1_map.numpy(), parameters['slice_axis'], parameters['load_slices']))
                subject.add_image(t1_map, 'T1_image')
            if parameters['add_aif_image']:
                subject.add_image(load_aif_image(subject, parameters), 'AIF_image')
            if 'mask' in parameters.keys() and 'load_mask' in parameters['mask'] and parameters['mask']['load_mask']:
                if 'permutation' in parameters['mask'].keys():
                    loaded_mask = load_mask(study_subject_path, parameters['mask']['permutation'])
                else:
                    loaded_mask = load_mask(study_subject_path)
                nonzero_mask = make_nonzero_mask(subject.get('signals').numpy())
                valid_mask = np.logical_and(loaded_mask, nonzero_mask)
            elif 'mask' in parameters.keys() and 'create_mask' in parameters['mask'] and parameters['mask']['create_mask']:
                mask = morphological_closing(t1_map.numpy()[0] > 550)
                mask = largest_connected_component_3d(mask)
                valid_mask = np.expand_dims(mask, 0)
            else:
                valid_mask = make_nonzero_mask(subject.get('signals').numpy())
            subject.add_image(tio.LabelMap(tensor=torch.Tensor(valid_mask)), 'valid_mask')
            dict_of_subjects[subject_id] = subject
        elif qmri_method == 'T2star':
            if 'xvals' in parameters.keys():
                subject = loader(study_subject_path, np.array(parameters['xvals'][idx]))
            else:
                subject = loader(study_subject_path)
            if 'load_slices' in parameters.keys():
                signals = subject.get('signals').numpy()
                signals = slice_data(signals, parameters['slice_axis'], parameters['load_slices'])
                subject.add_image(tio.Image(tensor=torch.Tensor(signals.astype('float64'))), 'signals')
            if 'mask' in parameters.keys() and parameters['mask']['load_mask']:
                if 'permutation' in parameters['mask'].keys():
                    loaded_mask = load_mask(study_subject_path, parameters['mask']['permutation'])
                else:
                    loaded_mask = load_mask(study_subject_path)
                nonzero_mask = make_nonzero_mask(subject.get('signals').numpy())
                valid_mask = np.logical_and(loaded_mask, nonzero_mask)
            else:
                valid_mask = make_nonzero_mask(subject.get('signals').numpy())
            subject.add_image(tio.LabelMap(tensor=torch.Tensor(valid_mask)), 'valid_mask')
            dict_of_subjects[subject_id] = subject
        else:
            raise NotImplementedError

        logging.info(f'in vivo data loaded from {subject_id} in {study_path}')
    logging.info(f'in vivo data loaded from {study_path}')

    return dict_of_subjects


def slice_data(signals, slice_axis, slice_tuple):
    """
    slices signals according to slice_axis and slice_tuple
    Args:
        signals: signals to slice
        slice_axis: slice axis
        slice_tuple: slice tuple

    Returns:
        signals: sliced signals

    """

    if slice_axis == 1:
        signals = signals[:, slice_tuple[0]:slice_tuple[1], :, :]
    elif slice_axis == 2:
        signals = signals[:, :, slice_tuple[0]:slice_tuple[1], :]
    elif slice_axis == 3:
        signals = signals[:, :, :, slice_tuple[0]:slice_tuple[1]]
    else:
        raise ValueError("slice_axis must be 1, 2 or 3")
    return signals


def load_mask(study_subject_path, permutation=None):
    """
    loads mask from study_subject_path
    Args:
        study_subject_path: path to study subject folder
        permutation: permutation of mask

    Returns:
        image: mask image
    """

    # find all files that match study path and subject id
    study_subject_path += '/mask/'

    for mask_slice_path in os.listdir(study_subject_path):
        file_path = os.path.join(study_subject_path, mask_slice_path)
        logging.info(f'start loading data from {file_path}')

        if file_path[-2:] == "gz" or file_path[-2:] == "ii":
            image = tio.Image(file_path)
            image.set_data(image.data.to(dtype=torch.float32))
            mask = image.numpy()
            if permutation is not None:
                mask = np.transpose(mask, permutation)

        elif file_path[-3:] == "dcm":
            image = tio.Image(file_path)
            mask = image.numpy()
            if permutation is not None:
                mask = np.transpose(mask, permutation)
    return mask


def load_TR(subject, TR):
    """
    loads TR image
    Args:
        subject: subject
        TR: TR value

    Returns:
        TR_image: TR image
    """

    # find all files that match study path and subject id
    shape = subject.get('signals').shape[1:]
    TR_array = np.ones(shape) * TR
    TR_array = np.expand_dims(TR_array, axis=0)
    TR_image = tio.Image(tensor=torch.Tensor(TR_array.astype('float32')))
    return TR_image


def make_nonzero_mask(signals):
    """
    creates a mask for signals that are not zero
    Args:
        signals: signals

    Returns:
        mask: mask for signals that are not zero

    """
    mask = np.min(signals, axis=0) > 0
    return np.expand_dims(mask, 0)
