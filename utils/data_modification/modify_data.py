import torch
import logging
import random
import numpy as np
import torchio as tio
from scipy.ndimage import binary_erosion


def flatten_data_structure(data):
    """
    flattens data structure
    Args:
        data: dict of data organized per type, application, study and per patient

    Returns:
        data_over_studies: list of dicts containing signals and corresponding bvalues

    """
    train_set = []
    validation_set = []
    test_set = []
    train_set_metadata = []
    validation_set_metadata = []
    test_set_metadata = []

    for type, data_per_type in data.items():
        for application, data_per_application in data_per_type.items():
            for study, data_per_study in data_per_application.items():
                data_per_study = list(data_per_study.values())
                if application == 'train':
                    train_set += data_per_study
                    train_set_metadata += len(data_per_study) * [[type, application, study]]
                if application == 'validation':
                    validation_set += data_per_study
                    validation_set_metadata += len(data_per_study) * [[type, application, study]]
                if application == 'test':
                    test_set += data_per_study
                    test_set_metadata += len(data_per_study) * [[type, application, study]]
    train_set = [[el, metadata] for el, metadata in zip(train_set, train_set_metadata)]
    validation_set = [[el, metadata] for el, metadata in zip(validation_set, validation_set_metadata)]
    test_set = [[el, metadata] for el, metadata in zip(test_set, test_set_metadata)]
    return train_set, validation_set, test_set


def multiple_lists_to_list_of_dicts(list_a, name_list_a, list_b, name_list_b, list_c=None, name_list_c=None):
    """
    combine multiple lists to a list of dicts

    Args:
        list_a: list a
        name_list_a: key name in dict
        list_b: list b
        name_list_b: key name in dict
        list_c: c
        name_list_c: key name in dict

    Returns:
        new_list: list of dicts

    """
    new_list = []
    if list_c is None:
        for idx, (el_a, el_b) in enumerate(zip(list_a, list_b)):
            new_dict = {name_list_a: el_a, name_list_b: el_b}
            new_list.append(new_dict)
    else:
        for idx, (el_a, el_b, el_c) in enumerate(zip(list_a, list_b, list_c)):
            new_dict = {name_list_a: el_a, name_list_b: el_b, name_list_c: el_c}
            new_list.append(new_dict)
    return new_list


def slice_and_multiply(subject, subject_id, multiplications, slice_factor):
    """
    slices and multiplies subjects.
    A functionality used for training to make the individual signal images, smaller, but allow more different
    configurations of the same signal.
    Args:
        subject: torchio subject
        subject_id: subject id
        multiplications: number of times to multiply the signals
        slice_factor: factor to slice the signals with

    Returns:
        list_of_subjects: list containing all subjects for a specified study who meet the subject_ids criteria

    """
    temp_dict = {}
    signals = subject.get('signals').numpy()
    try:
        mask = subject.get('valid_mask').numpy()[0, :, :, :]
    except AttributeError:
        mask = np.ones(signals.shape)[0, :, :, :]

    slice_tuple = [int(np.ceil(signals.shape[i] / slice_factor)) for i in range(1, 4)]
    #eroded_mask = binary_erosion(mask, structure=np.ones((3, 3, 1)), iterations=1)
    valid_idcs = np.where(mask[:-slice_tuple[0], :-slice_tuple[1], :-slice_tuple[2]] == 1)
    nb_valid_idcs = len(valid_idcs[0])
    for i in range(multiplications):
        try:
            random_idx = random.randint(0, nb_valid_idcs - 1)
        except ValueError as e:
            logging.error(f'Error in random index generation: {e}')
            raise
        x = valid_idcs[0][random_idx]
        y = valid_idcs[1][random_idx]
        z = valid_idcs[2][random_idx]
        sliced_signals = signals[:, x:x + slice_tuple[0], y:y + slice_tuple[1], z:z + slice_tuple[2]]
        sliced_mask = mask[x:x + slice_tuple[0], y:y + slice_tuple[1], z:z + slice_tuple[2]]
        sliced_mask = np.expand_dims(sliced_mask, axis=0)
        temp_subject_dict = {'signals': tio.Image(tensor=torch.Tensor(sliced_signals.astype('float64'))),
                             'valid_mask': tio.LabelMap(tensor=torch.Tensor(sliced_mask.astype('bool'))),
                             'xvals': subject.get('xvals')}
        if 'TR_image' in subject.get_images_dict().keys():
            sliced_TR = subject.get('TR_image').numpy()
            sliced_TR = sliced_TR[:, x:x + slice_tuple[0], y:y + slice_tuple[1], z:z + slice_tuple[2]]
            temp_subject_dict['TR_image'] = tio.Image(tensor=torch.Tensor(sliced_TR.astype('float64')))
        if 'AIF_image' in subject.get_images_dict().keys():
            sliced_AIF = subject.get('AIF_image').numpy()
            sliced_AIF = sliced_AIF[:, x:x + slice_tuple[0], y:y + slice_tuple[1], z:z + slice_tuple[2]]
            temp_subject_dict['AIF_image'] = tio.Image(tensor=torch.Tensor(sliced_AIF.astype('float64')))
        if 'T1_image' in subject.get_images_dict().keys():
            sliced_T1 = subject.get('T1_image').numpy()
            sliced_T1 = sliced_T1[:, x:x + slice_tuple[0], y:y + slice_tuple[1], z:z + slice_tuple[2]]
            temp_subject_dict['T1_image'] = tio.Image(tensor=torch.Tensor(sliced_T1.astype('float64')))
        temp_dict[f'{subject_id}_{i}'] = tio.Subject(temp_subject_dict)
    return temp_dict

