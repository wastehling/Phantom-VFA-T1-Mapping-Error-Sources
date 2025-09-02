import logging
from utils.data_processing.ProcessorCreator import ProcessorCreator


def preprocess_data(data, preprocessing_parameters):
    """
    function to preprocess data with preprocessing modules
    Args:
        data: data dict or list
        preprocessing_parameters: dict or list of parameters containing all processing modules

    Returns:
        data: data dict

    """

    def preprocess_dict_el(data, preprocessing_dict):
        """
        nested function to preprocess data with preprocessing modules, this is used if the preprocessing parameters are
        a list of dicts. A list of dicts is used if the same preprocessing modules are applied sequentially but with
        different parameters.
        Args:
            data: data dict
            preprocessing_dict: parameter dict containing all processing modules

        Returns:
            data: data dict

        """
        for preprocessing_module, preprocessing_parameters in preprocessing_dict.items():
                processor_parameters = preprocessing_parameters['processing_parameters']
                logging.info(f"Preprocessing data with {preprocessing_module} and parameters {processor_parameters}")
                processor = ProcessorCreator.create(preprocessing_module, processor_parameters)

                if 'inclusion_parameters' in preprocessing_parameters.keys():
                    inclusion_criteria = preprocessing_parameters['inclusion_parameters']

                    for type in data.keys():
                        if type in inclusion_criteria['include_type']:
                            logging.info(f"Preprocessing data of type {type}")
                            for application in data[type].keys():
                                if application in inclusion_criteria['include_application']:
                                    logging.info(f"Preprocessing data of application {application}")
                                    for study in data[type][application].keys():
                                        if study in inclusion_criteria['include_study']:
                                            logging.info(f"Preprocessing data of study {study}")
                                            for subject_idx, subject in data[type][application][study].items():
                                                logging.info(f"Preprocessing data of subject {subject_idx}")
                                                subject = processor(subject)
                                                data[type][application][study][subject_idx] = subject

                else:
                    processor = ProcessorCreator.create(preprocessing_module, processor_parameters)
                    for type in data.keys():
                        logging.info(f"Preprocessing data of type {type}")
                        for application in data[type].keys():
                            logging.info(f"Preprocessing data of application {application}")
                            for study in data[type][application].keys():
                                logging.info(f"Preprocessing data of study {study}")
                                for subject_idx, subject in data[type][application][study].items():
                                    logging.info(f"Preprocessing data of subject {subject_idx}")
                                    subject = processor(subject)
                                    data[type][application][study][subject_idx] = subject
        return data

    if isinstance(preprocessing_parameters, list):
        for dict_el in preprocessing_parameters:
            data = preprocess_dict_el(data, dict_el)
    else:
        data = preprocess_dict_el(data, preprocessing_parameters)

    return data


def get_weights(data):
    weights = []

    for dataset in data:
            weight = dataset['weight']
            nr_of_patients = len(dataset['patient_ids'])
            muls = dataset['slice_and_multiply']['multiplications']

            weights += [weight] * (nr_of_patients * muls)

    return weights

