import logging
import csv
import numpy as np

from models.FunctionCreator import FunctionCreator
from qmri.visualization.VisualizationCreator import VisualizationCreator
from utils.metrics.compute_metrics import compute_evaluation_metrics

import matplotlib
matplotlib.use('Agg')


def evaluation(model, parameters, data, device, target_folder):
    """
    evaluates a model on the data from test_loader and test_sim_loader
    Args:
        model: model to use for evaluation
        parameters: parameters
        data: dict containing  object containing in vivo test data
        device: device, either cuda or cpu
        target_folder: folder to save results

    Returns:
        data: dict containing predictions
        metric_dict: dict containing metrics
    """
    metric_dict = {}
    raw_results_path = target_folder + '/raw_results.csv'

    with open(raw_results_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'application', 'study', 'xvals', 'sequence_length', 'parameter', 'values', 'preds',
                         'supervision', 'signals', 'pred_signals'])

    for type in data.keys():
        type_folder = target_folder + '/plots/' + type
        for application in data[type].keys():
            application_folder = type_folder + '/' + application
            for study in data[type][application].keys():
                study_folder = application_folder + '/' + study
                for subject_id, subject in data[type][application][study].items():
                    subject_folder = study_folder + '/' + str(subject_id)
                    logging.info(f'predicting results for subject {subject_id} of study {study}')
                    prediction_function = FunctionCreator.create(parameters['model']['prediction_function'])
                    try:
                        prediction_subject = prediction_function(model, subject, parameters, device)
                        for image_name, image in prediction_subject.get_images_dict(intensity_only=False).items():
                            image.affine = subject.get('signals').affine
                            prediction_subject.add_image(image, image_name)

                        data[type][application][study][subject_id] = prediction_subject

                        if 'metrics' in parameters.keys():
                            try:
                                metric_dict = compute_evaluation_metrics(prediction_subject, metric_dict,
                                                                         parameters['metrics'], type, application,
                                                                         study, raw_results_path)
                            except Exception as e:
                                logging.info(f'error: {str(e)}, metrics not possible for subject {subject_id} of study'
                                             f' {study}')
                                pass
                        else:
                            logging.info('no metrics specified')
                            pass

                        for visualization_module, module_parameters in parameters['visualization'].items():
                            visualizer = VisualizationCreator.create(visualization_module)
                            function_name = visualization_module.rsplit('.', 1)[1]
                            try:
                                logging.info(f'info: {function_name} is evaluated now!')
                                visualizer(prediction_subject, module_parameters, subject_folder, name=function_name)
                            except Exception as e:
                                logging.info(f'error: {str(e)}, {function_name} plots not possible')
                                pass
                    except Exception as e:
                        logging.error(f'error: {str(e)}, prediction not possible for subject {subject_id} of '
                                      f'study {study}', exc_info=True)
                        pass

                if metric_dict:
                    parameter_keys = list(metric_dict[type][application][study].keys())
                    for parameter in parameter_keys:
                        metric_keys = list(metric_dict[type][application][study][parameter].keys())
                        for metric in metric_keys:
                            metric_dict[type][application][study][parameter][metric] = \
                                metric_dict[type][application][study][parameter][metric]
                            metric_dict[type][application][study][parameter][metric + '_mean'] = \
                                np.average(np.array(metric_dict[type][application][study][parameter][metric]))

    return data, metric_dict
