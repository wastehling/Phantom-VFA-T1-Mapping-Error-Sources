import logging
import csv

import numpy as np

from utils.metrics.MetricCreator import MetricCreator

def compute_evaluation_metrics(subject, metric_dict, metrics_parameters, type, application, study, raw_results_path=None):
    """
    Compute evaluation metrics for a subject
    Args:
        subject: subject
        metric_dict: metric dictionary
        metrics_parameters: parameters for metrics
        type: type (e.g. train, val, test)
        application: application (e.g. in-vivo, simulation)
        study: study (e.g. study1, study2)
        raw_results_path: path to raw results file

    Returns:

    """
    if type not in metric_dict.keys():
        metric_dict[type] = {}
    if application not in metric_dict[type].keys():
        metric_dict[type][application] = {}
    if study not in metric_dict[type][application].keys():
        metric_dict[type][application][study] = {}

    for metric_module, metric_parameters in metrics_parameters.items():
        if metric_parameters['qmri_model'] == 'IVIM':
            parameter_names = ['D', 'Dp', 'f', 'S0', 'signals']
            supervised_parameter_names = ['D_image', 'Dp_image', 'f_image', 'S0_image', 'signals']
            pred_parameter_names = ['prediction_D', 'prediction_Dp', 'prediction_f', 'prediction_S0', 'prediction_signal_pred']
        elif metric_parameters['qmri_model'] == 'VFA':
            parameter_names = ['M0', 'T1', 'signals', 'TR']
            supervised_parameter_names = ['M0_image', 'T1_image', 'signals', 'TR_image']
            pred_parameter_names = ['prediction_M0', 'prediction_T1', 'prediction_signal_pred', None]
        elif metric_parameters['qmri_model'] == 'DCE':
            parameter_names = ['ke', 've', 'vp', 'dt', 'signals']
            supervised_parameter_names = ['ke_image', 've_image', 'vp_image', 'dt_image', 'signals']
            pred_parameter_names = ['prediction_ke', 'prediction_ve', 'prediction_vp', 'prediction_dt', 'prediction_signal_pred']

        metric_dict = compute_predefined_metrics(subject, metric_dict, metric_module, metric_parameters, parameter_names,
                                                 supervised_parameter_names, pred_parameter_names, type, application, study)
        write_raw_metrics(subject, parameter_names, supervised_parameter_names, pred_parameter_names, type, application,
                          study, raw_results_path)

    return metric_dict


def compute_predefined_metrics(subject, metric_dict, metric_module, metric_parameters, parameter_names,
                               supervised_parameter_names, pred_parameter_names, type, application, study):
    """
    Compute predefined metrics for a subject
    Args:
        subject: subject to compute predefined metrics on
        metric_dict: metric dictionary
        metric_module: metric module
        metric_parameters: parameters associated with metric computation
        parameter_names: names of parameters
        supervised_parameter_names: names of supervised parameters
        pred_parameter_names: names of predicted parameters
        type: type (e.g. train, val, test)
        application: application (e.g. in-vivo, simulation)
        study: study (e.g. study1, study2)

    Returns:
        metric_dict: updated metric dictionary

    """

    for parameter_name, supervised_parameter_name, pred_parameter_name in \
            zip(parameter_names, supervised_parameter_names, pred_parameter_names):
        
        if parameter_name not in metric_dict[type][application][study].keys():
            metric_dict[type][application][study][parameter_name] = {}

        try:
            metric = MetricCreator.create(metric_module)
            metric_name = metric_module.rsplit('.', 1)[1]
            if metric_name not in metric_dict[type][application][study][parameter_name].keys():
                metric_dict[type][application][study][parameter_name][metric_name] = []
        except ImportError:
            logging.info(f'error: {metric_module} not found')
            pass

        try:
            metric_value = metric(subject.get(pred_parameter_name).numpy(),
                                  subject.get(supervised_parameter_name).numpy(),
                                  metric_parameters['range'][parameter_name])
            metric_dict[type][application][study][parameter_name][metric_name].append(metric_value)
        except:
            logging.info(f'error: {metric_name} computation not possible')
            pass
    return metric_dict


def write_raw_metrics(subject, parameter_names, supervised_parameter_names, pred_parameter_names, type, application,
                      study, raw_results_path=None):
    """
    write metrics dict to csv in path
    Args:
        subject: subject to write raw metrics
        parameter_names: names of parameters
        supervised_parameter_names: names of supervised parameters
        pred_parameter_names: names of predicted parameters
        type: type (e.g. train, val, test)
        application: application (e.g. in-vivo, simulation)
        study: study (e.g. study1, study2)
        raw_results_path: path to raw results file
    """
    for parameter_name, supervised_parameter_name, pred_parameter_name in zip(parameter_names, supervised_parameter_names, pred_parameter_names):
        if raw_results_path is not None:
            try:
                preds = subject.get(pred_parameter_name).numpy()
            except AttributeError:
                preds = None
            try:
                supervisions = subject.get(supervised_parameter_name).numpy()
            except AttributeError:
                supervisions = np.zeros_like(preds)
            if preds is None:
                preds = np.zeros_like(supervisions)

            differences = preds - supervisions
            xvals = subject.get('xvals').numpy().squeeze()
            xval_len = subject.get('xvals').numpy().flatten().__len__()
            signals = subject.get('signals').numpy()
            pred_signals = subject.get('prediction_signal_pred').numpy()
            shape = preds.shape
            for x in range(shape[1]):
                for y in range(shape[2]):
                    for z in range(shape[3]):
                        pred = np.round(preds[0, x, y, z], 6).astype('str')
                        supervision = supervisions[0, x, y, z].astype('str')
                        difference = np.round(differences[0, x, y, z], 6).astype('str')
                        signal = signals[:, x, y, z].astype('str')
                        pred_signal = pred_signals[:, x, y, z].astype('str')

                        value = [difference, pred, supervision, signal, pred_signal]
                        value.insert(0, parameter_name)
                        value.insert(0, xval_len)
                        value.insert(0, xvals)
                        value.insert(0, study)
                        value.insert(0, application)
                        value.insert(0, type)

                        # Step 5: Write the updated data back to the CSV file
                        with open(raw_results_path, 'a') as file:
                            writer = csv.writer(file)
                            writer.writerow(value)