import logging
import csv
import numpy as np
import importlib
import torch


class MetricCreator:
    """
    class dedicated to create metrics
    """
    @staticmethod
    def create(class_name, parameters):
        package, class_name = class_name.rsplit('.', 1)
        model_module = importlib.import_module(package)
        class_object = getattr(model_module, class_name)
        return class_object(**parameters)


class Metric:
    """
    Base class for metrics
    """
    def __init__(self, parameters):
        self.parameters = parameters
        self.metric = None
        self.average_metric = None
        self.average_metrics = []
        self.steps = 0
        self.cum_metric = 0

    def compute(self, prediction, supervision):
        raise NotImplementedError

    def update(self):
        self.cum_metric += self.metric
        self.steps += 1

    def clear_cache(self):
        self.average_metrics.append(self.average_metric)
        self.cum_metric = 0
        self.steps = 0
        self.average_metric = None

    def average(self):
        self.average_metric = self.cum_metric / self.steps


class RMSE(Metric):
    """
    Root Mean Squared Error
    """
    def __init__(self, parameters):
        super().__init__(parameters)
        self.name = 'RMSE'

    def compute(self, prediction, supervision):
        self.metric = np.sqrt(np.nanmean(np.square(np.subtract(prediction, supervision))))
        if self.parameters['normalization_method'] == 'mean':
            self.metric /= np.mean(supervision)
        elif self.parameters['normalization_method'] == 'range':
            self.metric /= (np.abs(np.max(supervision) - np.min(supervision)))
        elif self.parameters['normalization_method'] is None:
            pass
        return self.metric

    def log(self, parameter_name):
        logging.info(f'RMSE of {parameter_name} is {self.average_metric:.6f}')


class MSE(Metric):
    """
    Mean Squared Error
    """
    def __init__(self, parameters):
        super().__init__(parameters)
        self.name = 'MSE'

    def compute(self, prediction, supervision):
        self.metric = np.nanmean(np.square(np.subtract(prediction, supervision)))
        if self.parameters['normalization_method'] == 'mean':
            self.metric /= np.mean(supervision)
        elif self.parameters['normalization_method'] == 'range':
            self.metric /= (np.abs(np.max(supervision) - np.min(supervision)))
        elif self.parameters['normalization_method'] is None:
            pass
        return self.metric

    def log(self, parameter_name):
        logging.info(f'MSE of {parameter_name} is {self.average_metric:.6f}')


class MAE(Metric):
    """
    Mean Absolute Error
    """
    def __init__(self, parameters):
        super().__init__(parameters)
        self.name = 'MAE'

    def compute(self, prediction, supervision):
        self.metric = np.nanmean(np.abs(np.subtract(prediction, supervision)))
        if self.parameters['normalization_method'] == 'mean':
            self.metric /= np.mean(supervision)
        elif self.parameters['normalization_method'] == 'range':
            self.metric /= (np.abs(np.max(supervision) - np.min(supervision)))
        elif self.parameters['normalization_method'] is None:
            pass
        return self.metric

    def log(self, parameter_name):
        logging.info(f'MAE of {parameter_name} is {self.average_metric:.6f}')


class Bias(Metric):
    """
    Bias of prediction minus supervision
    """
    def __init__(self, parameters):
        super().__init__(parameters)
        self.name = 'Bias'

    def compute(self, prediction, supervision):
        self.metric = np.nanmean(np.subtract(prediction, supervision))
        if self.parameters['normalization_method'] == 'mean':
            self.metric /= np.mean(supervision)
        elif self.parameters['normalization_method'] == 'range':
            self.metric /= (np.abs(np.max(supervision) - np.min(supervision)))
        elif self.parameters['normalization_method'] is None:
            pass
        return self.metric

    def log(self, parameter_name):
        logging.info(f'Bias of {parameter_name} is {self.average_metric:.6f}')


class Variance(Metric):
    """
    Variance of difference between prediction and supervision
    """
    def __init__(self, parameters):
        super().__init__(parameters)
        self.name = 'Variance'

    def compute(self, prediction, supervision):
        self.metric = np.nanvar(np.subtract(prediction, supervision))
        if self.parameters['normalization_method'] == 'mean':
            self.metric /= np.mean(supervision)
        elif self.parameters['normalization_method'] == 'range':
            self.metric /= (np.abs(np.max(supervision) - np.min(supervision)))
        elif self.parameters['normalization_method'] is None:
            pass
        return self.metric

    def log(self, parameter_name):
        logging.info(f'Variance of {parameter_name} is {self.average_metric:.6f}')


def create_metric_list(parameters):
    """
    wrapper function to create metric list
    Args:
        parameters: parameters for metrics

    Returns:
        metric_list: list of metrics

    """
    metric_list = []
    for metric_name, metric_parameters in parameters.items():
        metric_list.append(MetricCreator.create(metric_name, metric_parameters))
    return metric_list


def write_metrics(metrics_dict, path):
    """
    write metrics dict to csv in path
    Args:
        metrics_dict: metrics dictionary to write
        path: path to write csv to

    """
    # insert header into list for readability
    metric_path = path + '/metrics.csv'
    with open(metric_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        for metric_name, metric in metrics_dict.items():
            wr.writerow([metric_name])
            for key, value in metric.items():
                wr.writerow([key, value])

