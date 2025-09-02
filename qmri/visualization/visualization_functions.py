import os
import random
import logging

import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import torchio

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go

from qmri.forward_model.ivim import ivim_parameters_to_signal
from qmri.forward_model.vfa import vfa_parameters_to_signal
from qmri.forward_model.dce import dce_parameters_to_signal


def store_images_as_nii(subject, visualization_parameters, path, name):
    """
    store parameter maps as nifti images
    Args:
        subject: Subject
        visualization_parameters: visualization parameters
        path: str of where to store parameter maps
        name: str of parameter map name

    """

    target_folder = path + '/parameter_maps/'
    os.makedirs(target_folder, exist_ok=True)
    images_dict = subject.get_images_dict()
    if 'valid_mask' in images_dict.keys():
        signal_mask = subject.get('valid_mask')
        for key in images_dict.keys():
            image = torch.mul(signal_mask.data, subject.get(key).data)
            image_to_save = torchio.Image(tensor=image)
            image_to_save.save(target_folder + name + '_' + key + ".nii.gz")
    else:
        for key in images_dict.keys():
            subject.get(key).save(target_folder + name + '_' + key + ".nii.gz")

