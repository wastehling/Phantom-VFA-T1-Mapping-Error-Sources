"""
Step 2 of reproducing the VFA phantom paper: figures 2-6, S1 and S2.

Figures 3-6 and S2 are made from <output_folder>/fitted_values_T1.csv (created by fit_phantom_t1.py); figures 2 and S1
show the B1+ maps directly from the DICOM export. Each figure has a GSP (top) and CM (bottom) panel, which are saved
separately and combined vertically, as in the manuscript. The final figures are saved in the PLOS figure format.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from qmri.data_loading.vfa.load_vfa_philips_dicom import read_dicom_volume, load_b1_map
from scripts.vfa_phantom_paper.fit_phantom_t1 import find_session
from scripts.vfa_phantom_paper.plotting import plot_phantom_fit, plot_b1_maps

# PLOS figure requirements: RGB TIFF with LZW compression, 300 dpi, at most 7.5 x 8.75 inch
PLOS_DPI = 300
PLOS_MAX_SIZE = (2250, 2625)


def combine_vertical(paths, out_path, gap=10):
    images = [Image.open(p).convert('RGB') for p in paths]
    canvas = Image.new('RGB', (max(im.width for im in images), sum(im.height for im in images) + gap * (len(images) - 1)),
                       (255, 255, 255))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height + gap
    canvas.save(out_path, compression='tiff_lzw')


def to_plos_format(path):
    """Rescales a figure to fit the PLOS maximum size and saves it as RGB TIFF (LZW) at 300 dpi."""
    image = Image.open(path).convert('RGB')
    image.thumbnail(PLOS_MAX_SIZE, Image.LANCZOS)
    image.save(path, compression='tiff_lzw', dpi=(PLOS_DPI, PLOS_DPI))


def b1_map_on_mask_slice(parameters, phantom, study):
    """B1+ map (percent) on its own grid, at the slice closest to the centre of the VFA mask slices."""
    study_cfg = parameters['studies'][phantom][study]
    session_index = find_session(parameters['data_root'], parameters['sessions'][study_cfg['session']])
    _, vfa_affine, _ = read_dicom_volume(session_index[study_cfg['vfa']])
    mask_slice = np.mean(parameters['phantoms'][phantom]['mask_slices'])
    centre = vfa_affine @ np.array([0, 0, mask_slice, 1])
    b1, b1_affine = load_b1_map(session_index[study_cfg['b1']])
    b1_slice = int(np.clip(np.round((np.linalg.inv(b1_affine) @ centre)[2]), 0, b1.shape[0] - 1))
    return b1[b1_slice] * 100


def make_figures(parameters):
    output_folder = parameters['output_folder']
    figure_folder = os.path.join(output_folder, 'figures')
    os.makedirs(os.path.join(figure_folder, 'panels'), exist_ok=True)
    df = pd.read_csv(os.path.join(output_folder, 'fitted_values_T1.csv'))

    for figure, cfg in parameters['figures'].items():
        panels = []
        for letter, phantom in zip('ab', ['GSP', 'CM']):
            sub = df[(df['phantom'] == phantom) & df['study_name'].isin(cfg['studies'])]
            fig_id, name = figure.split('_', 2)[1], figure.split('_', 2)[2]
            path = os.path.join(figure_folder, 'panels', f'fig_{fig_id}{letter}_{phantom}_{name}.tiff')
            plot_phantom_fit(sub, cfg, f"{phantom} {cfg['title']}", path)
            panels.append(path)
        combine_vertical(panels, os.path.join(figure_folder, f'{figure}.tiff'))
        to_plos_format(os.path.join(figure_folder, f'{figure}.tiff'))

    for figure, cfg in parameters['b1_map_figures'].items():
        b1_maps = {phantom: [b1_map_on_mask_slice(parameters, phantom, study) for study in cfg['studies']]
                   for phantom in ['GSP', 'CM']}
        plot_b1_maps(b1_maps, cfg['column_titles'], os.path.join(figure_folder, f'{figure}.tiff'),
                     arrow=cfg.get('arrow_GSP_DREAM'))
        to_plos_format(os.path.join(figure_folder, f'{figure}.tiff'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameterfile', type=str, default='parameters/vfa_phantom_paper.yml')
    parser.add_argument('--data_root', type=str, default=None, help='folder with the DICOM data, overrides the parameterfile')
    args = parser.parse_args()
    with open(args.parameterfile) as file:
        parameters = yaml.load(file, Loader=yaml.Loader)
    if args.data_root is not None:
        parameters['data_root'] = args.data_root
    if not os.path.isdir(parameters['data_root']):
        raise SystemExit(f"Data folder {parameters['data_root']} not found; download the data and pass --data_root")
    make_figures(parameters)
