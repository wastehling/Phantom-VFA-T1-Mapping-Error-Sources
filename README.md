**This fork additionally contains the code for the paper *Sources and relative importance of error in VFA T1
mapping in phantoms: B1+ correction and RF spoiling* (W. Stehling et al., currently in the submission process).
The DICOM data are available on Zenodo (DOI: [10.5281/zenodo.23450766](https://doi.org/10.5281/zenodo.23450766)).
See [scripts/vfa_phantom_paper/README.md](scripts/vfa_phantom_paper/README.md) for how to reproduce its figures.**

---

Disclaimer: this software is for research purposes only. Use at your own risk.

Authors: Daan Kuppens, Oliver Gurney-Champion at Amsterdam UMC, The Netherlands
Contact: d.kuppens@amsterdamumc.nl

NCDE-based parameter estimation in Quantitative MRI
This is a repository built around the implementation of neural controlled differential equations (NCDE) (based on the work by 
P. Kidger et al., https://github.com/patrick-kidger/NeuralCDE) for quantitative MRI.

This repository reads 3D MRI data and uses it for training or evaluation of a NCDE. 

Two main scripts can be run:
- scripts/train_NCDE.py
- scripts/evaluate_NCDE.py

e.g. with the following terminal commands:
python scripts/train_NCDE.py --parameterfile parameters/train_parameterfile.yml
python scripts/evaluate_NCDE.py --parameterfile parameters/evaluate_parameterfile.yml

When a train or evaluate script is run, a timestamped folder is created storing all the results of that experiment.

the repository is structured as follows:
- models: contains the neural network models used in the NCDE. Each model can be defined by the user, it should have at 
least a forward method that takes in the input data and returns the output data, defined in the prediction file.
- scripts: contains the main scripts to run the NCDE
- parameters: contains the parameter files for the scripts
- tests: contains the tests for the repository
- utils: contains utility functions for the repository
- qmri: contains the main forward models for the repository

This project builds upon prior work from the open-source community.
We gratefully acknowledge the following projects:

torchcde
Licensed under Apache License 2.0
Portions of this code are inspired by, or directly adapted from, torchcde.

IVIMNET
Licensed under GNU General Public License v3.0
Portions of this code are directly adapted from IVIMNET.

DCE-NET
Licensed under Apache License 2.0
Portions of this code are directly adapted from DCE-NET.