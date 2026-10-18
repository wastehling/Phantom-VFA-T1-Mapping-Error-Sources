# Sources of error in VFA T1 mapping in phantoms

Reproduces figures 2–6, S1 and S2 of *Sources and relative importance of error in VFA T1
mapping in phantoms: B1+ correction and RF spoiling* (W. Stehling et al.) from the Philips DICOM exports.

## Data

The DICOM data (`DicomData.zip`, 171 MB) are available on Zenodo (DOI: [10.5281/zenodo.23450766](https://doi.org/10.5281/zenodo.23450766)). The zip contains only the series
used in the paper. Download it and unzip it into one folder, which then contains one subfolder per exam:

```
unzip DicomData.zip -d /path/to/DicomData
```

```
/path/to/DicomData/
├── M_P5_20260517_M_P5_20260517/   (GSP phantom, day 2)
├── M_P5_20260518_M_P5_20260518/   (GSP phantom, day 1)
├── M_P6_20260517_M_P6_20260517/   (CM phantom, day 2)
└── M_P6_20260518_M_P6_20260518/   (CM phantom, day 1)
```

The series inside each exam may be zipped (`<series number>_<description>.zip`) or unzipped folders.

## Run

From the repository root:

```
python -m scripts.vfa_phantom_paper.fit_phantom_t1 --data_root /path/to/DicomData
python -m scripts.vfa_phantom_paper.make_figures --data_root /path/to/DicomData
```

Instead of `--data_root`, `data_root` can be set in `parameters/vfa_phantom_paper.yml`.

Requirements (the rest of the NCDE repository, e.g. torch, is not needed):

```
pip install -r scripts/vfa_phantom_paper/requirements.txt
```

Takes about 30 s on a laptop.

Outputs in `output/vfa_phantom_paper/`:
- `fitted_values_T1.csv`: mean T1 per vial and study, with reference values and errors
- `masks/`: vial masks (NIfTI, on the VFA grid) and a QC image per study (masks on the lowest flip angle, B1+ map)
- `figures/`: the combined figures (GSP top, CM bottom; PLOS format: RGB TIFF, LZW, 300 dpi, max. 2250 x 2625 px)
  and the separate panels (600 dpi)

## Method

- **Data**: one exam per phantom per day (`sessions` in the parameter file). VFA: 6 flip angles (2–25°) stored as
  consecutive series; Philips floating point values are used, as the scale factors differ per series.
- **B1+**: the `M_B1` image of the AFI / SDAM / DREAM series (percent), resampled onto the VFA slice using the DICOM
  geometry and clipped to [0.75, 1.25].
- **Fit**: voxel-wise SPGR fit (`models/classical_fitting/LSQ/least_squares.fit_least_squares_vfa`) with flip angle
  = nominal × B1+, M0 in [0, 1e8], T1 in [10, 3000] ms.
- **Masks**: on the three central slices of the 15-slice VFA slab (6 mm, matching the 2D SDAM/DREAM slice), vials are
  detected automatically on the lowest flip angle (`vial_masks.py`) and masked with circles 2 px smaller than the
  inner vial radius. Vial means are sorted and matched to the sorted reference values.
- **Reference values**: NIST measurements interpolated to the phantom temperature (21 °C), from `reference_values/`.
  For the CM phantom, the four inner spheres (T1 < 100 ms) are excluded.
