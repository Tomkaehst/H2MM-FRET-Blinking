# Light-Sheet smFRET Camera Simulation (review copy)

Private review branch — contains the light-sheet camera simulation notebook
and validation plots, extending the H2MM-FRET-Blinking simulation suite
(branch `light_sheet_smfret`).

## Contents

- `5_LightSheet_Camera_Simulation.ipynb` — replaces the confocal PSF +
  TCSPC-timestamp detection of notebooks 1–2 with a **light-sheet PSF +
  camera**:
  - light-sheet PSF computed with the `psfmodels` module (vectorial
    Gibson–Lanni widefield detection PSF on a z stack) multiplied by a
    Gaussian excitation sheet (w0 = λ_ex / (π·NA_sheet)),
  - wrapped in `LightSheetPSF` (`lightsheet_psf.py`), which implements the
    pybromo PSF interface (`eval`, `eval_xz`, `hash`, `to_hdf5`) and is
    stored in the pybromo trajectory HDF5 file,
  - particle diffusion simulated with pybromo at camera parameters
    (100 s, 1 ms step, 50 ms frame integration, 15 particles,
    D = 10 µm²/s, box = 128 × 128 px at 0.2 µm/px),
  - camera frames rendered by binning the particle emission on the pixel
    grid per z plane, convolving with the sum-normalized defocus-dependent
    detection kernel, and Poisson-sampling into uint16 frames,
  - physics-validation section comparing the simulated PSF to the
    analytical expectations before any data is generated.
- `lightsheet_psf.py` — the `LightSheetPSF` class and the camera-frame
  rendering (`render_frame_expected`, `render_camera_frames`).
- `lightsheet_sim_plots/` — validation and result figures (also embedded
  in the executed notebook).

## Validation numbers

| Quantity | Simulated | Expected |
|---|---|---|
| In-focus detection FWHM (0.05 µm grid) | 0.350 µm | 0.51·λ/NA = 0.370 µm |
| Effective axial FWHM | 1.250 µm | < detection-only (1.750 µm) and < sheet-only (4.572 µm) |
| Rendered photons, particle at z = 0 | 10000.0 | 10000.0 |
| Rendered photons, particle at z = +1 µm | 5883.2 | 5883.2 (excitation e^(−2(z/w0)²)) |
| Rendered photons, particle at z = +2 µm | 1198.0 | 1198.0 |
| Camera spot FWHM at z = 0 | 0.60 µm (2 px, pixel-limited) | ≈ 0.35 µm (sub-pixel) |
| Camera spot FWHM at z = +2 µm | 1.80 µm | broadened (defocus) |
| Photon conservation | exact at every defocus | sum-normalized kernels |

Parameters of the run: detection NA 0.8, em. 580 nm, ex. 488 nm,
sheet NA 0.08 (w0 = 1.94 µm), 50 kcps/particle at sheet center,
10 cps/pixel background, 2000 frames of 128 × 128 pixels.

To run the notebook end-to-end, only `pybromo`, `psfmodels`, `tables`
(PyTables) and `matplotlib` are required.
