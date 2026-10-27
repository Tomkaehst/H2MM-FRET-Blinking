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
- 20 % glycerol estimate (section 3): viscosity 1.8x -> D drops from 10 to
  5.6 um²/s, so observation times lengthen 1.8x (mean 104 -> 188 ms, ~5.4
  frames per molecule); the refractive-index mismatch (sample 1.359 behind a
  water-design objective, ni0 = 1.333) broadens the detection PSF laterally
  (0.35 -> 0.45 um FWHM) and axially (1.75 -> 2.00 um), doubles the axial
  asymmetry and cuts the Strehl-like peak ratio to ~0.44.
- observation-time statistics per molecule (section 6): each contiguous
  stay inside the sheet (excitation > 1/e²) is one crossing; ~31% of
  crossings are detectable (>= 100 photons), with a mean observation time
  of ~104 ms (~3 camera frames), limited by diffusion out of the sheet.
- `lightsheet_sim_plots/` — validation and result figures (also embedded
  in the executed notebook):
  - `fig1_psf_xy_xz_projections.png` — XY focal-plane slice, XY max
    projection, XZ projection and lateral/axial line profiles
  - `fig2_glycerol_effect.png` — axial and lateral detection PSF, water
    vs 20 % glycerol (RI mismatch), and the mismatched effective PSF
  - `fig3_camera_frames.png`, `fig4_intensity_stats.png`,
    `fig5_observation_time_stats.png`

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
| Sheet crossings per particle per 100 s | ~72 (10726 / 15) | diffusion in/out of the ~2 um sheet |
| Detectable crossings (>= 100 photons) | 31% | — |
| Observation time, detectable crossings | mean 104 ms, median 41 ms, max 1.37 s | ~ w0²/D scaling |
| Camera frames per detectable molecule | mean 3.01, median 2, P90 7 | t_frame = 50 ms |
| 20 % glycerol: D | 5.56 µm²/s | 10 / 1.8 (viscosity ratio) |
| 20 % glycerol: mean observation time | 188 ms | 104 ms x 1.8 |
| 20 % glycerol: lateral det. FWHM | 0.45 µm | 0.35 µm (water, matched) |
| 20 % glycerol: axial det. FWHM | 2.00 µm | 1.75 µm (water, matched) |
| 20 % glycerol: Strehl-like peak ratio | ~0.44 | spherical aberration (ns 1.359 vs ni0 1.333) |

Parameters of the run: detection NA 0.8, em. 580 nm, ex. 488 nm,
sheet NA 0.08 (w0 = 1.94 µm), 50 kcps/particle at sheet center,
10 cps/pixel background, 2000 frames of 128 × 128 pixels.

To run the notebook end-to-end, only `pybromo`, `psfmodels`, `tables`
(PyTables) and `matplotlib` are required.
