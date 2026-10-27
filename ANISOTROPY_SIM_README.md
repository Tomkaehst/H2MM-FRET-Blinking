# Fluorescence Anisotropy Simulation (review copy)

Private review branch — contains the polarization-resolved smFRET simulation
notebook and validation plots, copied from the archived
`Tomkaehst/H2MM-FRET-Blinking` repository (branch
`vibe/fluorescence-anisotropy-sim-8e0852`).

## Contents

- `5_Fluorescence_Anisotropy_Simulation.ipynb` — extends the PIE timestamp
  simulation (notebook 2 of H2MM-FRET-Blinking) with polarization-resolved
  photon streams:
  - cos^2 photoselection by the z-polarized excitation pulse (r0 = 0.4),
  - isotropic rotational diffusion of the flexibly linked dye during the
    excited-state delay (exact propagator, r(t) = 0.4 exp(-t/tau_rot)),
  - dipole radiation into parallel/perpendicular analyzers, with the
    component along the detection axis not collected (as physically required),
  - four photon streams: D-parallel, D-perpendicular, A-parallel, A-perpendicular
    (DexAem and AexAem separated by nanotime windows as in notebook 3),
  - IRF-convolved time-resolved anisotropy decays, saved as polarization-
    resolved photon-HDF5 (num_polarization_ch = 2).
- `plots/fig1_perrin_validation.png` — ensemble physics check: simulated r(t)
  vs the Perrin equation.
- `plots/fig2_time_resolved_anisotropy.png` — time-resolved anisotropy decays
  reconstructed from the nanotime histograms of the four streams.
- `plots/fig3_polarized_decays.png` — polarized TCSPC decays of the four
  streams (parallel above perpendicular at early times, as expected).

## Validation numbers (3 million simulated photons)

| Quantity | Simulated | Expected |
|---|---|---|
| r(0) | 0.399-0.400 | 0.400 |
| Donor steady-state r (tau_rot 1.5 ns, tau_D 1.5 ns) | 0.1999 | 0.2000 |
| Acceptor AexAem steady-state r (tau_rot 1.5 ns, tau_A 3 ns) | 0.1331 | 0.1333 |
| Collected fraction | 0.70 | 2/3 |
| DexAem (FRET) steady-state r | 0.017 | strongly depolarized (two-stage wobble) |

Parameters of the run: E = 0.5, tau_D = 1.5 ns, tau_A = 3 ns,
tau_rot = 1.5 ns (both dyes), donor-acceptor dipole angle 60 deg,
20 MHz pulsed excitation, 4096 TCSPC bins, Gaussian IRF fallback
(the experimental 532 nm IRF csv of notebook 3 is not in the repo).

To run the notebook end-to-end, the PyBroMo diffusion trajectory of
`1_Trajectory_Simulation.ipynb` (sim_id `aa3ff9`) and the PIE emission
simulations of notebook 2 are required as input.
