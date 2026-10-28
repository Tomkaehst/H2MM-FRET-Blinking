# Light-sheet PSF built with the `psfmodels` module, wrapped in a
# pybromo-compatible PSF class.
#
# The light-sheet PSF is modeled as the product of:
#   1. the excitation light sheet: a Gaussian sheet, thin in z, uniform
#      in x and y (illumination NA of the sheet objective)
#   2. the detection PSF: a widefield (camera-side) PSF computed with the
#      vectorial model of `psfmodels` (Gibson-Lanni), evaluated on a z stack
#      to include defocus
#
# The class implements the same interface as pybromo.psflib.GaussianPSF /
# NumericPSF (eval, eval_xz, hash, to_hdf5), so it can be passed to
# pbm.ParticlesSimulation and stored in the simulation HDF5 file.

import hashlib

import numpy as np
import psfmodels as psfm
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve


class LightSheetPSF:
    """Light-sheet PSF computed with psfmodels and usable in pybromo.

    The effective PSF is `excitation_sheet(z) * detection_psf(x, y, z)`,
    computed on a 3D grid (z, y, x) and evaluated by linear interpolation.

    Parameters
    ----------
    dxy : float
        Lateral pixel size of the PSF grid in micrometers (sample space).
        Use the camera pixel size to allow direct convolution with camera
        frames.
    nx : int
        Number of lateral pixels per axis (prefer odd numbers).
    dz : float
        Step of the PSF z grid in micrometers.
    nz_half : int
        Number of z planes on each side of the focal plane. The z grid
        spans [-nz_half * dz, +nz_half * dz].
    det_NA : float
        Numerical aperture of the detection objective.
    em_wvl : float
        Emission wavelength in micrometers.
    ex_wvl : float
        Excitation (light sheet) wavelength in micrometers.
    sheet_NA : float
        Numerical aperture of the light-sheet illumination. Defines the
        Gaussian sheet thickness: 1/e^2 intensity thickness
        `2 * w0` with `w0 = ex_wvl / (pi * sheet_NA)` (micrometers).
    ns : float
        Sample refractive index.
    ni0, ni, ng0, ng, tg, tg0, ti0 :
        Gibson-Lanni model parameters of psfmodels (see psfmodels.make_psf).
    psf_pytables : pytables array, optional
        If not None, load the PSF from a pytables array saved by
        `to_hdf5` (overrides all other arguments).
    """

    kind = 'lightsheet'

    def __init__(self, dxy=0.2, nx=51, dz=0.25, nz_half=24,
                 det_NA=0.8, em_wvl=0.58, ex_wvl=0.488, sheet_NA=0.08,
                 ns=1.333, ni0=1.333, ni=1.333, ng0=1.333, ng=1.333,
                 tg=170.0, tg0=170.0, ti0=150.0,
                 sheet_extent_x=np.inf, sheet_extent_y=np.inf,
                 psf_pytables=None):
        """`sheet_extent_x`, `sheet_extent_y`: 1/e^2 half-widths (um) of the
        illuminated field along the sheet (x) and along the beam propagation
        direction (y). `np.inf` (default) = uniform, infinitely wide sheet.
        A finite `sheet_extent_x` produces the edge vignetting of a real,
        finite light sheet on the camera frames."""
        self.fname = 'lightsheet_psf'
        if psf_pytables is not None:
            if 'kind' in psf_pytables.attrs:
                assert str(psf_pytables.attrs['kind']) == self.kind
            self.psf_volume = psf_pytables[:]
            for name in ['dxy', 'nx', 'dz', 'nz_half', 'det_NA', 'em_wvl',
                         'ex_wvl', 'sheet_NA', 'ns', 'ni0', 'ni']:
                setattr(self, name, psf_pytables.get_attr(name))
            self.sheet_extent_x = psf_pytables.attrs['sheet_extent_x'] \
                if 'sheet_extent_x' in psf_pytables.attrs else np.inf
            self.sheet_extent_y = psf_pytables.attrs['sheet_extent_y'] \
                if 'sheet_extent_y' in psf_pytables.attrs else np.inf
        else:
            self.dxy, self.nx = dxy, nx
            self.dz, self.nz_half = dz, nz_half
            self.det_NA, self.em_wvl = det_NA, em_wvl
            self.ex_wvl, self.sheet_NA = ex_wvl, sheet_NA
            self.ns, self.ni0, self.ni = ns, ni0, ni
            self.sheet_extent_x = sheet_extent_x
            self.sheet_extent_y = sheet_extent_y
            self._build_psf(ns=ns, ni0=ni0, ni=ni, ng0=ng0, ng=ng,
                            tg=tg, tg0=tg0, ti0=ti0)
        self._build_axes()
        self._build_interpolator()

    ## PSF computation

    def _build_psf(self, ns, ni0, ni, ng0, ng, tg, tg0, ti0):
        """Compute the effective light-sheet PSF volume with psfmodels."""
        # Detection PSF: widefield (no pinhole) vectorial PSF on a z stack
        det_psf = psfm.make_psf(
            2 * self.nz_half + 1, self.nx,
            dxy=self.dxy, dz=self.dz, NA=self.det_NA, wvl=self.em_wvl,
            ns=ns, ni0=ni0, ni=ni, ng0=ng0, ng=ng,
            tg=tg, tg0=tg0, ti0=ti0,
            model='vectorial', normalize=True)
        # z axis of make_psf is relative defocus from the focal plane
        self.detection_psf = det_psf.astype(np.float64)

        # Excitation light sheet: Gaussian intensity profile in z
        z_um = self.z_grid_um
        w0 = self.ex_wvl / (np.pi * self.sheet_NA)
        self.sheet_waist = w0
        self.excitation_profile = np.exp(-2 * (z_um / w0) ** 2)

        # Effective PSF = excitation * detection
        psf = self.detection_psf * self.excitation_profile[:, None, None]
        psf /= psf.max()
        self.psf_volume = psf.astype(np.float32)

    @property
    def z_grid_um(self):
        return (np.arange(2 * self.nz_half + 1) - self.nz_half) * self.dz

    def _build_axes(self):
        c = self.nx // 2
        self.x_um = (np.arange(self.nx) - c) * self.dxy
        self.y_um = self.x_um.copy()
        self.z_um = self.z_grid_um
        # axes in meters, used by the interpolation
        self.x_m = self.x_um * 1e-6
        self.y_m = self.y_um * 1e-6
        self.z_m = self.z_um * 1e-6

    def _build_interpolator(self):
        self._interp = RegularGridInterpolator(
            (self.z_m, self.y_m, self.x_m), self.psf_volume,
            method='linear', bounds_error=False, fill_value=0.0)

    ## pybromo PSF interface

    def eval(self, x, y, z):
        """Evaluate the PSF at positions (x, y, z) in meters.

        Accepts scalars or (broadcastable) arrays.
        """
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                       np.asarray(y, dtype=np.float64),
                                       np.asarray(z, dtype=np.float64))
        pts = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=-1)
        v = self._interp(pts)
        return v.reshape(z.shape)

    def eval_xz(self, x, z):
        """Evaluate the PSF along the (x, z) plane at y = 0.

        Note: pybromo assumes rotational symmetry around z when calling
        this method. A light-sheet PSF is NOT rotationally symmetric
        (the sheet is thin only in z); here the lateral argument is
        interpreted as the x coordinate along the sheet plane. Use
        `eval(x, y, z)` for the full 3D PSF.
        """
        return self.eval(x, 0.0, z)

    def hash(self):
        """Return an hash string computed on the PSF data."""
        s = (f'LightSheetPSF(dxy={self.dxy}, nx={self.nx}, dz={self.dz}, '
             f'nz_half={self.nz_half}, det_NA={self.det_NA}, '
             f'em_wvl={self.em_wvl}, ex_wvl={self.ex_wvl}, '
             f'sheet_NA={self.sheet_NA}, ns={self.ns}, '
             f'ni0={self.ni0}, ni={self.ni})')
        h = hashlib.md5(s.encode())
        h.update(np.ascontiguousarray(self.psf_volume).tobytes())
        return h.hexdigest()

    def __repr__(self):
        return (f'LightSheetPSF(dxy={self.dxy}um, nx={self.nx}, '
                f'dz={self.dz}um, nz_half={self.nz_half}, '
                f'det_NA={self.det_NA}, em_wvl={self.em_wvl}um, '
                f'ex_wvl={self.ex_wvl}um, sheet_NA={self.sheet_NA})')

    def to_hdf5(self, file_handle, parent_node='/'):
        """Store the PSF volume in `file_handle` (pytables) in `parent_node`.

        The PSF volume is stored as an array named `lightsheet_psf` with
        shape (nz, nx, nx), along with the model parameters as attributes.
        """
        tarray = file_handle.create_array(
            parent_node, name=self.fname, obj=self.psf_volume,
            title='Light-sheet PSF volume (z, y, x), psfmodels vectorial')
        for name in ['dxy', 'nx', 'dz', 'nz_half', 'det_NA', 'em_wvl',
                     'ex_wvl', 'sheet_NA', 'ns', 'ni0', 'ni', 'kind']:
            file_handle.set_node_attr(tarray, name, getattr(self, name))
        return tarray

    @classmethod
    def from_pytables(cls, psf_pytables):
        """Load a LightSheetPSF from a pytables array saved by `to_hdf5`."""
        return cls(psf_pytables=psf_pytables)

    ## Helpers for camera-frame rendering

    @property
    def focal_plane_psf(self):
        """2D detection PSF on the focal plane (nx, nx), sum-normalized.

        Suited as convolution kernel to distribute photon counts over
        camera pixels. See `detection_kernel` for the z-dependent kernel.
        """
        return self.detection_kernel(0.0)

    def detection_kernel(self, z):
        """Sum-normalized 2D detection PSF at defocus `z` (meters).

        The detection PSF of a widefield/camera system broadens with
        defocus; each z plane of the precomputed psfmodels volume is
        sum-normalized so that photon counts are conserved (the particle
        emits the same number of photons, the spot just spreads out).
        """
        z_um = np.asarray(z, dtype=np.float64) * 1e6
        iz = np.clip(np.round(z_um / self.dz).astype(int) + self.nz_half,
                     0, 2 * self.nz_half)
        k = self.detection_psf[iz].astype(np.float64)
        s = k.sum(axis=(-1, -2), keepdims=True)
        return np.divide(k, s, out=np.zeros_like(k), where=s > 0)

    @property
    def rayleigh_range(self):
        """Rayleigh range of the sheet beam (um): pi * w0^2 / lambda."""
        return np.pi * self.sheet_waist**2 / self.ex_wvl

    def excitation_weights(self, z, x=None, y=None):
        """Excitation sheet intensity at particle positions (meters).

        Gaussian-beam light sheet, thin in z and propagating along y:
        - thickness w(y) = w0 * sqrt(1 + (y/zR)^2), with the Rayleigh range
          zR = pi*w0^2/lambda (on-axis intensity stays 1 along y; the beam
          divergence only thickens the sheet)
        - `sheet_extent_x` (um): 1/e^2 half-width of the illuminated field
          along the sheet (x) — the finite footprint of the illumination
          optics; np.inf = uniform
        - `sheet_extent_y` (um): optional 1/e^2 envelope along the
          propagation direction (y); np.inf = none

        `x` and `y` are optional for backward compatibility: without them
        the profile is the z-only Gaussian (uniform laterally).
        """
        z = np.asarray(z, dtype=np.float64)
        if y is not None:
            y = np.broadcast_to(np.asarray(y, dtype=np.float64), z.shape)
        else:
            y = np.zeros_like(z)
        w_y = self.sheet_waist * np.sqrt(1 + (y * 1e6 / self.rayleigh_range)**2)
        w = np.exp(-2 * (z * 1e6 / w_y)**2)
        if x is not None and np.isfinite(self.sheet_extent_x):
            x = np.broadcast_to(np.asarray(x, dtype=np.float64), z.shape)
            w = w * np.exp(-2 * (x * 1e6 / self.sheet_extent_x)**2)
        if np.isfinite(self.sheet_extent_y):
            w = w * np.exp(-2 * (y * 1e6 / self.sheet_extent_y)**2)
        return w

    def render_frame_expected(self, x, y, z, t_slice=slice(None),
                              em_rate=50e3, t_step=1e-3, roi=128,
                              pixel_size_m=None):
        """Expected (noise-free) camera image of the particles at focus.

        The particle photon emission in the time slice `t_slice` is
        computed from the light-sheet excitation profile at each
        particle's z position, binned on the camera pixel grid, and
        convolved with the detection PSF on the focal plane.

        Parameters
        ----------
        x, y, z : arrays (num_particles, n_samples) in meters
        t_slice : slice selecting the time samples of one camera frame
        em_rate : emission rate in photons/s of a particle at the
            sheet center (z = 0)
        t_step : simulation time step in seconds
        roi : number of camera pixels per axis
        pixel_size_m : camera pixel size in meters (sample space).
            If None, use the PSF grid `dxy` (micrometers).

        Returns
        -------
        img : 2D array (roi, roi) with expected photon counts per pixel
        """
        if pixel_size_m is None:
            pixel_size_m = self.dxy * 1e-6
        half = roi / 2 * pixel_size_m
        edges = np.linspace(-half, half, roi + 1)
        exc = self.excitation_weights(z[:, t_slice],
                                      x=x[:, t_slice], y=y[:, t_slice])
        weights = em_rate * t_step * exc
        # Bin the emission of each particle separately per z slice, so
        # that each particle is convolved with the detection kernel at
        # its own defocus (the camera PSF broadens away from focus).
        img = np.zeros((roi, roi), dtype=np.float64)
        # Unique z planes present in this time slice
        zslice = z[:, t_slice]
        iz = np.clip(np.round(zslice * 1e6 / self.dz).astype(int)
                     + self.nz_half, 0, 2 * self.nz_half)
        for z_idx in np.unique(iz):
            mask = iz == z_idx
            H, _, _ = np.histogram2d(y[:, t_slice][mask].ravel(),
                                     x[:, t_slice][mask].ravel(),
                                     bins=[edges, edges],
                                     weights=weights[mask].ravel())
            kernel = self.detection_kernel((z_idx - self.nz_half) * self.dz * 1e-6)
            img += fftconvolve(H, kernel, mode='same')
        return np.clip(img, 0.0, None)

    def render_camera_frames(self, positions, t_step=1e-3, frame_steps=50,
                             em_rate=50e3, roi=128, pixel_size_m=None,
                             background_rate=0.0, rs=None):
        """Simulate a stack of camera frames from particle trajectories.

        For each frame, the expected image is computed with
        `render_frame_expected` and photon (shot) noise is applied by
        Poisson sampling. An optional uniform background rate
        (photons/s/pixel) is added before sampling.

        Parameters
        ----------
        positions : array (num_particles, 3, n_samples) in meters,
            as stored by pybromo (`ParticlesSimulation.position`)
        t_step : simulation time step in seconds
        frame_steps : number of simulation steps integrated per camera
            frame (frame integration time = frame_steps * t_step)
        em_rate : emission rate in photons/s of a particle at the
            sheet center (z = 0)
        roi : number of camera pixels per axis
        pixel_size_m : camera pixel size in meters (sample space)
        background_rate : background count rate per pixel in photons/s
        rs : numpy RandomState for the Poisson sampling

        Returns
        -------
        frames : uint16 array (n_frames, roi, roi)
        """
        if rs is None:
            rs = np.random.RandomState()
        if pixel_size_m is None:
            pixel_size_m = self.dxy * 1e-6
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        n_samples = positions.shape[-1]
        n_frames = n_samples // frame_steps
        bg = background_rate * frame_steps * t_step
        frames = np.zeros((n_frames, roi, roi), dtype=np.uint16)
        for f in range(n_frames):
            t_slice = slice(f * frame_steps, (f + 1) * frame_steps)
            img = self.render_frame_expected(
                x, y, z, t_slice=t_slice, em_rate=em_rate,
                t_step=t_step, roi=roi, pixel_size_m=pixel_size_m)
            img += bg
            frames[f] = rs.poisson(img).astype(np.uint16)
        return frames
