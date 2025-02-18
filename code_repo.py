# Code snippets archive to (reuse) in Jupyter notebooks
# If the function is adapted, add the date of the most
# recent edit.
# I know, it's kind of messy, but we stick to it for now...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import fretbursts as frb
import phconvert as phc
import H2MM_C as h2
import burstH2MM as bhm


def dwell_ES_histprojection(model, ax=None, states=None, figsize=None, plot_ranges=None, num_bins=None,
                            show_transarrows=True, state_kwargs=None, **kwargs):
    """
    Dwell E-S plot with histogram projections on the side.

    Parameters
    ----------
    model : H2MM_result
        Source of data.
    ax : matplotlib.axes.Axes, optional
        Axes to draw the main plot. If None, a new figure and axes are created.
    states : numpy.ndarray, optional
        Which states to plot. If None, all states are plotted.
    figsize : tuple, optional
        Size of the figure. Default is (6, 6).
    plot_ranges : list of lists, optional
        Ranges for E and S axes. Default is [[-0.1, 1.1], [-0.1, 1.1]].
    num_bins : int, optional
        Number of bins for histograms. Default is 25.
    show_transarrows : bool, optional
        Whether to show transition arrows. Default is True.
    state_kwargs : list of dicts, optional
        Keyword arguments for plotting each state.
    **kwargs : dict
        Additional keyword arguments for scatter and histogram styling.

    Returns
    -------
    None
        The function modifies an active Matplotlib figure, if provided. Users can call plt.show() or plt.savefig().
    """

    if states is None:
        states = np.unique(model.dwell_state)
    if figsize is None:
        figsize = (6, 6)
    if plot_ranges is None:
        plot_ranges = [[-0.1, 1.1], [-0.1, 1.1]]
    if num_bins is None:
        num_bins = 25
    if state_kwargs is None:
        state_kwargs = [{} for _ in states]

    # Create new figure if ax is not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 4, hspace=0.15, wspace=0.15)
        ax_main = fig.add_subplot(gs[1:4, 0:3])  # Main scatter plot
        ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)  # X-axis histogram
        ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)  # Y-axis histogram
    else:
        ax_main = ax
        fig = ax_main.figure
        gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=ax_main.get_subplotspec(), hspace=0.15, wspace=0.15)
        ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    # Extract E and S dwells
    dwells_idx, dwells_E, dwells_S = [], [], []
    for dwell in np.unique(model.dwell_state):
        indices = np.where(model.dwell_state == dwell)[0]
        dwells_idx.append(indices)
        dwells_E.append(model.dwell_E[indices])
        dwells_S.append(model.dwell_S[indices])

    # Main scatter plot
    for idx, state in enumerate(states):
        ax_main.scatter(dwells_E[state], dwells_S[state], s=1, **state_kwargs[idx], **kwargs)
    if show_transarrows:
        bhm.trans_arrow_ES(model, ax=ax_main)

    # Histogram projections
    for idx, state in enumerate(states):
        ax_xhist.hist(dwells_E[state], bins=num_bins, range=plot_ranges[0], alpha=0.9, histtype='step', lw=2, **kwargs)
        ax_yhist.hist(dwells_S[state], bins=num_bins, range=plot_ranges[1], orientation='horizontal', alpha=0.9, histtype='step', lw=2, **kwargs)

    # Style plot
    plt.setp(ax_xhist.get_xticklabels(), visible=False)
    plt.setp(ax_yhist.get_yticklabels(), visible=False)
    ax_main.set_xlim(*plot_ranges[0])
    ax_main.set_ylim(*plot_ranges[1])
    ax_xhist.set_xlim(*plot_ranges[0])
    ax_xhist.grid(visible=True, axis='both', linestyle='-', alpha=0.75)
    ax_xhist.set_ylabel("# Dwells")
    ax_yhist.set_ylim(*plot_ranges[1])
    ax_yhist.grid(visible=True, axis='both', linestyle='-', alpha=0.75)
    ax_yhist.set_xlabel("# Dwells")
    ax_main.set_xlabel(r'$E$')
    ax_main.set_ylabel(r'$S$')

return fit, ax_main, ax_xhist, ax_yhist


# Defining helper functions to process raw data
def shift_nanotimes(nanotimes_array: np.array, detectors_array: np.array, detector_id: int, tcspc_bin_shift) -> np.array:
    '''
    Shift the nanotimes in the nanotimes array by a certain number of bins (- -> to the left, + -> to the right).
    This is sometimes necessary to make the D-D and D-A photons overlap relative to each other on two spectrally-
    split detector streams.
    The shift is applied to an unsigned integer array and may wrap some entries around and out of the range
    of the 12 bit nanotimes bin assignment. Thus, entries that become larger than the maximum bin number
    are set to 0.
    '''
    nanotimes_shifted = nanotimes_array
    nanotimes_shifted[np.where(detectors_array == detector_id)] -= tcspc_bin_shift
    nanotimes_shifted[np.where(nanotimes_shifted > 4095)] = 0

    return nanotimes_shifted

def combine_spc_files(file_list, set_file):
    # Use phconvert to stitch several spc files together
    # Allows to treat it as one combined experiment

    assert len(file_list) > 1, "Only one file was supplied"

    spc_file = phc.bhreader.load_spc(file_list[0])
    set_file = phc.bhreader.load_set(set_file)


    macrotime = spc_file[0]
    routing = spc_file[1]
    nanotimes = spc_file[2]


    for i in range(1, len(file_list)):
        spc_file = phc.bhreader.load_spc(file_list[i])
        macrotime = np.append(macrotime, spc_file[0] + macrotime[-1])
        routing = np.append(routing, spc_file[1])
        nanotimes = np.append(nanotimes, spc_file[2])

    spc_file = (macrotime, routing, nanotimes, spc_file[3])


    return spc_file, set_file


def spc_to_hdf_hasseltsetup(
        spc_file: list,
        set_file: str,
        description: str,
        author: str,
        sample_name: str,
        buffer_name: str,
        dye_names: str,
        DD_par_shift: int,
        DD_perp_shift: int,
        AA_par_shift: int,
        AA_perp_shift: int,
        output_path: str
    ):
    '''
    Converts input .spc data to photon-HDF5 format and allows data annotation in this format.
    This function is specific for the smFRET microscope at UHasselt/DBI group, but can be
    adapted for different setups in the same way.
    The user may supply metadata to be saved in the photon-HDF5 file and may manipulate
    the data to allow easier processing in FRETbursts, e.g. shifting the nanotimes array.
    After convertion, the photon-HDF5 file is saved to the disk at the specified location.
    '''


    # Loading files OR stitch multiple spc files
    ## Check if list of files was supplied
    if len(spc_file) > 1:
        # Stitch single photon recorndna
        spc_data, set_data = combine_spc_files(spc_file, set_file)
    else:
        spc_data = phc.bhreader.load_spc(spc_file[0])
    # Check if set file is valid
    try:
        set_data = phc.bhreader.load_set(set_file)
    except:
        print('Invalid .set file... Using parameters from other file...')


    # Experimental description field
    identity = dict(
        author = author,
        author_affiliation = 'Dynamic Bioimaging / Hasselt University'
    )

    sample = dict(
        sample_name = sample_name,
        buffer_name = buffer_name,
        dye_names = dye_names
    )



    provenance = dict(
        filename = spc_file[0],
        creation_time = set_data['identification']['Date']
    )


    # Setup field
    measurement_specs = dict(
        measurement_type = 'generic', # Specific to UHasselt/DBI setup!
        detectors_specs = {
            'spectral_ch1': [0, 1],
            'spectral_ch2': [2, 4]
            #'polarization_ch1': [6, 2],
            #'polarization_ch2': [7, 4]
        },
        laser_repetition_rate = 80e-9/3,
        alex_excitation_period1 = (1100, 2399),
        alex_excitation_period2 = (2400, 3900)
    )

    setup = dict(
        num_pixels = 2,
        num_spots = 1,
        num_spectral_ch = 2,
        num_polarization_ch = 1,
        num_split_ch = 1,
        modulated_excitation = True,
        excitation_alternated = [True, True],
        excitation_cw = [False, False],
        lifetime = True,
        laser_repetition_rates = [80e-9/3],
        excitation_wavelengths = [483e-9, 633e-9],
        detection_wavelengths = [525e-9, 705e-9]
    )


    # Photon data
    ## Shift nanotimes according to user
    nanotimes_shifted = spc_data[2]

    if DD_par_shift:
        print('Shifting DD par nanotimes...')
        nanotimes_shifted = shift_nanotimes(nanotimes_shifted, spc_data[1], 0, DD_par_shift) # 6
    if DD_perp_shift:
        print('Shifting DD perp nanotimes...')
        nanotimes_shifted = shift_nanotimes(nanotimes_shifted, spc_data[1], 1, DD_perp_shift) # 7
    if AA_par_shift:
        print('Shifting AA par nanotimes...')
        nanotimes_shifted = shift_nanotimes(nanotimes_shifted, spc_data[1], 2, AA_par_shift) # 2
    if AA_perp_shift:
        print('Shifting AA perp nanotimes...')
        nanotimes_shifted = shift_nanotimes(nanotimes_shifted, spc_data[1], 4, AA_perp_shift) # 4

    plt.hist(nanotimes_shifted[np.where(spc_data[1] == 0)], bins = 250, label = 'BBpar (0)', histtype='step')
    plt.hist(nanotimes_shifted[np.where(spc_data[1] == 1)], bins = 250, label = 'BBperp (1)', histtype='step')
    plt.hist(nanotimes_shifted[np.where(spc_data[1] == 2)], bins = 250, label = 'RRpar (2)', histtype='step')
    plt.hist(nanotimes_shifted[np.where(spc_data[1] == 4)], bins = 250, label = 'RRperp (4)', histtype='step')
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.show()


    ## organize photon data into dict
    photon_data = dict(
        timestamps = spc_data[0],
        timestamps_specs = {
            'timestamps_unit': set_data['sys_params']['SP_TAC_R']
        },
        detectors = spc_data[1],
        nanotimes = nanotimes_shifted,
        nanotimes_specs = {
            'tcspc_unit': set_data['sys_params']['SP_TAC_TC'],
            'tcspc_range': set_data['sys_params']['SP_ADC_RE'] * set_data['sys_params']['SP_TAC_TC'],
            'tcspc_num_bins': set_data['sys_params']['SP_ADC_RE']
        },
        measurement_specs = measurement_specs
    )


    # Combine into single dict
    # and save to HDF5
    h5_data = dict(
        identity = identity,
        sample = sample,
        provenance = provenance,
        description = description,
        photon_data = photon_data,
        setup = setup
    )


    # Write data to disk
    if output_path:
        phc.hdf5.save_photon_hdf5(
            h5_data,
            output_path,
            overwrite = True,
            close = True
        )
    else:
        raise FileNotFoundError


# BVA Functions
# BVA functions from https://bursth2mm.readthedocs.io/en/latest/burstH2MM_HP3_TE300_PIE_paper.html#Environment-setup

def bin_bva(E, std, R, B_thr):
    E, std = np.concatenate(E), np.concatenate(std)
    bn = np.linspace(0,1, R+1)
    std_avg, E_avg = np.empty(R), np.empty(R)
    for i, (bb, be) in enumerate(zip(bn[:-1], bn[1:])):
        mask = (bb <= E) * (E < be)
        if mask.sum() > B_thr:
            std_avg[i], E_avg[i] = np.mean(std[mask]), np.mean(E[mask])
        else:
            std_avg[i], E_avg[i] = -1, -1
    return E_avg, std_avg

def BVA(data, chunck_size):
    """
    Perform BVA analysis on a given data set.
    Calculates the std dev E in each burst.
    ----------
    d: FRETBursts data object
        A FRETBursts data object, must have burst selection already performed.
    chunk_size: int
        Size of the sub-bursts to assess E
    Returns
    -------
    E_eff: list[np.ndarray[float]]
        Raw FRET efficiency of each burst
    std_E: list[np.ndarray[float]]
        Standard deviation of FRET values of each burst

    """
    E_eff, std_E = list(), list()
    for ich, mburst in enumerate(data.mburst): # iterate over channels
        # create lists to store values for burst in channel
        stdE, E = list(), list()
        # get masking arrays before iterating over bursts
        Aem = data.get_ph_mask(ich=ich, ph_sel=frb.Ph_sel(Dex='Aem')) # get acceptor instances to calculate E
        Dex = data.get_ph_mask(ich=ich, ph_sel=frb.Ph_sel(Dex='DAem')) # get Dex mask to remove Aex photons
        for istart, istop in zip(mburst.istart, mburst.istop): # iterate over each burst
            phots = Aem[istart:istop+1][Dex[istart:istop+1]] # Dex photons in burst True if Aem and False if Dem
            # list of number of Aem in each chunch, easier as list comprehansion
            Esub = [phots[nb:ne].sum() for nb, ne in zip(range(0,phots.size, chunck_size),
                                                         range(chunck_size,phots.size+1, chunck_size))]
            # Cacluate burst-wise value and append to list of bursts in channel
            stdE.append(np.std(Esub)/chunck_size) # calculate and append standard deviation
            E.append(sum(Esub)/(len(Esub)*chunck_size)) # calculate E

        # convert single-channel list to array and add to channel list
        E_eff.append(np.array(E))
        std_E.append(np.array(stdE))
    return E_eff, std_E




def count_dwells(model: bhm.H2MM_result, selection_idxs: list) -> int:
    """
    Counts the number of dwells in a set of bursts from an H2MM model.

    Parameters
    ----------
    model: H2MM_result model, e.g. mp.models[3]
    selection_idxs: list of states to be tested in binary format
        e.g. [0b0001, 0b0011] for dwells in state 0 and state 0 + 1

    Returns
    -------
    n_dwells: int
    """
    n_dwells = 0
    for selection_idx in selection_idxs:
        n_dwells += np.where(model.burst_type == int(selection_idx))[0].shape[0]
    
    return n_dwells



def get_purification_mask(model: bhm.H2MM_result, select_states: list) -> np.array:
    """
    Returns a selection mask of bursts which contain the dwells of the 
    H2MM_results model provided. 

    Parameters
    ----------
    model: bhm.H2MM_result
        H2MM model containing burst dwell classification as 'burst_type'
        attribute
    selected_states: list of binary
        States/dwells to select

    Returns
    -------
    Indices of selected bursts based on the H2MM_results classification.
    
    Selected states should be supplied as binary. For example, to select 
    bursts that dwelled in state 0, state 1, and states 0 and 1:
        [0b0001, 0b0010, 0b0011]

    Example
    -------
    purified_bursts = get_purification_mask(mp.models[model_id], select_states=[0b0001, 0b0010, 0b0011])
    ds_purified = ds.select_bursts_mask_apply(purified_bursts)
    """
    return np.where(np.isin(model.burst_type, select_states))


def get_burst_state_dwell_time(model: bhm.BurstSort.H2MM_result, state: int) -> np.array:
    """
    Returns the time in ms that a state was found in the given
    list of bursts.
    The H2MM_result knows both the list and trajectory of DWELLS
    and the dwells in a burst. Since we want to represent the
    ES plot (in the traditional sense), the persistence time
    of the asked for state needs to be calculated on a burst
    and not a dwell basis!

    Arguments
    ---
    model: bhm.BurstSort.H2MM_result 
        Result from an burstH2MM run
    state_color: int
        State to be represented by the color
    c_range: list[int]
        Min and max range for the color map.

    Returns
    ---
    burst_state_dur: np.array
        Duration a state was found in a burst, in milliseconds.
    """
    n_bursts = model.burst_type.shape[0]
    n_dwells = model.dwell_dur.shape[0]
    dwells_per_burst = model.burst_dwell_num # No. of identified dwells in each burst

    burst_state_dur = np.zeros(shape = (n_bursts), dtype=np.float32)

    dwell_index = 0

    for burst in range(n_bursts):
        for dwell in range(dwells_per_burst[burst]):
            if np.isin(model.dwell_state[dwell_index], state):#model.dwell_state[dwell_index] == state:
                burst_state_dur[burst] += model.dwell_dur[dwell_index]
            dwell_index += 1

    print(dwell_index, n_dwells)

    return burst_state_dur
