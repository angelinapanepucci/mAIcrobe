"""
THE CODE IN THIS FILE IS DIRECTLY TAKEN FROM THE GITHUB REPOSITORY OF THE
ORIGINAL AUTHORS OF MICROMORPH, SIMONE COPPOLA AND SEAMUS HOLDEN.
https://github.com/HoldenLab/micromorph/blob/master/src/micromorph/bacteria/fluorescence_fitting.py
"""

import numpy as np
from lmfit import Model
from scipy.signal import convolve


def gaussian(params: tuple or list, x: np.array) -> np.array:
    """
    Gaussian function with 1 peak.

    Parameters
    ----------
    params: tuple or list
        parameters of the gaussian function [centre, sigma]
    x: np.array
        values at which the model is evaluated

    Returns
    -------
    values: np.array
        value(s) of the gaussian at x
    """

    a0 = params[0]
    a1 = params[1]

    # values = np.exp(-(x - a0) ** 2 / (a1 ** 2)) / (a1 * np.sqrt(2 * np.pi))
    values = np.exp(-((x - a0) ** 2) / (a1**2))
    return values


def ring_profile(
    x: np.array,
    psfFWHM: int or float,
    R: int or float,
    x0: int or float,
    offset: int or float,
    amp: int or float,
) -> np.array:
    """
    This function produces a line profile of a septum using an explicit 'tilted circle' model.
    It is based off code written in MATLAB by Séamus Holden, described in [Whitley et al. 2021](
    https://www.nature.com/articles/s41467-021-22526-0) and hosted in

    Parameters
    ----------
    x: np.array
        values at which the model is evaluated
    x0: float
        the centre of the ring
    R: float
        the radius of the ring
    psfFWHM: float
        the FWHM of the microscope PSF, in the unit of x_values
    offset: float
        the background value
    amp: float
        the amplitude of the ring profile

    Returns
    -------
    image_profile: np.array
        value(s) of the ring profile at x
    """
    # n = 10
    # x_original = np.copy(x) # new
    # x = np.linspace(np.min(x), np.max(x), len(x) * n) # new
    # Calculate the sigma (standard deviation) of the PSF
    sigma = psfFWHM / 2.35

    x_extended = np.insert(x, 0, 0)

    profile_integral = 2 * R * np.real(np.emath.arcsin((x_extended - x0) / R))

    image_profile = np.diff(profile_integral)

    # # add zero to the beginning of the array
    # image_profile = np.insert(image_profile, 0, 0)

    # Convolution with gaussian function
    gauss_profile = gaussian([0, sigma], profile_integral)
    # gauss_profile = np.insert(gauss_profile, 0, 0)
    image_profile = convolve(image_profile, gauss_profile, mode="same")

    # image_profile = image_profile / np.max(image_profile) # temp
    image_profile = image_profile * amp + offset

    # calculate center of mass of image_profile
    com_profile = np.average(x, weights=image_profile)

    index_x0 = np.argmin(np.abs(x - x0))
    index_com = np.argmin(np.abs(x - com_profile))

    # shift the profile to the center of mass
    image_profile = np.roll(image_profile, -index_x0 + index_com)

    # image_profile_original = np.copy(image_profile) # new
    # image_profile = image_profile[::n] # new
    # image_profile_interp = np.interp(x_original, x, image_profile)
    return image_profile


def get_initial_guess(x: np.array, y: np.array):
    """
    Get initial guess for ring profile model.

    Parameters
    ----------
    x: np.array
        values at which the model is evaluated
    y: np.array
        measured ring profile

    Returns
    -------
    initial_guess: dict
        initial guess for the parameters, in a dictionary
    """
    # Determine initial guess for the parameters
    amplitude_guess = np.max(y)
    bg_guess = np.min(y)

    # x0 is the middle position
    x0_guess = x[len(x) // 2]

    # find location minimum in first half of the data
    min_loc = x[np.argmax(y[: len(y) // 2])]
    # find location of maximum in second half of the data
    max_loc = x[(np.argmax(y[len(y) // 2 :]) + len(y) // 2)]

    # print(max_loc - min_loc)
    width_guess = (
        max_loc - min_loc
    ) / 2  # this could possibly be determined from the FWHM of the peak

    initial_guess = {
        "x0": x0_guess,
        "R": width_guess,
        "offset": bg_guess,
        "amp": amplitude_guess,
    }

    return initial_guess


def prepare_data(profile: np.array) -> np.array:
    """
    Prepare the data for fitting, normalising the profile.

    Parameters
    ----------
    profile: np.array
        the profile to be prepared

    Returns
    -------
    profile: np.array
        the normalised profile
    """
    # Normalise the profile
    profile = profile / np.max(profile)

    return profile


def fit_ring_profile(x: np.array, y: np.array, psfFWHM: float):
    """
    Fit the ring profile using a tilted circle model.

    Parameters
    ----------
    x: np.array
        values at which the model is evaluated
    y: np.array
        measured ring profile
    psfFWHM: float
        the FWHM of the microscope PSF, in the unit of x_values

    Returns
    -------
    result: lmfit.model.ModelResult
        the result of the fitting
    """
    # Prepare the data
    y = prepare_data(y)

    # upscale the data - helps with ring fitting
    n = 10
    x_upscaled = np.linspace(np.min(x), np.max(x), len(x) * n)
    y_upscaled = np.interp(x_upscaled, x, y)

    # Get initial guess for the parameters
    initial_guess = get_initial_guess(x_upscaled, y_upscaled)

    # Create the model
    gmodel = Model(ring_profile)

    # Set the parameters
    params = gmodel.make_params(
        x0=initial_guess["x0"],
        R=initial_guess["R"],
        offset=initial_guess["offset"],
        amp=initial_guess["amp"],
        psfFWHM=psfFWHM,
    )

    # Fix the psfFWHM
    params["psfFWHM"].vary = False

    # Fit the model
    result = gmodel.fit(y_upscaled, params, x=x_upscaled)
    return result
