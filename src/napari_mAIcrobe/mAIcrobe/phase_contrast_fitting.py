from lmfit import Model
import numpy as np

super_gaussian_order = 4

def super_gaussian_fwhm(width: float, order: float or int = super_gaussian_order) -> float:
    """
    Convert the raw firred super-Gaussian width parameter to full width at half maximum.

    Parameters
    ----------
    width: float
        Raw fitted width parameter from the super-Gaussian fit.
    order: float or int
        In this code, the order is fixed to 4.

    Returns
    --------
    float
        FWHM-corrected width in the same units as 'width'.

    """
    return 2 * width * (np.log(2)/2) ** (1 / order)

def super_gaussian(x: np.array, center: float, width: float, amplitude: float, order: float or int, offset: float or
    int) -> np.array:
    """
    Super Gaussian model (top hat) to be used for fitting phase contrast profiles.

    Parameters
    ----------
    x: np.array
        values at which the model is evaluated
    center: float
        center of the super gaussian
    width: float
        width of the super gaussian
    amplitude: float
        amplitude of the super gaussian
    order: int
        order of the super gaussian (makes it more or less top-hat shaped)
    offset: float
        offset of the super gaussian (background)

    Returns
    -------
    value: np.array
        value(s) of the super gaussian at x
    """

    value = np.exp(-2*((x - center) / width) ** (order)) * amplitude + offset
    if np.isnan(value).any():
        print(f'nan value found for x={x}, center={center}, width={width}, order={order}, amplitude={amplitude}')
    return value

def fit_top_hat_profile(x: np.array, y: np.array):
    """
    Fit the phase contrast profile using a super gaussian (top-hat) model.

    Parameters
    ----------
    x: np.array
        x (independent) values of the profile

    y: np.array
        y values of the profile - this is what gets fitted

    Returns
    -------
    result : lmfit object
        the result of the fitting process, see lmfit documentation for more details
    """
    # Invert the profile
    # y = invert_profile(y)

    # Get initial guess for the parameters
    initial_guess = get_initial_guess(x, y)

    # Create the model
    gmodel = Model(super_gaussian)

    # Set the parameters
    params = gmodel.make_params(center=initial_guess['center'],
                                width=initial_guess['width'],
                                amplitude=initial_guess['amplitude'],
                                order=super_gaussian_order,
                                offset=initial_guess['offset'])

    # fix order
    params['order'].vary = False

    # Fit the model
    result = gmodel.fit(y, params, x=x, nan_policy='raise')
    return result


def get_initial_guess(x: np.array, y: np.array) -> dict:
    """
    Get the initial guess for super gaussian model.

    Parameters
    ----------
    x: np.array
        values at which the model is evaluated
    y: np.array
        measured phase contrast profile

    Returns
    -------
    initial_guess: dict
        initial guess for the parameters, in a dictionary
    """
    # Determine initial guess for the parameters
    amplitude_guess = np.max(y)
    offset_guess = np.min(y)

    # find location minimum in first half of the data
    min_loc = x[np.argmin(y[:len(y) // 2])]
    # find location of minimum in second half of the data
    max_loc = x[(np.argmin(y[len(y) // 2:]) + len(y) // 2)]

    center_guess = (max_loc + min_loc) / 2

    width_guess = (max_loc - min_loc) / 2

    # Store all the initial guesses in a dictionary
    initial_guess = {'center': center_guess, 'width': width_guess, 'amplitude': amplitude_guess, 'offset': offset_guess}

    return initial_guess

def invert_profile(profile: np.array) -> np.array:
    """
    Convenience function to invert the profile to make it more suitable for fitting.

    Parameters
    ----------
    profile: np.array
        the profile to be inverted

    Returns
    -------
    profile: np.array
        the inverted profile
    """
    # Invert the profile
    profile = np.max(profile) - profile

    return profile

def fit_phase_contrast_profile(x: np.array, y: np.array):
    """
    Fit the phase contrast profile using a super gaussian (top-hat) model.

    Parameters
    ----------
    x: np.array
        x (independent) values of the profile

    y: np.array
        y values of the profile - this is what gets fitted

    Returns
    -------
    result : lmfit object
        the result of the fitting process, see lmfit documentation for more details
    """
    # Invert the profile
    y = invert_profile(y)

    # Get initial guess for the parameters
    initial_guess = get_initial_guess(x, y)

    # Create the model
    gmodel = Model(super_gaussian)

    # Set the parameters
    params = gmodel.make_params(center=initial_guess['center'],
                                width=initial_guess['width'],
                                amplitude=initial_guess['amplitude'],
                                order=super_gaussian_order,
                                offset=initial_guess['offset'])

    # fix order
    params['order'].vary = False
    params['width'].min = 0


    # Fit the model
    result = gmodel.fit(y, params, x=x, nan_policy='raise')

    return result