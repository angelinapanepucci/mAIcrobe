"""
THE CODE IN THIS FILE IS DIRECTLY TAKEN FROM THE GITHUB REPOSITORY OF THE
ORIGINAL AUTHORS OF MICROMORPH, SIMONE COPPOLA AND SEAMUS HOLDEN. I HAVE MADE MINOR MODIFICATIONS
TO THE CODE TO MAKE IT WORK WITH THE REST OF THE CODE IN THIS REPOSITORY, BUT THE ORIGINAL CODE CAN BE FOUND HERE:
https://github.com/HoldenLab/micromorph/blob/master/src/micromorph/bacteria/shape_analysis.py
"""

import logging

import numpy as np
from scipy.interpolate import BSpline, splrep
from shapely.geometry import Point, Polygon
from skimage.measure import profile_line
from skimage.morphology import skeletonize

from .fluorescence_fitting import fit_ring_profile
from .phase_contrast_fitting import (
    fit_phase_contrast_profile,
    super_gaussian_fwhm,
)
from .utilities import (
    get_boundary_coords,
    get_width_profile_lines,
    prune_short_branches,
    trace_axis,
)

"""
Collection of functions to analyse the shape of bacteria. These are all wrappers that run multiple other functions,
but generate the output required for the Bacteria class.
"""


def get_bacteria_length(
    medial_axis_extended: np.array, pxsize: float or int = 1.0
) -> float:
    """
    Calculate the length of a segmented bacterium.

    Parameters
    ----------
    medial_axis_extended : np.array
        The extended medial axis of the bacterium, whose arclength is calculated.
    pxsize : float, optional
        The pixel size of the image. The final distance will be multiplied by this value (default is 1.0).

    Returns
    -------
    total_distance : float
        Total distance covered by the extended medial axis (in px).
    """

    # Get distance vectors
    distance_vectors = np.diff(medial_axis_extended, axis=0)
    distance_vectors = np.concatenate((np.array([[0, 0]]), distance_vectors))

    # From the vectors, calculate the sums
    distances_individual = np.sqrt(np.sum(distance_vectors**2, axis=1))

    # now get total distance
    total_distance = np.sum(distances_individual) * pxsize

    return total_distance


def get_bacteria_widths(
    img: np.array,
    med_ax: np.array,
    n_lines: int = 5,
    pxsize: float or int = 1.0,
    psfFWHM: float or int = 0.250,
    fit_type: str or None = None,
    line_magnitude: float or int = 20,
) -> np.array:
    """
    Calculate the width of a segmented bacterium.

    Parameters
    ----------
    img : np.array
        The original image.
    med_ax : np.array
        The medial axis which will be used to find profile lines to fit.
    n_lines : int, optional
        The number of lines to fit (default is 5).
    pxsize : float or int, optional
        The pixel size of the image (default is 1.0).
    psfFWHM : float or int, optional
        The FWHM of the PSF (default is 5).
    fit_type : str or None, optional
        The type of fit to use (default is None).
    line_magnitude : float or int, optional
        The length of the line used to measure the profile (default is 20).

    Returns
    -------
    all_widths: np.array
        Array of widths.
    """
    xh, yh, xl, yl = get_width_profile_lines(
        med_ax, n_points=n_lines, line_magnitude=line_magnitude
    )

    profile_points = np.column_stack((xh, yh, xl, yl))

    all_widths = np.array([])

    for points in profile_points:
        # img or bound_transform
        current_profile = profile_line(
            img, (points[1], points[0]), (points[3], points[2])
        )

        x = np.arange(0, len(current_profile)) * pxsize

        try:
            if fit_type == "phase":
                result = fit_phase_contrast_profile(x, current_profile)
                width = super_gaussian_fwhm(result.params["width"].value)

            elif fit_type == "fluorescence":
                result = fit_ring_profile(x, current_profile, psfFWHM)
                width = result.params["R"].value * 2
            else:
                raise ValueError('Invalid fit type".')

            all_widths = np.append(all_widths, width)
        except:
            # logging.info("Error fitting profile, skipping this one.")
            pass

    return all_widths


def get_bacteria_boundary(
    mask: np.array, boundary_smoothing_factor: int or None = 7
) -> np.array:
    """
    Get the (smoothed) boundary of a segmented bacterium.

    Parameters
    ----------
    mask : np.array
        The binary image (with a single bacterium only).
    boundary_smoothing_factor : int or None, optional
        The number of frequencies to use for FFT smoothing (default is 7).

    Returns
    -------
    boundary: np.array
        XY coordinates of the boundary (smoothed).
    """

    # Get initial coordinates for boundary coords
    boundary_xy = get_boundary_coords(mask)
    # boundary_xy = fix_coordinates_order(boundary_xy) # Performance of this may be questionable.

    if boundary_smoothing_factor is None:
        return boundary_xy
    else:
        x = boundary_xy[:, 0]
        y = boundary_xy[:, 1]

        x_fft = np.fft.rfft(x)
        x_fft[boundary_smoothing_factor:] = 0

        y_fft = np.fft.rfft(y)
        y_fft[boundary_smoothing_factor:] = 0

        x_smoothed = np.fft.irfft(x_fft)
        y_smoothed = np.fft.irfft(y_fft)

        boundary_xy_smoothed = np.column_stack((x_smoothed, y_smoothed))
        return np.fliplr(boundary_xy_smoothed)


def get_medial_axis(mask: np.array) -> np.array:
    """
    Get the medial axis of a segmented bacterium.

    Parameters
    ----------
    mask : np.array
        The binary image (with a single bacterium only).

    Returns
    -------
    medial_axis : np.array
        The medial axis of the bacterium.
    """
    bacteria_skeleton = skeletonize(mask)

    # Get the medial axis
    bacteria_skeleton = prune_short_branches(bacteria_skeleton)

    # Get the medial axis
    medial_axis = trace_axis(bacteria_skeleton)

    return medial_axis


# def smooth_medial_axis(medial_axis: np.array, boundary: np.array, error_threshold: float = 0.05, max_iter: int = 50, spline_val: int = 1, n_spline_points = 500) -> np.array:
#     """
#     Smooth the medial axis of a segmented bacterium, using the boundary as a reference.

#     Parameters
#     ----------
#     medial_axis : np.array
#         The medial axis of the bacterium.
#     boundary : np.array
#         The boundary of the bacterium.
#     error_threshold : float, optional
#         The threshold for the error in the smoothing (default is 0.05).
#     max_iter : int, optional
#         The maximum number of iterations for the smoothing (default is 50).

#     Returns
#     -------
#     smoothed_medial_axis: np.array
#         The smoothed medial axis.
#     """
#     print("NEW VERSION!! - 2")
#     # Sanity check that all points in the medial axis are INSIDE the boundary.
#     # This is due to the fact that we use the smoothed boundary for this calculation, but the medial axis is
#     # calculated from the mask, which can lead to points extending outside the boundary


#     medial_axis = np.array([uniform_filter1d(medial_axis[:,0], size=5), uniform_filter1d(medial_axis[:,1], size=5)]).T

#     iter_n = 0
#     previous_error = np.inf
#     while iter_n < max_iter:
#         # logging.info(f"Starting iteration {iter_n}")
#         middle_points = []
#         all_points = []
#         perpendicular_points = []

#         for i in range(medial_axis.shape[0] - 1):
#             # calculate the angle between the first and second point of the medial axis
#             angle = np.arctan2(medial_axis[i + 1, 1] - medial_axis[i, 1], medial_axis[i + 1, 0] - medial_axis[i, 0])

#             # now find the angle perpendicular to this angle
#             angle_perpendicular = angle + np.pi / 2

#             # calculate the middle point between the first and second point of the medial axis
#             middle_point = (medial_axis[i + 1] + medial_axis[i]) / 2


#             # get the closest point on the boundary to the perpendicular line
#             distances = np.abs((boundary[:, 1] - middle_point[1]) * np.cos(angle_perpendicular) - (
#                         boundary[:, 0] - middle_point[0]) * np.sin(angle_perpendicular))


#             # get the index of the closest point
#             # FIXME: this is ugly, but it works
#             idx = np.argmin(distances)

#             closest_point = boundary[idx]

#             # get the second closest
#             distances[idx] = np.inf
#             idx = np.argmin(distances)
#             closest_point2 = boundary[idx]

#             perpendicular_points.append(closest_point)
#             perpendicular_points.append(closest_point2)

#             # get the middle point between the two closest points
#             middle_point = (closest_point + closest_point2) / 2
#             middle_points.append(middle_point)

#             all_points.append(medial_axis[i])
#             all_points.append(middle_point)
#             all_points.append(medial_axis[i + 1])

#         # convert to numpy array
#         all_points = np.array(all_points)

#         # smooth the points
#         x = all_points[:, 0]
#         y = all_points[:, 1]

#         all_points = np.array([uniform_filter1d(x, size=5), uniform_filter1d(y, size=5)]).T

#         # all_points = np.array([x_smooth_s(calculation_values), y_smooth_s(calculation_values)]).T

#         # check if there are any large deviations in the smoothed points
#         # if there are, we need to stop the smoothing
#         delta_x = np.abs(np.diff(all_points[:, 0]))
#         if np.max(delta_x) > 100:
#             # logging.info(f"Large deviation in x, stopping smoothing here. Delta is {np.max(delta_x)}")
#             break

#         if iter_n >= 0:
#             current_error = np.abs(np.mean(all_points - medial_axis))
#             logging.info(f"Current error: {current_error}, iteration: {iter_n}/{max_iter}")
#             print(f"Current error: {current_error}, iteration: {iter_n}/{max_iter}")
#             if current_error > previous_error:
#                 pass
#                 logging.info(f"Error increased, stopping smoothing here. iteration: {iter_n}")
#                 print(f"Error increased, stopping smoothing here. iteration: {iter_n}")
#                 # break
#             elif current_error < error_threshold:
#                 medial_axis = all_points
#                 logging.info("Error below threshold, stopping smoothing here.")
#                 print("Error below threshold, stopping smoothing here.")
#                 break
#             else:
#                 medial_axis = all_points
#                 previous_error = current_error

#         iter_n += 1
#     return medial_axis


def smooth_medial_axis(
    medial_axis: np.array,
    boundary: np.array,
    error_threshold: float = 0.05,
    max_iter: int = 50,
    spline_val: int = 1,
    spline_spacing=0.25,
) -> np.array:
    """
    Smooth the medial axis of a segmented bacterium, using the boundary as a reference.

    Parameters
    ----------
    medial_axis : np.array
        The medial axis of the bacterium.
    boundary : np.array
        The boundary of the bacterium.
    error_threshold : float, optional
        The threshold for the error in the smoothing (default is 0.05).
    max_iter : int, optional
        The maximum number of iterations for the smoothing (default is 50).

    Returns
    -------
    smoothed_medial_axis: np.array
        The smoothed medial axis.
    """
    # Sanity check that all points in the medial axis are INSIDE the boundary.
    # This is due to the fact that we use the smoothed boundary for this calculation, but the medial axis is
    # calculated from the mask, which can lead to points extending outside the boundary

    boundary_polygon = Polygon(boundary)
    medial_axis = np.array(
        [
            point
            for point in medial_axis
            if Point(point).within(boundary_polygon)
        ]
    )

    # smooth medial axis before starting
    # smooth the points
    x = medial_axis[:, 0]
    y = medial_axis[:, 1]

    # need to calculate arclength of the medial axis, so we can use it for smoothing
    distances = np.sqrt(np.sum(np.diff(medial_axis, axis=0) ** 2, axis=1))
    distances = np.concatenate(
        (np.array([0]), distances)
    )  # add 0 for the first point
    total_arclength_distance = np.cumsum(
        distances
    )  # cumulative sum to get the arclength

    # n_spline_points is such that each point should be spaced px_spacing apart
    n_spline_points = int(
        np.ceil(total_arclength_distance[-1] / spline_spacing)
    )

    # we are fitting x and y independently of each other
    try:
        indexes = np.arange(0, len(x), 1)

        tck_s = splrep(indexes, x, s=len(x), k=spline_val)
        x_smooth_s = BSpline(*tck_s)

        tck_sy = splrep(indexes, y, s=len(y), k=spline_val)
        y_smooth_s = BSpline(*tck_sy)
    except:
        return medial_axis

    calculation_values = np.linspace(0, len(x) - 1, n_spline_points)

    medial_axis = np.array(
        [x_smooth_s(calculation_values), y_smooth_s(calculation_values)]
    ).T

    iter_n = 0
    previous_error = np.inf
    while iter_n < max_iter:
        # logging.info(f"Starting iteration {iter_n}")
        middle_points = []
        all_points = []
        perpendicular_points = []

        for i in range(medial_axis.shape[0] - 1):
            # calculate the angle between the first and second point of the medial axis
            angle = np.arctan2(
                medial_axis[i + 1, 1] - medial_axis[i, 1],
                medial_axis[i + 1, 0] - medial_axis[i, 0],
            )

            # now find the angle perpendicular to this angle
            angle_perpendicular = angle + np.pi / 2

            # calculate the middle point between the first and second point of the medial axis
            middle_point = (medial_axis[i + 1] + medial_axis[i]) / 2

            # get the closest point on the boundary to the perpendicular line
            distances = np.abs(
                (boundary[:, 1] - middle_point[1])
                * np.cos(angle_perpendicular)
                - (boundary[:, 0] - middle_point[0])
                * np.sin(angle_perpendicular)
            )

            # get the index of the closest point
            # FIXME: this is ugly, but it works
            idx = np.argmin(distances)

            closest_point = boundary[idx]

            # get the second closest
            distances[idx] = np.inf
            idx = np.argmin(distances)
            closest_point2 = boundary[idx]

            perpendicular_points.append(closest_point)
            perpendicular_points.append(closest_point2)

            # get the middle point between the two closest points
            middle_point = (closest_point + closest_point2) / 2
            middle_points.append(middle_point)

            all_points.append(medial_axis[i])
            all_points.append(middle_point)
            all_points.append(medial_axis[i + 1])

        # convert to numpy array
        all_points = np.array(all_points)
        all_points = np.array(
            [
                point
                for point in all_points
                if Point(point).within(boundary_polygon)
            ]
        )

        # smooth the points
        x = all_points[:, 0]
        y = all_points[:, 1]

        # we are fitting x and y independently of each other
        indexes = np.arange(0, len(x), 1)

        tck_s = splrep(indexes, x, s=len(x), k=spline_val)
        x_smooth_s = BSpline(*tck_s)

        tck_sy = splrep(indexes, y, s=len(y), k=spline_val)
        y_smooth_s = BSpline(*tck_sy)

        calculation_values = np.linspace(0, len(x) - 1, n_spline_points)

        all_points = np.array(
            [x_smooth_s(calculation_values), y_smooth_s(calculation_values)]
        ).T

        # check if there are any large deviations in the smoothed points
        # if there are, we need to stop the smoothing
        delta_x = np.abs(np.diff(all_points[:, 0]))
        if np.max(delta_x) > 100:
            # logging.info(f"Large deviation in x, stopping smoothing here. Delta is {np.max(delta_x)}")
            break

        if iter_n >= 0:
            current_error = np.abs(np.mean(all_points - medial_axis))
            # logging.info(f"Current error: {current_error}, iteration: {iter_n}/{max_iter}")
            if current_error > previous_error:
                pass
                # logging.info("Error increased, stopping smoothing here.")
                # break
            elif current_error < error_threshold:
                medial_axis = all_points
                # logging.info("Error below threshold, stopping smoothing here.")
                break
            else:
                medial_axis = all_points
                previous_error = current_error

        iter_n += 1
    return medial_axis


def extend_medial_axis_roughly(
    medax, bnd, max_iter=500, min_distance_to_boundary=1, step_size=1
):
    """
    Extend the medial axis to the boundary, without any smoothing.

    Parameters
    ----------
    medax : np.array
        The medial axis.
    bnd : np.array
        The boundary.
    max_iter : int, optional
        The maximum number of iterations (default is 50).
    min_distance_to_boundary : int, optional
        The minimum distance to the boundary (default is 1).
    step_size : int, optional
        The step size to take (default is 1).

    Returns
    -------
    extended_medial_axis : np.array
        The extended medial axis.
    """
    medax_ext = np.copy(medax)

    # Generate a polygon to use for checking if the points are inside the boundary
    boundary_polygon = Polygon(bnd)
    # medial_axis = np.array([point for point in medial_axis if Point(point).within(boundary_polygon)])

    # We need to check that there's at least two points in the medial axis
    if medax.shape[0] < 2:
        return medax_ext

    # Direction second to last to last point
    direction = np.arctan2(
        medax[-1, 1] - medax[-2, 1], medax[-1, 0] - medax[-2, 0]
    )

    dist_to_bound = np.inf
    N = 0
    while N < max_iter:  # dist_to_bound > min_distance_to_boundary and
        step_vector = step_size * np.array(
            [np.cos(direction), np.sin(direction)]
        )
        next_point = medax_ext[-1, :] + step_vector
        distances = np.sqrt(
            (next_point[0] - bnd[:, 0]) ** 2 + (next_point[1] - bnd[:, 1]) ** 2
        )

        # check if the next point is outside the boundary
        if Point(next_point[0], next_point[1]).within(boundary_polygon):
            # logging.info("Next point is inside. Continuing")
            medax_ext = np.vstack((medax_ext, next_point))
        else:
            # logging.info("Next point is outside. Stopping")
            medax_ext = np.vstack((medax_ext, next_point))
            break

        N += 1

    # FIXME: UGLY - probably needs a second function which is called twice, not this ...
    # Repeat in the opposite direction

    # Direction second to first point
    direction = np.arctan2(
        medax[1, 1] - medax[0, 1], medax[1, 0] - medax[0, 0]
    )

    dist_to_bound = np.inf
    N = 0
    while N < max_iter:  # dist_to_bound > min_distance_to_boundary and
        step_vector = (
            -step_size * np.array([np.cos(direction), np.sin(direction)]) * 2
        )
        next_point = medax_ext[0, :] + step_vector
        #     dist_to_bound = np.min(np.sqrt((next_point[0] - bnd[:, 0]) ** 2 + (next_point[1] - bnd[:, 1]) ** 2))
        #     # check if the next point is outside the boundaryp
        #     logging.info(f"Next point: {next_point}, distance to boundary: {dist_to_bound}")
        # check if the next point is outside the boundary#
        if Point(next_point[0], next_point[1]).within(boundary_polygon):
            medax_ext = np.vstack((next_point, medax_ext))
        else:
            logging.debug("Next point is outside. Stopping")
            break

        N += 1

    # remove first and last point
    medax_ext = np.delete(medax_ext, 0, axis=0)
    medax_ext = np.delete(medax_ext, -1, axis=0)

    return medax_ext


def extend_medial_axis(
    medial_axis: np.array,
    boundary: np.array,
    max_iter: int = 50,
    error_threshold: float = 0.05,
    min_distance_to_boundary: int = 1,
    step_size: float or int = 1,
) -> np.array:
    """
    Extend the medial axis to the boundary, using the boundary as a reference, then smooth it to get the final result.

    Parameters
    ----------
    medial_axis : np.array
        The medial axis.
    boundary : np.array
        The boundary.
    max_iter : int, optional
        The maximum number of iterations for extending the medial axis(default is 50).
    min_distance_to_boundary : int, optional
        The minimum distance to the boundary (default is 1). This determines when the extension stops.
    step_size : float or int, optional
        The step size to take when extending the medial axis (default is 1 px).

    Returns
    -------
    extended_medial_axis: np.array
        The extended and smoothed medial axis.
    """

    medial_axis_extended_roughly = extend_medial_axis_roughly(
        medial_axis, boundary, step_size=0.05
    )

    # medial_axis_extended = smooth_medial_axis(medial_axis_extended_roughly, boundary, error_threshold=0.0005)

    return medial_axis_extended_roughly
