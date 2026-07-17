import numpy as np
from scipy.ndimage import binary_hit_or_miss
from skimage.morphology import label
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
from scipy.interpolate import splrep, splev

"""
Collection of utility functions used by the shape analysis functions, mainly to do with  binary image processing.
"""


def find_endpoints(img: np.array) -> np.array:
    """
        A function to find the endpoints of a binary image containing a line (i.e., a medial axis).

        Parameters
        ----------
        img : np.array
            The binary image.

        Returns
        -------
        endpoints : np.array
            A binary image with only the endpoints of the line.
    """
    endpoints = np.zeros(np.shape(img))

    endpoint_options = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0],
                                 [0, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 0],
                                 [0, 0, 0],
                                 [0, 0, 0],
                                 [0, 1, 1],
                                 [0, 0, 0],
                                 [0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1],
                                 [0, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0],
                                 [0, 0, 0],
                                 [0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, 0],
                                 [1, 1, 0],
                                 [0, 0, 0],
                                 ])

    endpoint_structures = np.reshape(endpoint_options, (8, 3, 3))

    for box in endpoint_structures:
        current_attempt = binary_hit_or_miss(img, box)

        endpoints = endpoints + current_attempt
    return endpoints


def find_furthest_point(point: np.array, candidates: np.array) -> tuple[np.array, int]:
    """
    Find the furthest point from a given point, from a list of candidates.


    Parameters
    ----------
    point : np.array
        A np.array with 2 columns and 1 row (x, y)
    candidates : np.array
        An np.array with 2 columns and N rows (N being the number of candidates)

    Returns
    -------
    closest_point : np.array
        The furthest point
    max_index : int
        The index of the furthest point in the candidates array
    """
    distances_xy = candidates - point

    distances = np.sum(np.square(distances_xy), 1)

    max_index = np.where(distances == np.max(distances))

    max_index = max_index[0][0]  # Ensures we take the first one if there's two points with same distance

    closest_point = candidates[max_index, :]

    return closest_point, max_index


def find_closest_point(point: np.array, candidates: np.array) -> tuple[np.array, int]:
    """
    Find the closest point to a given point, from a list of candidates.

    Parameters
    ----------
    point : np.array
        A np.array with 2 columns and 1 row (x, y)
    candidates : np.array
        An np.array with 2 columns and N rows (N being the number of candidates)

    Returns
    -------
    closest_point : np.array
        The closest point
    min_index : int
        The index of the closest point in the candidates array
    """

    distances_xy = candidates - point

    distances = np.sum(np.square(distances_xy), 1)

    min_index = np.where(distances == np.min(distances))

    min_index = min_index[0][0]  # Ensures we take the first one if there's two points with same distance

    closest_point = candidates[min_index, :]

    return closest_point, min_index

def find_branchpoints(skeleton: np.array) -> np.array:
    """
    Function to find the branchpoints of a binary image containing a skeleton.

    Parameters
    ----------
    skeleton : np.array
        The skeletonised binary image

    Returns
    -------
    branch_points : np.array
        A binary image with only the branchpoints of the skeleton
    """
    selems = list()
    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
    selems.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
    selems.append(np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]))
    selems = [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]

    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))

    branch_points = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        branch_points |= binary_hit_or_miss(skeleton, selem)

    return branch_points


def prune_short_branches(skeleton: np.array) -> np.array:
    """
    Function to find the longest branch in a skeleton.

    Parameters
    ----------
    skeleton : np.array
        The binary image, skeletonised

    Returns
    -------
    pruned_skeleton : np.array
        A binary image with only the longest branch of the skeleton
    """
    if not skeleton.dtype == bool:
        skeleton = skeleton.astype(bool)

    branch_points = find_branchpoints(skeleton)
    separated_branches = np.copy(skeleton) ^ branch_points

    labelled_branches = label(separated_branches)

    unique_labels = np.unique(labelled_branches)

    # remove zero, as it's the background
    unique_labels = unique_labels[1:]

    # find the highest occurence
    max_occurence = 0
    max_label = 0
    for n in unique_labels:
        occurence = np.sum(labelled_branches == n)
        if occurence > max_occurence:
            max_occurence = occurence
            max_label = n

    pruned_skeleton = labelled_branches == max_label

    return pruned_skeleton


def get_coords(binary_image: np.array) -> np.array:
    """
    A function to get the coordinates of the non-zero pixels in a binary image.

    Parameters
    ----------
    binary_image : np.array
        A binary image

    Returns
    -------
    xy : np.array
        An np.array with 2 columns and N rows (N being the number of non-zero pixels)
    """

    xy = np.fliplr(np.asarray(np.where(binary_image)).T)
    return xy


def trace_axis(medial_axis_image: np.array) -> np.array:
    """
    Trace the medial axis, returning an ordered array of points from one end to the other. The first endpoint is defined as the most leftmost of the endpoints.

    Parameters
    ----------
    medial_axis_image : np.array
        The binary image with the medial axis

    Returns
    -------
    medial_axis : np.array
        An np.array with 2 columns and N rows (N being the number of points in the medial axis)
    """
    # Want to get medial axis in ordered point list.

    # Get coordinates (unordered) of the medial axis
    medial_axis_non_ordered = get_coords(medial_axis_image)

    # Grab endpoints of the medial axis line, and get their coordinates
    endpoints_image = find_endpoints(medial_axis_image)
    endpoints = get_coords(endpoints_image)

    # Pick one endpoint to start ordering from
    # This is arbitrary, so let's pick point that is the leftmost
    chosen_endpoint_index = np.where(endpoints[:, 0] == np.min(endpoints[:, 0]))
    chosen_endpoint_index = chosen_endpoint_index[0][0]

    # Create a new array where the ordered points will be stored
    medial_axis = np.zeros(medial_axis_non_ordered.shape)

    # Assign first point, and remove it from medial_axis_non_ordered
    medial_axis[0, :] = endpoints[chosen_endpoint_index, :]

    index_to_delete = np.where((medial_axis_non_ordered[:, 0] == endpoints[chosen_endpoint_index, 0]) &
                               (medial_axis_non_ordered[:, 1] == endpoints[chosen_endpoint_index, 1]))

    medial_axis_non_ordered = np.delete(medial_axis_non_ordered, index_to_delete, 0)

    for i, point in enumerate(medial_axis, start=1):
        previous_point = medial_axis[i - 1, :]

        if medial_axis_non_ordered.shape[0] > 1:
            closest_point, index = find_closest_point(previous_point, medial_axis_non_ordered)

            medial_axis[i, :] = closest_point

            # Delete point from pool
            medial_axis_non_ordered = np.delete(medial_axis_non_ordered, index, 0)
        else:
            medial_axis[i, 0] = medial_axis_non_ordered[0][0]
            medial_axis[i, 1] = medial_axis_non_ordered[0][1]
            break

    return medial_axis


def fix_coordinates_order(coordinates: np.array) -> np.array:
    """
    This function fixes the coordinate order in a list of points (usually extracted from a binary image).
    In shorts, it picks a point and then finds the closest point to it, and so on.

    Parameters
    ----------
    coordinates : np.array
        An np.array with 2 columns and N rows (N being the number of points)

    Returns
    -------
    ordered_coordinates : np.array
        An np.array with 2 columns and N rows (N being the number of points)
    """
    ordered_coordinates = np.zeros(coordinates.shape)

    # Set first coordinate of the boundary
    ordered_coordinates[0, :] = coordinates[0, :]
    coordinates = np.delete(coordinates, 0, 0)

    for i, point in enumerate(ordered_coordinates, start=1):
        previous_point = ordered_coordinates[i - 1, :]
        if coordinates.shape[0] > 1:
            closest_point, index = find_closest_point(previous_point, coordinates)

            ordered_coordinates[i, :] = closest_point

            coordinates = np.delete(coordinates, index, 0)
        else:
            candidate = np.array([[coordinates[0][0], coordinates[0][1]]])

            distance_to_candidate = np.sqrt(np.sum((candidate - previous_point) ** 2))

            if distance_to_candidate < 3:
                ordered_coordinates[i, 0] = coordinates[0][0]
                ordered_coordinates[i, 1] = coordinates[0][1]
            else:
                ordered_coordinates = np.delete(ordered_coordinates, i, 0)
            break

    return ordered_coordinates


def get_boundary_coords(binary_image: np.array) -> np.array:
    """
    Runs the find boundary function from skimage, and then extracts the coordinates of the boundary.

    Parameters
    ----------
    binary_image : np.array
        A binary image

    Returns
    -------
    boundary_xy : np.array
        An np.array with 2 columns and N rows (N being the number of boundary pixels). The points will not be ordered.
    """

    # Get boundaries from binary image
    # boundary_img = find_boundaries(binary_image, connectivity=4, mode='inner', background=0)
    boundary_xy = find_contours(binary_image)[0]

    # # ...and extract coordinates
    # boundary_xy = get_coords(boundary_img)
    return boundary_xy


def find_closest_boundary_to_axis(medial_axis_coords: np.array, boundary_coords: np.array,
                                  start_position: int) -> np.array:
    """
    A function to find the closest point on the boundary to the medial axis.

    Parameters
    ----------
    medial_axis_coords : np.array
        An np.array with 2 columns and N rows (N being the number of points in the medial axis).
    boundary_coords : np.array
        An np.array with 2 columns and M rows (M being the number of points in the boundary).
    start_position : int
        Either 1 or -1. If 1, the medial axis will be ordered from the first point to the last. If -1, the medial axis will be ordered from the last point to the first.

    Returns
    -------
    closest_on_boundary : np.array
        An np.array with 2 columns and 1 row, the coordinates of the closest point on the boundary.
    """
    # Start position can be either 1 or -1
    if start_position == 1:
        pass
    else:
        medial_axis_coords = np.flipud(medial_axis_coords)

    endpoint = medial_axis_coords[0, :]

    if len(medial_axis_coords) > 10:
        points_of_interest = medial_axis_coords[1:10, :]
    else:
        points_of_interest = medial_axis_coords[1::, :]

    search_vectors = points_of_interest - endpoint
    # This is to fix bug that happens when medial_axis_coords is only two points.
    if len(search_vectors) == 2:
        search_vectors = np.reshape(search_vectors, [1, 2])

    search_angles = np.arctan2(search_vectors[:, 1], search_vectors[:, 0])

    mean_search_angle = np.mean(search_angles)

    # Get angles from endpoint to all the boundary points.
    vectors_to_boundary_points = endpoint - boundary_coords
    angles_to_boundary_points = np.arctan2(vectors_to_boundary_points[:, 1], vectors_to_boundary_points[:, 0])

    selected_points = np.where((angles_to_boundary_points > mean_search_angle - 0.3) &
                               (angles_to_boundary_points < mean_search_angle + 0.3))

    # Filter the boundary points
    filtered_boundary_points = boundary_coords[selected_points, :]

    # closest_on_boundary = find_closest_point(endpoint, filtered_boundary_points)
    closest_on_boundary = find_furthest_point(endpoint, filtered_boundary_points)
    closest_on_boundary = np.array(closest_on_boundary[0][0])
    return closest_on_boundary


def calculate_distance_along_line(coordinates: np.array) -> np.array:
    """
    Function to calculate the distance along a line defined by a set of coordinates.

    Parameters
    ----------
    coordinates : np.array
        An np.array with 2 columns and N rows (N being the number of points in the line).

    Returns
    -------
    distances_cumulative : np.array
        An np.array with 1 column and N rows (N being the number of points in the line).
    """
    # Get distance vectors
    distance_vectors = np.diff(coordinates, axis=0)
    distance_vectors = np.concatenate((np.array([[0, 0]]), distance_vectors))

    # From the vectors, calculate the sums
    distances_individual = np.sqrt(np.sum(distance_vectors ** 2, axis=1))
    distances_cumulative = np.cumsum(distances_individual)

    return distances_cumulative


def get_width_profile_lines(medial_axis: np.array, n_points: int = 3, line_magnitude: int or float = 10):
    """
    Function to get the lines that will be used to calculate the width of the bacterium through fitting.

    Parameters
    ----------
    medial_axis : np.array
        An np.array with 2 columns and N rows (N being the number of points in the medial axis).
    n_points : int
        The number of points to sample along the medial axis.
    line_magnitude : int or float
        The length of the lines that will be perpendicular to the medial axis.

    Returns
    -------
    x_high, y_high, x_low, y_low : np.arrays
        np.arrays with 1 column and N rows (N being the number of points in the medial axis).
    """

    # Find slope of medial axis
    delta_vectors = np.diff(medial_axis, axis=0)
    parallel_angles = np.arctan2(delta_vectors[:, 1], delta_vectors[:, 0])
    perpendicular_angles = parallel_angles + np.deg2rad(90)

    # Remove first point from the medial axis, to make it same length as other vectors
    medial_axis = np.delete(medial_axis, 0, axis=0)

    # Get arclength curves
    distances = calculate_distance_along_line(medial_axis)

    # Get spline fit of distances vs x
    x_coords = medial_axis[:, 0]
    y_coords = medial_axis[:, 1]


    if len(distances) > 3:
        x_spline = splrep(distances, x_coords)
        y_spline = splrep(distances, y_coords)
        angles_spline = splrep(distances, perpendicular_angles)

        sampling_distances = np.linspace(0, distances[-1], n_points, endpoint=False)

        x_points = []
        y_points = []
        angles = []

        for val in sampling_distances:
            x_points.append(splev(val, x_spline))
            y_points.append(splev(val, y_spline))
            angles.append(splev(val, angles_spline))
    else:
        x_points = x_coords
        y_points = y_coords
        angles = perpendicular_angles

    x_ref = x_points
    y_ref = y_points
    perpendicular_angles = angles

    # Create lines 1 distance_line away from medial axis point, in each direction
    dx_perp = np.cos(perpendicular_angles) * line_magnitude / 2
    dy_perp = np.sin(perpendicular_angles) * line_magnitude / 2

    x_high = x_ref + dx_perp
    y_high = y_ref + dy_perp

    x_low = x_ref - dx_perp
    y_low = y_ref - dy_perp

    return x_high, y_high, x_low, y_low


def apply_mask_to_image(img: np.array, mask: np.array, method: str = 'min') -> np.array:
    """
    A function to apply a dilation to a mask, and then apply the mask to the image.
    This makes all the points outside the region of interest black.

    Parameters
    ----------
    img : np.array
        The image to be masked
    mask : np.array
        The mask to be applied to the image

    Returns
    -------
    img_masked : np.array
        The masked image
    """

    mask_inverted = np.invert(np.copy(mask))

    img_masked = np.copy(img)
        
    # Below is to avoid problems with different data types
    if mask_inverted.dtype == 'bool':
        idx = mask_inverted == True
    else:
        idx = mask_inverted == 255

    # img_masked[idx] = np.mean(img)
    if method == 'min':
        img_masked[idx] = np.min(img)
    elif method == 'max':
        img_masked[idx] = np.max(img)
    elif method == 'mean':
        img_masked[idx] = np.mean(img)
    else:
        # raise a warning
        Warning('Method not recognised, using min instead')
        img_masked[idx] = np.min(img)

    return img_masked