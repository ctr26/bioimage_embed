import numpy as np
from scipy.interpolate import interp1d, splprep, splev


def cart2pol(x, y):
    return (np.sqrt(x**2 + y**2), np.arctan2(y, x))


def pol2cart(rho, phi):
    return (rho * np.cos(phi), rho * np.sin(phi))


def cubic_polar_resample_contour(contour: np.array, size: int) -> np.array:
    """Star convex resampling of a contour using cubic interpolation

    Args:
        contour (np.Array): scikit image contour
        size (int): control points to interpolate to

    Returns:
        np.Array: new contour
    """
    contour_y, contour_x = contour[0][:, 0], contour[0][:, 1]
    rho, phi = cart2pol(contour_x, contour_y)

    rho_interp = interp1d(np.linspace(0, 1, len(rho)), rho, kind="cubic")(
        np.linspace(0, 1, size)
    )
    phi_interp = interp1d(np.linspace(0, 1, len(phi)), phi, kind="cubic")(
        np.linspace(0, 1, size)
    )

    xii, yii = pol2cart(rho_interp, phi_interp)
    return np.array([xii, yii])


def contour_to_xy(contour: np.array):
    return contour[0][:, 0], contour[0][:, 1]


def resample_contour_with_equal_arc_length(contour, size):
    """Resample a contour using a uniform spline
    Author: @ctr26

    Args:
        contour (np.array): scikit image contour
        size (int): Control points to interpolate to

    Returns:
        np.Array: new contour
    """
    # Convert contour to x and y arrays (assuming this function is defined)
    contour_y, contour_x = contour_to_xy(contour)

    # Create a spline representation of the contour
    tck, u = splprep([contour_x, contour_y], s=0)

    # Compute the derivative of the spline
    dx, dy = splev(u, tck, der=1)

    # Calculate the arc length as the integral of the speed
    arc_length = cumtrapz(np.sqrt(dx**2 + dy**2), u, initial=0)
    total_length = arc_length[-1]

    # Find equally spaced arc length points
    arc_length_points = np.linspace(0, total_length, size)

    # Map these points to corresponding u values on the spline
    u_new = np.interp(arc_length_points, arc_length, u)

    # Evaluate the spline at these new u values
    new_points = np.array(splev(u_new, tck))

    return new_points.T  # Transposed to match the format of the input contour


def uniform_spline_resample_contour(contour: np.array, size: int) -> np.array:
    """Resample a contour using a uniform spline
    Author: @afoix

    Args:
        contour (np.array): scikit image contour
        size (int): Control points to interpolate to

    Returns:
        np.Array: new contour
    """
    contour_y, contour_x = contour_to_xy(contour)
    tck, u = splprep([contour_x, contour_y], s=0)
    u_new = np.linspace(u.min(), u.max(), size)
    return np.array(splev(u_new, tck))
