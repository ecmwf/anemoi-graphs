import numpy as np
import torch


def cartesian_to_latlon_degrees(xyz: np.ndarray) -> np.ndarray:
    """3D to lat-lon (in degrees) conversion.

    Convert 3D coordinates of points to the (lat, lon) on the sphere containing
    them.

    Parameters
    ----------
    xyz : np.ndarray
        The 3D coordinates of points.

    Returns
    -------
    np.ndarray
        A 2D array of lat-lon coordinates of shape (N, 2).
    """
    lat = np.arcsin(xyz[..., 2] / (xyz**2).sum(axis=1)) * 180.0 / np.pi
    lon = np.arctan2(xyz[..., 1], xyz[..., 0]) * 180.0 / np.pi
    return np.array((lat, lon), dtype=np.float32).transpose()


def cartesian_to_latlon_rad(xyz: np.ndarray) -> np.ndarray:
    """3D to lat-lon (in radians) conversion.

    Convert 3D coordinates of points to its coordinates on the sphere containing
    them.

    Parameters
    ----------
    xyz : np.ndarray
        The 3D coordinates of points.

    Returns
    -------
    np.ndarray
        A 2D array of the coordinates of shape (N, 2) in radians.
    """
    lat = np.arcsin(xyz[..., 2] / (xyz**2).sum(axis=1))
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    return np.array((lat, lon), dtype=np.float32).transpose()


def latlon_rad_to_cartesian(locations: torch.Tensor, radius: float = 1) -> torch.Tensor:
    """Convert planar coordinates to 3D coordinates in a sphere.

    Parameters
    ----------
    locations : torch.Tensor of shape (N, 2)
        The 2D coordinates of the points, in radians.
    radius : float, optional
        The radius of the sphere containing los points. Defaults to the unit sphere.

    Returns
    -------
    torch.Tensor of shape (N, 3)
        3D coordinates of the points in the sphere.
    """
    latr, lonr = locations[..., 0], locations[..., 1]
    x = radius * torch.cos(latr) * torch.cos(lonr)
    y = radius * torch.cos(latr) * torch.sin(lonr)
    z = radius * torch.sin(latr)
    return torch.stack((x, y, z), dim=-1)
