import numpy as np
import pyvista as pv


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


def line_points(points: np.ndarray):
    line = polyline_from_points(points)
    tube = line.tube(radius=2)
    return tube


def pv_series(p, points, color, width=10):
    p.add_mesh(line_points(points[:, 1, :]), render_points_as_spheres=True, color=color)
    p.add_mesh(line_points(points[:, 2, :]), render_points_as_spheres=True, color=color)
    p.add_mesh(line_points(2 * points[:, 0, :] - points[:, 1, :]), render_points_as_spheres=True, color=color)
    p.add_mesh(line_points(2 * points[:, 0, :] - points[:, 2, :]), render_points_as_spheres=True, color=color)

    p.add_lines(np.asarray([2 * points[0, 0, :] - points[0, 1, :], points[0, 2, :]]), color=color, width=width)
    p.add_lines(np.asarray([2 * points[0, 0, :] - points[0, 2, :], points[0, 1, :]]), color=color, width=width)
    p.add_lines(np.asarray([2 * points[0, 0, :] - points[0, 1, :], 2 * points[0, 0, :] - points[0, 2, :]]), color=color, width=width)
    p.add_lines(np.asarray([points[0, 1, :], points[0, 2, :]]), color=color, width=width)

    p.add_lines(np.asarray([2 * points[-1, 0, :] - points[-1, 1, :], points[-1, 2, :]]), color=color, width=width)
    p.add_lines(np.asarray([2 * points[-1, 0, :] - points[-1, 2, :], points[-1, 1, :]]), color=color, width=width)
    p.add_lines(np.asarray([2 * points[-1, 0, :] - points[-1, 1, :], 2 * points[-1, 0, :] - points[-1, 2, :]]), color=color, width=width)
    p.add_lines(np.asarray([points[-1, 1, :], points[-1, 2, :]]), color=color, width=width)
