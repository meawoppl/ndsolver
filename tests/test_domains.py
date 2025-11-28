"""Procedural test domain generation using boolean/coordinate math."""

import numpy as np


def make_circle_domain(size=150, center=None, radius=None):
    """Create a domain with a circular obstacle.

    Args:
        size: Domain size (square)
        center: (x, y) center of circle, defaults to domain center
        radius: Circle radius, defaults to size/4

    Returns:
        Binary array where 1=solid, 0=fluid
    """
    if center is None:
        center = (size / 2, size / 2)
    if radius is None:
        radius = size / 4

    y, x = np.ogrid[:size, :size]
    dist_sq = (x - center[0])**2 + (y - center[1])**2
    return (dist_sq <= radius**2).astype(np.int8)


def make_two_circles_domain(size=150):
    """Create a domain with two circular obstacles."""
    y, x = np.ogrid[:size, :size]

    # Two circles offset from center
    c1 = (size * 0.35, size * 0.5)
    c2 = (size * 0.65, size * 0.5)
    r = size * 0.15

    circle1 = (x - c1[0])**2 + (y - c1[1])**2 <= r**2
    circle2 = (x - c2[0])**2 + (y - c2[1])**2 <= r**2

    return (circle1 | circle2).astype(np.int8)


def make_annulus_domain(size=150, inner_radius=None, outer_radius=None):
    """Create a domain with an annular (ring) obstacle."""
    if inner_radius is None:
        inner_radius = size * 0.15
    if outer_radius is None:
        outer_radius = size * 0.35

    y, x = np.ogrid[:size, :size]
    cx, cy = size / 2, size / 2
    dist_sq = (x - cx)**2 + (y - cy)**2

    return ((dist_sq >= inner_radius**2) & (dist_sq <= outer_radius**2)).astype(np.int8)


def make_rectangle_domain(size=150, rect_size=None):
    """Create a domain with a rectangular obstacle."""
    if rect_size is None:
        rect_size = (size // 3, size // 4)

    solid = np.zeros((size, size), dtype=np.int8)

    x0 = (size - rect_size[0]) // 2
    y0 = (size - rect_size[1]) // 2
    x1 = x0 + rect_size[0]
    y1 = y0 + rect_size[1]

    solid[y0:y1, x0:x1] = 1
    return solid


def make_grid_domain(size=150, n_obstacles=3, obstacle_size=None):
    """Create a domain with a grid of square obstacles."""
    if obstacle_size is None:
        obstacle_size = size // (n_obstacles * 3)

    solid = np.zeros((size, size), dtype=np.int8)
    spacing = size // (n_obstacles + 1)

    for i in range(n_obstacles):
        for j in range(n_obstacles):
            cx = spacing * (i + 1)
            cy = spacing * (j + 1)
            x0 = cx - obstacle_size // 2
            y0 = cy - obstacle_size // 2
            x1 = x0 + obstacle_size
            y1 = y0 + obstacle_size
            solid[y0:y1, x0:x1] = 1

    return solid


def make_channel_domain(size=150, wall_thickness=None):
    """Create a channel with solid walls on top and bottom."""
    if wall_thickness is None:
        wall_thickness = size // 6

    solid = np.zeros((size, size), dtype=np.int8)
    solid[:wall_thickness, :] = 1
    solid[-wall_thickness:, :] = 1
    return solid


def make_staggered_pillars_domain(size=150, n_rows=3, n_cols=4):
    """Create a domain with staggered circular pillars."""
    y, x = np.ogrid[:size, :size]
    solid = np.zeros((size, size), dtype=bool)

    radius = size / (max(n_rows, n_cols) * 4)
    row_spacing = size / (n_rows + 1)
    col_spacing = size / (n_cols + 1)

    for row in range(n_rows):
        offset = col_spacing / 2 if row % 2 else 0
        cy = row_spacing * (row + 1)
        for col in range(n_cols):
            cx = col_spacing * (col + 1) + offset
            if cx < size:
                solid |= ((x - cx)**2 + (y - cy)**2 <= radius**2)

    return solid.astype(np.int8)


def make_sine_channel_domain(size=150, amplitude=None, frequency=2):
    """Create a channel with sinusoidal walls."""
    if amplitude is None:
        amplitude = size * 0.15

    y, x = np.ogrid[:size, :size]

    # Sinusoidal wall positions
    wall_y = size / 2 + amplitude * np.sin(2 * np.pi * frequency * x / size)
    wall_thickness = size * 0.1

    upper_wall = y < (wall_y - wall_thickness)
    lower_wall = y > (wall_y + wall_thickness)

    return (upper_wall | lower_wall).astype(np.int8)


def make_random_circles_domain(size=150, n_circles=10, seed=42):
    """Create a domain with randomly placed circular obstacles."""
    rng = np.random.default_rng(seed)

    y, x = np.ogrid[:size, :size]
    solid = np.zeros((size, size), dtype=bool)

    min_radius = size * 0.03
    max_radius = size * 0.1

    for _ in range(n_circles):
        cx = rng.uniform(max_radius, size - max_radius)
        cy = rng.uniform(max_radius, size - max_radius)
        r = rng.uniform(min_radius, max_radius)
        solid |= ((x - cx)**2 + (y - cy)**2 <= r**2)

    return solid.astype(np.int8)


def make_porous_media_domain(size=150, porosity=0.7, seed=42):
    """Create a random porous media domain with target porosity."""
    rng = np.random.default_rng(seed)
    solid = (rng.random((size, size)) > porosity).astype(np.int8)
    return solid


# For backward compatibility with text_test.py imports
# This generates a similar-looking domain to the original hardcoded one
def _make_legacy_domain():
    """Generate a domain similar to the original text_test.py."""
    size = 150
    y, x = np.ogrid[:size, :size]

    # Main circular region on right side
    c1 = (100, 75)
    r1 = 40
    circle1 = (x - c1[0])**2 + (y - c1[1])**2 <= r1**2

    # Secondary features
    c2 = (30, 75)
    r2 = 25
    circle2 = (x - c2[0])**2 + (y - c2[1])**2 <= r2**2

    # Small feature
    c3 = (75, 40)
    r3 = 15
    circle3 = (x - c3[0])**2 + (y - c3[1])**2 <= r3**2

    return (circle1 | circle2 | circle3).astype(np.int8)


# Default test domain (150x150)
lists = _make_legacy_domain()
test_solid = lists  # Alias


if __name__ == "__main__":
    # Demo: print domain shapes
    print("Circle domain:", make_circle_domain().shape)
    print("Two circles:", make_two_circles_domain().shape)
    print("Annulus:", make_annulus_domain().shape)
    print("Rectangle:", make_rectangle_domain().shape)
    print("Grid:", make_grid_domain().shape)
    print("Channel:", make_channel_domain().shape)
    print("Staggered pillars:", make_staggered_pillars_domain().shape)
    print("Sine channel:", make_sine_channel_domain().shape)
    print("Random circles:", make_random_circles_domain().shape)
    print("Porous media:", make_porous_media_domain().shape)
    print("Legacy domain:", lists.shape)
