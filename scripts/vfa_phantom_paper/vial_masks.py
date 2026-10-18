import numpy as np
from scipy import ndimage


def detect_vials(image, n_vials, min_area=150, max_area=1200, min_roundness=0.7):
    """
    Detects phantom vials as round regions enclosed by thin dark walls. The walls are enhanced with a black top-hat
    filter, so detection does not depend on the contrast between vial content and background.
    Args:
        image: 2D image with visible vial walls, e.g. the lowest flip angle
        n_vials: number of vials to return (largest regions first)
        min_area, max_area: allowed vial area in pixels
        min_roundness: minimal area / area of the enclosing circle

    Returns:
        list of (x, y, r): centre (column, row) and equivalent inner radius in pixels
    """
    smooth = ndimage.gaussian_filter(image / np.percentile(image, 99), 0.7)
    tophat = ndimage.black_tophat(smooth, size=7)
    inside_phantom = ndimage.binary_fill_holes(smooth > 0.15)

    candidates = []
    for threshold in (0.05, 0.08, 0.12, 0.16, 0.2):
        walls = ndimage.binary_dilation(tophat > threshold)
        labels, _ = ndimage.label(inside_phantom & ~walls)
        for idx, bbox in enumerate(ndimage.find_objects(labels), 1):
            component = labels[bbox] == idx
            area = component.sum()
            height, width = component.shape
            if not min_area <= area <= max_area:
                continue
            if area / (np.pi * (max(height, width) / 2) ** 2) < min_roundness or min(height, width) / max(height, width) < 0.8:
                continue
            cy, cx = ndimage.center_of_mass(component)
            candidates.append((bbox[1].start + cx, bbox[0].start + cy, np.sqrt(area / np.pi)))

    # the same vial is found at several thresholds, keep the largest non-overlapping regions
    vials = []
    for x, y, r in sorted(candidates, key=lambda c: -c[2]):
        if all((x - vx) ** 2 + (y - vy) ** 2 > (0.8 * (r + vr)) ** 2 for vx, vy, vr in vials):
            vials.append((x, y, r))
    if len(vials) < n_vials:
        raise ValueError(f'Found {len(vials)} vials, expected {n_vials}')
    return vials[:n_vials]


def circular_vial_masks(shape, vials, shrink):
    """
    Creates a label image with one circular mask per vial, slightly smaller than the inner vial radius.
    Args:
        shape: (rows, columns)
        vials: list of (x, y, r) from detect_vials
        shrink: number of pixels subtracted from the vial radius

    Returns:
        label image, 0 = background, 1..n = vials
    """
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    labels = np.zeros(shape, dtype=np.int16)
    for idx, (x, y, r) in enumerate(vials, 1):
        labels[(cols - x) ** 2 + (rows - y) ** 2 <= (r - shrink) ** 2] = idx
    return labels
