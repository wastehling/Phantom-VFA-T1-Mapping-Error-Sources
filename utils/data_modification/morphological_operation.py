import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import label


def morphological_closing(mask: np.ndarray, structure: np.ndarray = None):
    """
    Perform morphological closing on a boolean mask.
    Closing is a dilation followed by an erosion.

    Parameters:
    - mask: numpy array of type bool
    - structure: Structuring element (default is a 3x3 square)

    Returns:
    - Closed boolean mask
    """
    if structure is None:
        structure = np.ones((3, 3), dtype=bool)

    if mask.ndim == 2:
        dilated = binary_dilation(mask, structure)
        closed_mask = binary_erosion(dilated, structure)

    elif mask.ndim == 3:
        # If 3D, apply operation slice-wise along the third dimension
        closed_mask = np.zeros_like(mask, dtype=bool)
        for i in range(mask.shape[2]):
            dilated = binary_dilation(mask[:, :, i], structure)
            closed_mask[:, :, i] = binary_erosion(dilated, structure)

    else:
        raise ValueError("Input mask must be 2D or 3D.")

    return closed_mask


def largest_connected_component_3d(mask: np.ndarray):
    """
    Finds the largest connected component in a 3D boolean mask.

    Parameters:
    - mask: 3D numpy array of dtype bool

    Returns:
    - A boolean array of the same shape as `mask`, with only the largest connected component retained.
    """
    if mask.ndim != 3:
        raise ValueError("Input mask must be a 3D array.")

    # Define connectivity (6-connectivity in 3D)
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    labeled_array, num_features = label(mask, structure=structure)

    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    # Count voxels in each component
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # background is labeled 0

    # Get label of the largest component
    largest_label = component_sizes.argmax()

    # Create mask for the largest component
    largest_component_mask = labeled_array == largest_label

    return largest_component_mask
