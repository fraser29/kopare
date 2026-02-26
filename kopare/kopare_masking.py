from ngawari import vtkfilters
import vtk
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label


def mask_external_air(imageData: vtk.vtkImageData, median_filter_size: int, arrayName: str) -> vtk.vtkImageData:

    imageData_med = vtkfilters.filterVtiMedian(imageData, median_filter_size)
    A = vtkfilters.getArrayAsNumpy(imageData_med, arrayName, RETURN_3D=True)
    threshold = threshold_otsu(A) * 0.5
    mask = A < threshold
    mask = keep_components_touching_side_faces(mask)
    return mask, imageData_med, threshold


def keep_components_touching_side_faces(mask: np.ndarray) -> np.ndarray:
    """
    Keep only connected mask components that touch side faces 1-4.

    Assumes volume axis order is (z, y, x):
    - face 1/2: x=0 and x=-1
    - face 3/4: y=0 and y=-1
    - face 5/6 (excluded): z=0 and z=-1
    """
    if mask.size == 0:
        return mask

    labels = label(mask, connectivity=1)
    # print(f"Number of components: {labels.max()}, {labels.shape}")
    # for iLabel in range(1, labels.max() + 1):
    #     print(f"Component {iLabel} size: {np.sum(labels == iLabel)}")
    # return labels
    if labels.max() == 0:
        return mask
    touching_labels = np.unique(
        np.concatenate(
            [
                labels[0, :, :].ravel(),   # x-min
                labels[-1, :, :].ravel(),  # x-max
                labels[:, 0, :].ravel(),   # y-min
                labels[:, -1, :].ravel(),  # y-max
            ]
        )
    )
    touching_labels = touching_labels[touching_labels != 0]
    if touching_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)

    return np.isin(labels, touching_labels).astype(np.int16)