from ngawari import vtkfilters
import vtk
import numpy as np
import logging
from skimage.filters import threshold_otsu, rank
from skimage.measure import label
from skimage import img_as_float
from skimage.morphology import binary_erosion, ball
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure
import SimpleITK as sitk

logger = logging.getLogger("kopare")

def mask_external_air(imageData: vtk.vtkImageData, 
                    arrayName: str,
                    n_shrink_wrap_iterations: int = 5,
                    ) -> vtk.vtkImageData :
    """
    Mask external air in a 3D volume data.

    Args:
        imageData: vtkImageData object containing the volume data.
        arrayName: name of the array containing the volume data.
        n_shrink_wrap_iterations: number of iterations for the shrink wrap algorithm. Default is 5.
    Returns:
        mask: numpy array containing the mask.
    """
    if n_shrink_wrap_iterations > 5:
        n_shrink_wrap_iterations = 5
        logger.warning(f"Set n_shrink_wrap_iterations to {n_shrink_wrap_iterations} (to avoid memory issues)")
    A = vtkfilters.getArrayAsNumpy(imageData, arrayName, RETURN_3D=True)
    airThreshold = _get_air_threshold_from_slices(A)
    logger.debug(f"Air threshold is {airThreshold}")
    face_contour = _build_air_contour(imageData, airThreshold, n_shrink_wrap_iterations)
    imageData_mask = vtkfilters.duplicateImageData(imageData)
    vtkfilters.filterMaskImageBySurface(imageData_mask, face_contour, fill_value=1, arrayName="LabelMap")
    # mask = A < airThreshold
    # mask = keep_components_touching_side_faces(mask)
    vtkfilters.setArrayAsScalars(imageData_mask, "LabelMap")
    return imageData_mask


def _build_air_contour(imageData: vtk.vtkImageData, airThreshold: float, n_shrink_wrap_iterations: int) -> vtk.vtkPolyData:
    """
    Build a contour of the air in the volume data.
    """
    face_contour = vtkfilters.contourFilter(imageData, airThreshold)
    face_contour = vtkfilters.getConnectedRegionLargest(face_contour) 
    initial_wrap = vtkfilters.shrinkWrapData(face_contour)
    initial_wrap = mark_planar_faces(initial_wrap)
    face_contour_closed = iterative_shrink_wrap(face_contour, wrapped_surface=initial_wrap,
                                                max_iterations=n_shrink_wrap_iterations)
    return face_contour_closed


def mark_planar_faces(shrinkwrap: vtk.vtkPolyData,
                      planar_angle_threshold_deg: float = 10.0,
                      planar_percent_threshold: float = 35.0,
                      within_bounds_rel_threshold: float = 0.1) -> vtk.vtkPolyData:
    """
    """
    n_cells = shrinkwrap.GetNumberOfCells()
    face_labels = np.zeros(n_cells, dtype=np.int32)
    face_fix = np.zeros(n_cells, dtype=np.int32)
    centroids, norms = vtkfilters.getPolyDataCenterPtNormal(shrinkwrap)
    cp = np.array(shrinkwrap.GetCenter())
    bounds = shrinkwrap.GetBounds()
    sides = {1: [0, [-1,0,0]],
             2: [0, [1,0,0]],
             3: [1, [0,-1,0]],
             4: [1, [0,1,0]],
             5: [2, [0,0,-1]],
             6: [2, [0,0,1]]
             }
    bounds_pts = [cp for _ in range(6)]
    for i in range(6):
        bounds_pts[i][sides[i+1][0]] = bounds[i]
    max_bound = max([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    plane_dist_thresh = within_bounds_rel_threshold * max_bound
    logger.debug(f"Shrinkwrap: Maximum bound = {max_bound}. Plane distance threshold = {plane_dist_thresh:0.1f}")

    side_counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    for k1 in range(n_cells):
        for k2 in range(6):
            angle = vtkfilters.ftk.angleBetween2Vec(norms[k1], sides[k2+1][1], RETURN_DEGREES=True)
            plane_dist = vtkfilters.ftk.distanceToPlane(centroids[k1], sides[k2+1][1], bounds_pts[k2])
            if plane_dist < plane_dist_thresh: # Close to "boundary"
                if angle < 30: # Face normal is general orthogonal direction
                    side_counts[k2+1] += 1
                if angle < planar_angle_threshold_deg: # Face normal is true orthogonal direction
                    face_labels[k1] = k2+1
    logger.debug(f"Total faces on each side: {side_counts}")
    for k1 in side_counts.keys():
        if k1 == 0:
            continue
        side_counts[k1] = int(np.sum(face_labels==k1) / float(side_counts[k1]) * 100.0)
    logger.debug(f"Percent planar faces on each side: {side_counts}")
    for k1 in range(n_cells):
        if side_counts[face_labels[k1]] > planar_percent_threshold:
            face_fix[k1] = 1.0
    vtkfilters.addNpArray(shrinkwrap, face_labels, "Planar", pointData=False)
    vtkfilters.addNpArray(shrinkwrap, face_fix, "FIXED", pointData=False)
    shrinkwrap = vtkfilters.cellToPointData(shrinkwrap)
    return shrinkwrap


def _subdivide(polydata, nSubdivisions: int = 2):
    filter = vtk.vtkLinearSubdivisionFilter()
    filter.SetInputData(polydata)
    filter.SetNumberOfSubdivisions(nSubdivisions)
    filter.Update()
    return filter.GetOutput()


def iterative_shrink_wrap(target_surface: vtk.vtkPolyData,
                            wrapped_surface: vtk.vtkPolyData,
                            max_iterations: int = 5,
                            fixed_point_threshold: float = 0.3 ):
    
    c0 = 0
    for c0 in range(max_iterations):
        c0 += 1
        wrapped_surface = _subdivide(wrapped_surface, 1)
        pts_orig = vtkfilters.getPtsAsNumpy(wrapped_surface)
        logger.debug(f"Iterative shrink wrap, iteration {c0}, number of points {len(pts_orig)}")
        pts_fixed = vtkfilters.getArrayAsNumpy(wrapped_surface, "FIXED") > fixed_point_threshold
        wrapped_surface = vtkfilters.shrinkWrapData(target_surface, wrapped_surface)
        for k1 in range(wrapped_surface.GetNumberOfPoints()):
            if pts_fixed[k1]:
                wrapped_surface.GetPoints().SetPoint(k1, pts_orig[k1])
    else:
        logger.warning(f"iterative_shrink_wrap reached {max_iterations} max_iterations without full convergence.")
    return wrapped_surface



def _get_air_threshold_from_slices(Array3D: np.ndarray) -> float:
    """
    Identify the threshold value for air

    """
    # dims = Array3D.shape
    # thresholds = [threshold_otsu(Array3D[i, :, :]) for i in range(dims[0])]
    # return np.median(thresholds)
    walls = {
        0: Array3D[0, :, :],
        2: Array3D[-1, :, :],
        1: Array3D[:, 0, :],
        3: Array3D[:, -1, :],
        4: Array3D[:, :, 0],
        5: Array3D[:, :, -1],
    }
    Awalls = np.hstack([walls[iFace].flatten() for iFace in range(6)])
    threshold = threshold_otsu(Awalls)
    # print(f"Threshold: {threshold}")
    # faces_air = {iFace: bool(np.percentile(walls[iFace], 99) < threshold) for iFace in range(6)}
    # Awalls_air = np.hstack([walls[iFace].flatten() for iFace in range(6) if faces_air[iFace]])
    return threshold


def denoise3DA(A3D, alpha, patch_size, patch_distance):
    """
    Non-local means denoising of a 3D array.

    Args:
        A3D: 3D numpy array.
        alpha: alpha parameter for the non-local means denoising.
        patch_size: patch size for the non-local means denoising.
        patch_distance: patch distance for the non-local means denoising.

    Returns:
        dnA: 3D numpy array containing the denoised data.
    """
    dnA = np.zeros(A3D.shape)
    for k1 in range(A3D.shape[2]):
        iA = img_as_float(A3D[:,:,k1])
        sigmaEst = np.mean(estimate_sigma(iA))
        dnA[:,:,k1] = denoise_nl_means(iA, h=alpha*sigmaEst, patch_size=patch_size, patch_distance=patch_distance)
    return dnA



def keep_components_touching_side_faces(mask: np.ndarray) -> np.ndarray:
    """
    Keep only connected mask components that touch side faces 1-4.

    Assumes volume axis order is (z, y, x):
    - face 1/2: x=0 and x=-1
    - face 3/4: y=0 and y=-1
    - face 5/6 (excluded): z=0 and z=-1

    Returns:
        np.ndarray: 3D numpy array containing the mask.
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


def scaleImageData(imageData, scaleFactor):
    res = imageData.GetSpacing()
    oo = imageData.GetOrigin()
    imageData.SetSpacing([i*scaleFactor for i in res])
    imageData.SetOrigin([i*scaleFactor for i in oo])
    return imageData

# =========================================================================================
# Mask edge smoothin
# =========================================================================================

def _gaussianSmooth(imageData, standardDeviation, radiusFactor):
    gaussian = vtkfilters.vtk.vtkImageGaussianSmooth()
    gaussian.SetDimensionality(3)
    gaussian.SetInputData(imageData)
    gaussian.SetStandardDeviations(standardDeviation, standardDeviation, standardDeviation)
    gaussian.SetRadiusFactors(radiusFactor, radiusFactor, radiusFactor)
    gaussian.Update()
    return gaussian.GetOutput()


def _get_edge_mask(mask_array, nErode_iter):
    bw = (mask_array > 0.5).astype(int)
    bw_e = binary_erosion(bw, ball(nErode_iter))
    # if nErode_iter > 1:
    #     for _ in range(nErode_iter):
    #         bw_e = binary_erosion(bw_e)
    edge_bw = np.logical_and(bw, ~bw_e)
    return edge_bw


def smooth_at_mask_edge(AImage, Amask, n_iterations):
    edge_bw = _get_edge_mask(mask_array=Amask, nErode_iter=n_iterations)
    AImageMax = rank.maximum(AImage, footprint=ball(n_iterations))
    AImage[edge_bw] = AImageMax[edge_bw]
    return AImage

    





# =========================================================================================
# Bias field correction
# =========================================================================================
def sitk_image_fromArray(arr3D: np.ndarray, origin: tuple[float, float, float], spacing: tuple[float, float, float]) -> sitk.Image:
    """
    Create a SimpleITK image from a 3D numpy array.

    Args:
        arr3D: 3D numpy array.
        origin: origin of the image.
        spacing: spacing of the image.
    """
    img_sitk = sitk.GetImageFromArray(arr3D)
    img_sitk.SetOrigin(origin)
    img_sitk.SetSpacing(spacing)
    return img_sitk

def sitkHelper_VTKToITKImage(imgVTK, arrayName=None):
    if arrayName is not None:
        imgA = vtkfilters.getArrayAsNumpy(imgVTK, arrayName)
    else:
        imgA = vtkfilters.getScalarsAsNumpy(imgVTK)
    dims = list(imgVTK.GetDimensions())
    if np.ndim(imgA) > 1:
        imgA = np.reshape(imgA, dims+[3], 'F')
        imgA = np.transpose(imgA, (2, 1, 0, 3))
    else:
        imgA = np.reshape(imgA, imgVTK.GetDimensions(), 'F')
        imgA = np.transpose(imgA, (2, 1, 0))

    oo = imgVTK.GetPoint(0)
    imgSITK = sitk_image_fromArray(imgA, oo, imgVTK.GetSpacing())
    imgSITK_ = sitk.Cast(imgSITK, sitk.sitkFloat32)
    return imgSITK_

def sitkHelper_ITKToVTKImage(imObj: sitk.Image, arrayName: str = 'PixelData') -> vtk.vtkImageData:
    """

    Args:
        imObj: SimpleITK image object.
        arrayName: name of the array to set in the vtkImageData object. Default is 'PixelData'.

    Returns:
        vtkImageData object.
    """
    ii = vtk.vtkImageData()
    ii.SetSpacing(imObj.GetSpacing())
    ii.SetOrigin(imObj.GetOrigin())
    k,j,i = imObj.GetSize()
    ii.SetDimensions([k,j,i])
    arr = sitk.GetArrayFromImage(imObj)
    if np.ndim(arr) == 4:
        arr = np.transpose(arr, (2, 1, 0, 3))
        arr = np.reshape(arr, (np.prod(arr.shape[:3]), arr.shape[-1]), 'F')
    else:
        arr = np.transpose(arr, (2, 1, 0))
        arr = np.ndarray.flatten(arr, 'F')
    vtkfilters.addNpArray(ii, arr, arrayName, SET_SCALAR=True)
    return ii



def biasFieldCorrection(imageData: vtk.vtkImageData, arrayName: str, numberFittingLevels: int = 4,
                        maxIterations: int = 50, shrinkFactor: int | list[int] | None = None, maskImage: vtk.vtkImageData | None = None) -> vtk.vtkImageData:
    """
    Bias field correction using SimpleITK.

    Args:
        imageData: vtkImageData object containing the volume data.
        arrayName: name of the array containing the volume data.
        numberFittingLevels: number of fitting levels. Default is 4.
        maxIterations: maximum number of iterations. Default is 50.
        shrinkFactor: shrink factor. Default is None.
        maskImage: vtkImageData object containing the mask. Default is None.
        arrayNameOut: name of the array to set in the vtkImageData object. Default is None.

    Returns:
        vtkImageData object containing the bias field corrected volume data.
    """
    inputImage = sitkHelper_VTKToITKImage(imageData, arrayName=arrayName)
    image = inputImage
    if shrinkFactor is not None:
        if not type(shrinkFactor) == list:
            shrinkFactor = [shrinkFactor]
        shrinkFactor = shrinkFactor * inputImage.GetDimension()
        image = sitk.Shrink(inputImage, shrinkFactor)
        if maskImage is not None:
            maskImage = sitk.Shrink(maskImage, shrinkFactor)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([maxIterations] * numberFittingLevels)
    ## 
    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.1)
    corrector.SetConvergenceThreshold(1e-5)
    corrector.SetWienerFilterNoise(0.01)
    corrector.SetNumberOfHistogramBins(128)
    corrector.SetSplineOrder(3)
    ##
    if maskImage is not None:
        output = corrector.Execute(image, maskImage)
    else:
        output = corrector.Execute(image)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    output = inputImage / sitk.Exp( log_bias_field )            
    output = sitk.RescaleIntensity(output)
    output = sitk.Cast(output, sitk.sitkUInt16)
    return sitkHelper_ITKToVTKImage(output, arrayName=arrayName)


# =========================================================================================
# SIGNAL INVERSION
# =========================================================================================
def contrastStretch_percentile(array: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    # If get a zerodivision error then have a problem and should throw error.
    A_plus = array[array>0.00001]
    pLow, pHigh = np.percentile(A_plus, (pmin, pmax))
    return exposure.rescale_intensity(array, in_range=(pLow, pHigh), out_range=(100, 2.0**13)).astype(np.int16)



def signalLogInverse(array: np.ndarray) -> np.ndarray:
    A_plus1 = array + 1.0
    AI = -1.0 * np.log(A_plus1)
    AI = AI + np.abs(np.min(AI))
    return AI

