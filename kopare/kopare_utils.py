from ngawari import vtkfilters
import vtk
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
import SimpleITK as sitk
from kopare.iterative_shrink_wrap import iterative_shrink_wrap


def mask_external_air(imageData: vtk.vtkImageData, 
                    median_filter_size: int, 
                    arrayName: str,
                    numberFittingLevels: int = 4,
                    maxIterations: int = 50,
                    shrinkFactor: int | list[int] | None = None,
                    maskImage: vtk.vtkImageData | None = None,
                    arrayNameOut: str | None = None,
                    denoising_alpha: float = 5,
                    denoising_patch_size: int = 9,
                    denoising_patch_distance: int = 5,
                    gaussian_smoothing_sigma: float = 0.0,
                    WORKING_DIR: str = None,
                    kopare_class = None,
                    ) -> np.ndarray :
    """
    Mask external air in a 3D volume data.

    Args:
        imageData: vtkImageData object containing the volume data.
        median_filter_size: size of the median filter to apply to the volume data.
        arrayName: name of the array containing the volume data.
        numberFittingLevels: number of fitting levels. Default is 4.
        maxIterations: maximum number of iterations. Default is 50.
        shrinkFactor: shrink factor. Default is None.
        maskImage: vtkImageData object containing the mask. Default is None.
        arrayNameOut: name of the array to set in the vtkImageData object. Default is None.
        denoising_alpha: alpha parameter for the non-local means denoising. Default is 5.
        denoising_patch_size: patch size for the non-local means denoising. Default is 9.
        denoising_patch_distance: patch distance for the non-local means denoising. Default is 5.
        gaussian_smoothing_sigma: sigma parameter for the Gaussian smoothing. Default is 0.
    Returns:
        mask: numpy array containing the mask.
    """
    A = vtkfilters.getArrayAsNumpy(imageData, arrayName, RETURN_3D=True)
    imageData = vtkfilters.duplicateImageData(imageData)
    if denoising_alpha > 0:
        A = denoise3DA(A, denoising_alpha, denoising_patch_size, denoising_patch_distance)
    vtkfilters.setArrayFromNumpy(imageData, A, arrayName, IS_3D=True, SET_SCALAR=True)
    if (kopare_class is not None) and (denoising_alpha > 0):
        kopare_class._write_intermediate_files(imageData, f"imageData_denoised")


    if numberFittingLevels > 0:
        imagedata_BC = biasFieldCorrection(imageData=imageData, 
                                        arrayName=arrayName, 
                                        numberFittingLevels=numberFittingLevels, 
                                        maxIterations=maxIterations, 
                                        shrinkFactor=shrinkFactor, 
                                        maskImage=maskImage, 
                                        arrayNameOut=arrayNameOut)
        if kopare_class is not None:
            kopare_class._write_intermediate_files(imagedata_BC, f"imagedata_BC")
    else:
        imagedata_BC = imageData

    SMOOTHED = False
    if median_filter_size > 0:
        imageData_smooth = vtkfilters.filterVtiMedian(vtiObj=imagedata_BC, filterKernalSize=median_filter_size)
        SMOOTHED = True
    elif gaussian_smoothing_sigma > 0:
        imageData_smooth = vtkfilters.filterVtiGaussian(vtiObj=imagedata_BC, sigma=gaussian_smoothing_sigma)
        SMOOTHED = True
    else:
        imageData_smooth = imagedata_BC
    if kopare_class is not None and SMOOTHED:
        kopare_class._write_intermediate_files(imageData_smooth, f"imageData_smooth")

    A = vtkfilters.getArrayAsNumpy(imageData_smooth, arrayName, RETURN_3D=True)
    airThreshold = _get_air_threshold_from_slices(A)
    face_contour = _build_air_contour(imageData_smooth, arrayName, airThreshold)
    if kopare_class is not None:
        kopare_class._write_intermediate_files(face_contour, f"face_contour")
    mask = A < airThreshold
    mask = keep_components_touching_side_faces(mask)
    return mask


def _subdivide(polydata, nSubdivisions: int = 2):
    filter = vtk.vtkLinearSubdivisionFilter()
    filter.SetInputData(polydata)
    filter.SetNumberOfSubdivisions(nSubdivisions)
    filter.Update()
    return filter.GetOutput()


def _build_air_contour(imageData: vtk.vtkImageData, arrayName: str, airThreshold: float) -> vtk.vtkPolyData:
    """
    Build a contour of the air in the volume data.
    """
    face_contour = vtkfilters.contourFilter(imageData, airThreshold)
    face_contour = vtkfilters.getConnectedRegionLargest(face_contour)

    # Close contour at boundaries
    image_bounds = imageData.GetBounds()
    res = np.mean(imageData.GetSpacing())
    shrinkSrc = vtkfilters.buildSphereSource(face_contour.GetCenter(), .35, 50)
    face_contour_closed = vtkfilters.shrinkWrapData(face_contour, shrinkSrc)
    face_contour_closed = _subdivide(face_contour_closed, 1)
    face_contour_closed = vtkfilters.shrinkWrapData(face_contour, face_contour_closed)
    face_contour_closed = _subdivide(face_contour_closed, 2)
    face_contour_closed = vtkfilters.shrinkWrapData(face_contour, face_contour_closed)
    face_contour_closed = _subdivide(face_contour_closed, 1)
    face_contour_closed = vtkfilters.shrinkWrapData(face_contour, face_contour_closed)
    # face_contour_closed = vtkfilters.decimateTris(face_contour_closed, 0.9)
    # face_contour_closed = iterative_shrink_wrap(face_contour, 
    #                                             movement_threshold=0.002,
    #                                             max_iterations=50, 
    #                                             sphere_resolution=80,
    #                                             max_edge_length=0.02)
    # MADE_FACE_CONTOUR = False
    # for X, N in zip(image_bounds, [0, 0, 1, 1, 2, 2]):
    #     normal = np.zeros(3)
    #     normal[N] = 1
    #     cp = np.array(imageData.GetCenter())
    #     X1 = np.array(imageData.GetCenter())
    #     X1[N] = X
    #     dN = cp - X1
    #     dN = dN / np.linalg.norm(dN)
    #     X2 = X1 + dN * res*5.0
    #     face_contour_slice = vtkfilters.sliceByPlane(face_contour, X2, normal)
    #     if face_contour_slice.GetNumberOfPoints() > 1000:
    #         MADE_FACE_CONTOUR = True
    #         convexHull = vtkfilters.getPtsAsNumpy(vtkfilters.shrinkWrapData(face_contour_slice))
    #         # pts = vtkfilters.getPtsAsNumpy(face_contour_slice)
    #         # convexHull = vtkfilters.ftk.buildConvexHUll3D(pts, nPts=2000, TO_SPLINE=False)
    #         convexHull2 = convexHull.copy()
    #         convexHull2[:, N] = X
    #         pp = vtkfilters.buildPolydataFromXYZ(np.concatenate([convexHull, convexHull2], axis=0))
    #         ppW = vtkfilters.shrinkWrapData(pp)
    #         vtkfilters.filterMaskImageBySurface(imageData, ppW, fill_value=airThreshold*2.0, arrayName="LabelMap")
    # if MADE_FACE_CONTOUR:
    #     ALM = vtkfilters.getArrayAsNumpy(imageData, "LabelMap", RETURN_3D=True)
    #     AOrig = vtkfilters.getArrayAsNumpy(imageData, arrayName, RETURN_3D=True)
    #     AOrig[ALM > 1] = airThreshold*5.0
    #     vtkfilters.setArrayFromNumpy(imageData, AOrig, arrayName, IS_3D=True, SET_SCALAR=True)
    # face_contour_closed = vtkfilters.contourFilter(imageData, airThreshold)
    # face_contour_closed = vtkfilters.getConnectedRegionLargest(face_contour_closed)
    return face_contour_closed



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
                        maxIterations: int = 50, shrinkFactor: int | list[int] | None = None, maskImage: vtk.vtkImageData | None = None,
                        arrayNameOut: str | None = None) -> vtk.vtkImageData:
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
    if arrayNameOut is None:
        arrayNameOut = arrayName
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
    if maskImage is not None:
        output = corrector.Execute(image, maskImage)
    else:
        output = corrector.Execute(image)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    output = inputImage / sitk.Exp( log_bias_field )
    return sitkHelper_ITKToVTKImage(output, arrayName=arrayNameOut)
