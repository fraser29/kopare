import numpy as np
from kopare.sinus_detection import segment_sinus_and_airways
from kopare.sinus_detection import grid_search_algorithms
from kopare.sinus_detection import benchmark_algorithms

from kopare.kopare import kopare_main, load_parameters, DEFAULT_PARAMETER_FILE
from kopare import kopare_utils
from ngawari import fIO, vtkfilters
import os
import matplotlib.pyplot as plt
from spydcmtk import spydcm

PixelData = "PixelData"
LabelMap = "LabelMap"

ROOT_DIR = "/Volume/WORK/KISPI/KMRB/"

def read_subject_data(N):
    internal_air_file = f"{ROOT_DIR}//BB{N:06d}/PROCESSED/internal-air.vti"
    image_data = fIO.readVTKFile(f"{ROOT_DIR}/BB{N:06d}/PROCESSED/imageData_original.vti")
    mask_data = fIO.readVTKFile(f"{ROOT_DIR}//BB{N:06d}/PROCESSED/imageData_mask.vti")
    if not os.path.isfile(internal_air_file):
        manual_data_s = fIO.readVTKFile(f"{ROOT_DIR}//BB{N:06d}/PROCESSED/sinus.nrrd")
        manual_data_a = fIO.readVTKFile(f"{ROOT_DIR}//BB{N:06d}/PROCESSED/airway.nrrd")
        int_air_image = vtkfilters.duplicateImageData(image_data)
        aName = vtkfilters.getArrayNames(manual_data_s)[0]
        AS = vtkfilters.getArrayAsNumpy(manual_data_s, aName, RETURN_3D=True)
        AA = vtkfilters.getArrayAsNumpy(manual_data_a, aName, RETURN_3D=True)
        AA[AS>0] = AS[AS>0]
        AA = np.flip(AA, axis=2)
        vtkfilters.setArrayFromNumpy(int_air_image, AA, LabelMap, SET_SCALAR=True, IS_3D=True)
        fOut = fIO.writeVTKFile(int_air_image, internal_air_file)
        print(f"Written {fOut}")
    else:
        int_air_image = fIO.readVTKFile(internal_air_file)
        AA = vtkfilters.getArrayAsNumpy(int_air_image, LabelMap, RETURN_3D=True)
    imageA = vtkfilters.getArrayAsNumpy(image_data, PixelData, RETURN_3D=True)
    maskA = vtkfilters.getArrayAsNumpy(mask_data, LabelMap, RETURN_3D=True)
    return image_data, imageA, maskA, AA

def run_test_A(N):
    image_data, imageA, maskA, AA = read_subject_data(N)

    # image, body_mask are numpy arrays with same shape
    pred = segment_sinus_and_airways(
        imageA,
        maskA,
        method="persistent_dark_after_smoothing",
        sigma_large=7.5,
        min_region_size=300,
        hole_size=400,
    )

    dims = imageA.shape
    print(dims)
    nY = 100

    # fig, axs = plt.subplots(2,2)
    # axs[0][0].imshow(imageA[:,nY,:])
    # axs[1][0].imshow(maskA[:,nY,:])
    # axs[0][1].imshow(pred[:,nY,:])
    # axs[1][1].imshow(AA[:,nY,:])
    # plt.show()

    image_out = vtkfilters.duplicateImageData(image_data)
    vtkfilters.setArrayFromNumpy(image_out, pred.astype(np.int16), PixelData, IS_3D=True, SET_SCALAR=True)
    fOut = fIO.writeVTKFile(image_out, f"{ROOT_DIR}/BB{N:06d}/PROCESSED/imageData_mask_air-internal_PDAS.vti")
    print(f"Written {fOut}")

def run_test_B(N):
    image_data, imageA, maskA, AA = read_subject_data(N)

    # image, body_mask are numpy arrays with same shape
    pred = segment_sinus_and_airways(
        imageA,
        maskA,
        method="thick_region_filter",
        opening_radius=3,
        regrow_radius=0,
        min_region_size=400,
    )

    dims = imageA.shape
    print(dims)
    nY = 100

    # fig, axs = plt.subplots(2,2)
    # axs[0][0].imshow(imageA[:,nY,:])
    # axs[1][0].imshow(maskA[:,nY,:])
    # axs[0][1].imshow(pred[:,nY,:])
    # axs[1][1].imshow(AA[:,nY,:])
    # plt.show()

    image_out = vtkfilters.duplicateImageData(image_data)
    vtkfilters.setArrayFromNumpy(image_out, pred.astype(np.int16), PixelData, IS_3D=True, SET_SCALAR=True)
    fOut = fIO.writeVTKFile(image_out, f"{ROOT_DIR}/BB{N:06d}/PROCESSED/imageData_mask_air-internal_TRF.vti")
    print(f"Written {fOut}")

def build_cases(SNList):
    cases = []
    for sn in SNList:
        _, imageA, maskA, AA = read_subject_data(sn)
        cases.append({
            "image": imageA, "body_mask": maskA, "manual_mask": AA, "case_id": sn
        })
    return cases


def benchmark():
    cases = build_cases(range(3,16))
    results = benchmark_algorithms(cases)
    print(results["persistent_dark_after_smoothing"]["summary"])


def runOptimisation():
    
    cases = build_cases(range(3,16))
    method_param_grids = {
        "threshold_external_reference": {
            "min_region_size": [100, 200, 400],
            "hole_size": [100, 300],
            "closing_radius": [0, 1, 2],
        },
        "thick_region_filter": {
            "opening_radius": [1, 2, 3],
            "regrow_radius": [0, 1, 2],
            "min_region_size": [100, 200, 400],
        },
        "persistent_dark_after_smoothing": {
            "sigma_large": [1.5, 2.0, 2.5, 3.0],
            "min_region_size": [100, 200, 400],
            "hole_size": [150, 300, 600],
        },
    }

    all_search = grid_search_algorithms(
        cases,
        method_param_grids=method_param_grids,
        objective="dice_mean",
    )

    print("Best method:", all_search["best_overall"]["method"])
    print("Best params:", all_search["best_overall"]["best_params"])
    print("Best Dice mean:", all_search["best_overall"]["best_score"])


def test_BC():
    iif = "{ROOT_DIR}/BB000004/PROCESSED/imageData_original.vti"
    ii = fIO.readVTKFile(iif)
    iiBC = kopare_utils.biasFieldCorrection(ii, PixelData, numberFittingLevels=6, maxIterations=100, shrinkFactor=3)
    fOut = fIO.writeVTKFile(iiBC, iif[:-4]+"_BC.vti")
    print(fOut)

def test_inversion():
    dcmDir = "/Volume/WORK/KISPI/KMRB/BB000001/RAW/DICOM/Ortho_7_2d20b7ca-14e4-43_20240418/SE18_3d_overview_RR"
    outDir = "/Volume/WORK/KISPI/KMRB/BB000001/PROCESSED"
    dcmSeries = spydcm.dcmTK.DicomSeries.setFromDirectory(dcmDir)
    vtiDict = dcmSeries.buildVTIDict()
    ii = list(vtiDict.values())[0]
    fOut = fIO.writeVTKFile(ii, os.path.join(outDir, "imageData_original.vti"))
    print(fOut)
    iiInv = kopare_utils.signalLogInverse(ii, PixelData)

    # AI = contrastStretch_percentile(AI)
    fOut = fIO.writeVTKFile(iiInv, os.path.join(outDir, "imageData_original_Inv.vti"))
    print(fOut)


def test_wrap():
    iif = f"{ROOT_DIR}/BB000001/PROCESSED/imageData_smooth.vti"
    ii = fIO.readVTKFile(iif)
    face_contour = vtkfilters.contourFilter(ii, 100.0)
    face_contour = vtkfilters.getConnectedRegionLargest(face_contour)
    sw = vtkfilters.shrinkWrapData(face_contour)

    # Detect and label any planar cut faces on both the shrinkwrap and the
    # original isosurface contour.
    for tag, mesh in [("sw", sw)]:
        labeled, is_cut, cuts = kopare_utils.detect_and_label_cut_faces(mesh)
        print(f"[{tag}] is_cut={is_cut}  n_cuts_found={len(cuts)}")
        for c in cuts:
            print(f"  cut {c['label']}: flat_fraction={c['flat_area_fraction']:.3f}"
                  f"  n_cells={c['n_cut_cells']}"
                  f"  normal={np.round(c['plane_normal'], 3)}")
        out_path = iif[:-4] + f"_{tag}_cut_labeled.vtp"
        fOut = fIO.writeVTKFile(labeled, out_path)
        print(f"  Written: {fOut}")



if __name__ == "__main__":
    # run_test(3)
    # run_test(4)
    # benchmark()
    # runOptimisation()
    # run_test_A(3)
    # run_test_B(3)
    # test_BC()
    # test_inversion()
    test_wrap()