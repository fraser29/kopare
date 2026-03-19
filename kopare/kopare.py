"""Command-line entry point and processing scaffold for kopare."""

from __future__ import annotations

import argparse
import logging
from ngawari import fIO
from ngawari import vtkfilters
import vtk
from spydcmtk import spydcm
import sys
from pathlib import Path
from typing import Any
import numpy as np

from kopare import kopare_utils
from kopare.sinus_detection import segment_sinus_and_airways

# =========================================================================================
# Constants and helper functions
# =========================================================================================
THIS_DIR = Path(__file__).parent
DEFAULT_PARAMETER_FILE = THIS_DIR / "kopare_parameters.json"
DEFAULT_LOGGER_NAME = "kopare" # DO NOT CHANGE
PERMITTED_OUTPUT_FORMATS = [".mha", ".vti", ".nii", ".nii.gz"]
PixelData = "PixelData"
LabelMap = "LabelMap"

def configure_logging(verbose: bool = False) -> None:
    """Configure terminal logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# =========================================================================================
# Main Processing Class
# =========================================================================================
class kopare_main:
    """Main processing class for a kopare run."""

    def __init__(
                    self,
                    input_directory: Path,
                    output_dir: Path,
                    parameters: dict[str, Any],
                    quiet: bool = False,
                    verbose: bool = False,
                ) -> None:
        ##
        self.input_directory = input_directory
        self.output_dir = output_dir
        self.parameters = parameters
        self.verbose = verbose
        self.logger = logging.getLogger(DEFAULT_LOGGER_NAME)
        if quiet:
            self.logger.setLevel(logging.ERROR)
        elif verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self.write_intermediate_files = self.parameters["Write_intermediate_files"]
        self.dcmSeries = None
        self.imageData_original = None


    # ----------------------------------------------------------------------------------------
    # Input processing
    # ----------------------------------------------------------------------------------------
    def _process_input_directory(self) -> None:
        """Process the input directory."""
        if not self.input_directory:
            self.logger.error("No input files found to process.")
            return None
        self.logger.info("Processing input directory: %s", self.input_directory)
        try:
            self.dcmSeries = spydcm.dcmTK.DicomSeries.setFromDirectory(self.input_directory, HIDE_PROGRESSBAR=not self.verbose)
        except ValueError:
            raise ValueError(f"ERROR: More than one DICOM series found in input directory: {self.input_directory}")
        self.logger.info("Found %d DICOMs", len(self.dcmSeries))
        vtiDict = self.dcmSeries.buildVTIDict(TRUE_ORIENTATION=self.parameters.get("Write_true_orientation_image_data", False))
        if len(vtiDict) != 1:
            raise ValueError(f"ERROR: More than one volume data found in input directory: {self.input_directory}")
        ii = list(vtiDict.values())[0]
        scale = self.parameters.get("Scale", 1.0)
        self.imageData_original = kopare_utils.scaleImageData(ii, scaleFactor=scale)
        self.latest_imageData = self.imageData_original

    # ----------------------------------------------------------------------------------------
    # Main processing
    # ----------------------------------------------------------------------------------------
    def run(self) -> int:
        """Run the processing pipeline."""
        self.logger.info("Starting processing")
        self.logger.info("Input directory: %s", self.input_directory)
        self.logger.info("Output directory: %s", self.output_dir)
        self.logger.info("Loaded %d parameter entries", len(self.parameters))

        self._process_input_directory()

        # Set up parameters
        medFilterSize = self.parameters["Median_filter_size"]

        self._write_intermediate_files(self.imageData_original, f"imageData_original")

        ## DATA PREPARATION STEPS
        self._bias_field_correction()
        A_for_inverse = vtkfilters.getArrayAsNumpy(self.latest_imageData, PixelData, RETURN_3D=True)
        self._denoise()
        self._median_filter()
        self._gaussian_smooth()

        ## THIS IS THE MAIN FUNCTION CALL - BIAS FIELD CORRECTION, MEDIAN FILTERING, AND EXTERNAL AIR MASKING
        image_mask_external = kopare_utils.mask_external_air(imageData=self.latest_imageData, 
                                                        arrayName=PixelData, 
                                                        n_shrink_wrap_iterations=self.parameters.get("n_shrink_wrap_iterations", 5))
        # self._write_intermediate_files(face_contour, f"contour_external_air")
        self._write_intermediate_files(image_mask_external, f"mask_external_air")

        ## SINUS / AIRWAYS
        Amask_external = vtkfilters.getArrayAsNumpy(image_mask_external, LabelMap, RETURN_3D=True)
        sinus_and_airways_mask = self._sinus_and_airways_detection(Amask_external)


        ## INVERSION AND MASKING
        A_inverse = kopare_utils.signalLogInverse(A_for_inverse)
        self._write_intermediate_files(A_inverse, f"imageData_inv")
        AI_C = kopare_utils.contrastStretch_percentile(A_inverse)
        self._write_intermediate_files(AI_C, f"imageData_inv_contrast_stretched")
        AI_C[Amask_external<0.5] = 0.0
        AI_C[sinus_and_airways_mask<0.5] = 0.0
        self._write_intermediate_files(AI_C, f"imageData_inv_masked")

        # Smooth edges at mask boundaries
        A_masked_smoothed = kopare_utils.smooth_at_mask_edge(AI_C, 
                                                                 Amask_external, 
                                                                 n_iterations=self.parameters["EdgeSmoothing_nIterations"])
        self._write_intermediate_files(A_masked_smoothed, f"imageData_masked_smoothed")
        A_masked_smoothed[Amask_external<0.5] = 0.0
        self._write_intermediate_files(A_masked_smoothed, f"imageData_pseudoCT")

        ## WRITE PSEUDOCT DICOMS
        image_masked = vtkfilters.duplicateImageData(self.imageData_original)
        vtkfilters.setArrayFromNumpy(image_masked, A_masked_smoothed, PixelData, IS_3D=True, SET_SCALAR=True)
        self.dcmSeries.sortBySlice_InstanceNumber()
        tagUpdateDict={"SeriesNumber": int(self.dcmSeries.getTag("SeriesNumber"))*1000, "SeriesDescription": [0x0008103e, "LO", "Kopare PseudoCT"]}
        scale = self.parameters.get("Scale", 1.0)
        image_masked_s = kopare_utils.scaleImageData(image_masked, scaleFactor=1.0/scale)
        dcmDirOut = spydcm.dcmTK.writeVTIToDicoms(image_masked_s, 
                                                    self.dcmSeries[0], 
                                                    outputDir=self.output_dir, 
                                                    tagUpdateDict=tagUpdateDict)
        self.logger.info(f"Written {dcmDirOut}")

        return 0


    def _denoise(self):
        if self.parameters["Denoising_alpha"] > 0:
            A = vtkfilters.getArrayAsNumpy(self.latest_imageData, PixelData, RETURN_3D=True)
            A_DN = kopare_utils.denoise3DA(A, self.parameters["Denoising_alpha"], self.parameters["Denoising_patch_size"], self.parameters["Denoising_patch_distance"])
            vtkfilters.setArrayFromNumpy(self.latest_imageData, A_DN, PixelData, IS_3D=True, SET_SCALAR=True)
            if self.write_intermediate_files:
                self._write_intermediate_files(self.latest_imageData, f"imageData_denoised")
        else:
            self.logger.warning("Denoising not performed as Denoising_alpha is 0")


    def _gaussian_smooth(self):
        if self.parameters["Gaussian_smoothing_sigma"] > 0:
            imageData_smoothed = kopare_utils.gaussianSmooth(self.latest_imageData, self.parameters["Gaussian_smoothing_sigma"], self.parameters["Gaussian_smoothing_radius_factor"])
            if self.write_intermediate_files:
                self._write_intermediate_files(imageData_smoothed, f"imageData_smoothed")
            self.latest_imageData = imageData_smoothed
        else:
            self.logger.warning("Gaussian smoothing not performed as Gaussian_smoothing_sigma is 0")


    def _median_filter(self):
        if self.parameters["Median_filter_size"] > 0:
            imageData_median = vtkfilters.filterVtiMedian(vtiObj=self.latest_imageData, filterKernalSize=self.parameters["Median_filter_size"])
            if self.write_intermediate_files:
                self._write_intermediate_files(imageData_median, f"imageData_median")
            self.latest_imageData = imageData_median
        else:
            self.logger.warning("Median filtering not performed as Median_filter_size is 0")


    def _bias_field_correction(self, inputData=None):
        if inputData is None:
            inputData = self.latest_imageData
        if self.parameters["BC_Number_of_fitting_levels"] > 0:
            imageData_BC = kopare_utils.biasFieldCorrection(inputData, PixelData, self.parameters["BC_Number_of_fitting_levels"], self.parameters["BC_Maximum_number_of_iterations"], self.parameters["BC_Shrink_factor"])
            if self.write_intermediate_files:
                self._write_intermediate_files(imageData_BC, f"imageData_BC")
            self.latest_imageData = imageData_BC
        else:
            self.logger.warning("Bias field correction not performed as BC_Number_of_fitting_levels is 0")

    def _sinus_and_airways_detection(self, Amask_external):        
        sinus_and_airways_mask = np.ones(Amask_external.shape) # Default in case do not run
        if self.parameters["Sinus_detection_method"] and self.parameters["Sinus_airway_parameters"][self.parameters["Sinus_detection_method"]]:
            Aoriginal = vtkfilters.getArrayAsNumpy(self.latest_imageData, PixelData, RETURN_3D=True)
            sinus_and_airways_mask = segment_sinus_and_airways(Aoriginal,
                                                            Amask_external, 
                                                            method=self.parameters["Sinus_detection_method"], 
                                                            **self.parameters["Sinus_airway_parameters"][self.parameters["Sinus_detection_method"]])
            
            image_sinus_airways_mask = vtkfilters.duplicateImageData(self.imageData_original)
            vtkfilters.setArrayFromNumpy(image_sinus_airways_mask, sinus_and_airways_mask.astype(np.int16), LabelMap, IS_3D=True, SET_SCALAR=True)
            self._write_intermediate_files(image_sinus_airways_mask, f"mask_sinus_and_airways")
        else:
            self.logger.warning(f"Sinus and airways detection not performed. Detection method unknown: {self.parameters['Sinus_detection_method']}")
        return sinus_and_airways_mask


    def _write_intermediate_files(self, imageData: vtk.vtkImageData | np.ndarray, output_prefix: str) -> None:
        """Write intermediate files to output directory."""
        if not self.write_intermediate_files:
            return
        if isinstance(imageData, np.ndarray):
            imageDataD = vtkfilters.duplicateImageData(self.latest_imageData)
            vtkfilters.setArrayFromNumpy(imageDataD, imageData, PixelData, IS_3D=True, SET_SCALAR=True)
            imageData = imageDataD
        output_format = self.parameters["Output_format"].lower()
        if not output_format.startswith("."):
            output_format = "." + output_format
        if not output_format in PERMITTED_OUTPUT_FORMATS:
            raise ValueError(f"ERROR: Invalid output format: {output_format}. Permitted formats: {PERMITTED_OUTPUT_FORMATS}")
        if vtkfilters.isVTP(imageData):
            fOut = fIO.writeVTKFile(imageData, self.output_dir / f"{output_prefix}.stl")
            self.logger.info(f"Wrote {output_prefix} STL file to {fOut}")
            return
        fOut = fIO.writeVTKFile(imageData, self.output_dir / f"{output_prefix}{output_format}")
        self.logger.info(f"Wrote {output_prefix}{output_format} file to {fOut}")

# =========================================================================================
# =========================================================================================


# =========================================================================================
# Input Validation
# =========================================================================================
def validate_input_dir(input_dir: Path) -> None:
    """Validate that the input directory exists and is a directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")


def load_parameters(parameter_file: Path) -> dict[str, Any]:
    """Load and validate JSON parameters."""
    if not parameter_file.exists():
        raise FileNotFoundError(f"Parameter file not found: {parameter_file}")
    if not parameter_file.is_file():
        raise IsADirectoryError(f"Parameter path is not a file: {parameter_file}")

    return fIO.parseJsonToDictionary(parameter_file)


def resolve_output_dir(
    output_dir_arg: Path | None,
    input_dir: Path | None = None,
) -> Path:
    """Resolve output directory from CLI argument or default naming."""
    if output_dir_arg:
        return output_dir_arg.expanduser().resolve()
    if input_dir is not None:
        return input_dir.parent 
    raise ValueError("No output directory provided.")



# =========================================================================================
# =========================================================================================
def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)

    input_directory = args.input_dir.expanduser().resolve()
    parameter_file = args.parameter_file.expanduser().resolve()

    try:
        validate_input_dir(input_directory)
        output_dir = resolve_output_dir(args.output_dir, input_directory)
        parameters = load_parameters(parameter_file)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("ERROR: %s", exc)
        return 1

    app = kopare_main(
        input_directory=input_directory,
        output_dir=output_dir,
        parameters=parameters
    )
    return app.run()


# =========================================================================================
# =========================================================================================

# =========================================================================================
# CLI Argument Parsing
# =========================================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="kopare",
        description=(
            "Process input DICOM files (in input directory) using "
            "parameters from a JSON file and write results to an output directory."
        ),
    )
    parser.add_argument("-i", "--input-dir", type=Path, required=True, help="Directory containing input DICOM files.")
    parser.add_argument("-p", "--parameter-file", type=Path, default=DEFAULT_PARAMETER_FILE, help=f"Path to JSON parameter file (default: {DEFAULT_PARAMETER_FILE}).")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Output directory. Defaults to '<input_dir>_output'.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (debug) logging.")
    return parser.parse_args(argv)


# =========================================================================================
# Main Execution
# =========================================================================================
if __name__ == "__main__":
    sys.exit(main())
