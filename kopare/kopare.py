"""Command-line entry point and processing scaffold for kopare."""

from __future__ import annotations

import argparse
import logging
from ngawari import fIO, isVTP
from ngawari import vtkfilters
import vtk
from spydcmtk import spydcm
import sys
from pathlib import Path
from typing import Any
import numpy as np

from kopare import kopare_utils

# =========================================================================================
# Constants and helper functions
# =========================================================================================
THIS_DIR = Path(__file__).parent
DEFAULT_PARAMETER_FILE = THIS_DIR / "kopare_parameters.json"
DEFAULT_LOGGER_NAME = "kopare"
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


        ## THIS IS THE MAIN FUNCTION CALL - BIAS FIELD CORRECTION, MEDIAN FILTERING, AND EXTERNAL AIR MASKING
        image_mask_external = kopare_utils.mask_external_air(imageData=self.imageData_original, 
                                                        median_filter_size=medFilterSize, 
                                                        arrayName=PixelData, 
                                                        numberFittingLevels=self.parameters["BC_Number_of_fitting_levels"], 
                                                        maxIterations=self.parameters["BC_Maximum_number_of_iterations"], 
                                                        shrinkFactor=self.parameters["BC_Shrink_factor"], 
                                                        denoising_alpha=self.parameters["Denoising_alpha"],
                                                        denoising_patch_size=self.parameters["Denoising_patch_size"],
                                                        denoising_patch_distance=self.parameters["Denoising_patch_distance"],
                                                        n_shrink_wrap_iterations=self.parameters["n_shrink_wrap_iterations"],
                                                        kopare_class=self)



        # image_mask = vtkfilters.duplicateImageData(self.imageData_original)
        # vtkfilters.setArrayFromNumpy(image_mask, mask3D_numpy, "Labels", IS_3D=True, SET_SCALAR=True)
        self._write_intermediate_files(image_mask_external, f"imageData_mask")


        A = vtkfilters.getArrayAsNumpy(self.imageData_original, PixelData)
        AME = vtkfilters.getArrayAsNumpy(image_mask_external, LabelMap)
        A[AME<0.5] = 0.0
        image_masked = vtkfilters.duplicateImageData(self.imageData_original)
        vtkfilters.setArrayFromNumpy(image_masked, A, PixelData, SET_SCALAR=True)
        self._write_intermediate_files(image_masked, f"imageData_masked")

        # # TODO: internal air mask, invert, smooth edges, write modified DICOMS 
        # image_masked_smoothed = kopare_utils.smooth_at_mask_edge(image_masked, 
        #                                                          image_mask_external, 
        #                                                          PixelData, 
        #                                                          LabelMap, 
        #                                                          n_iterations=self.parameters["EdgeSmoothing_nIterations"])
        # self._write_intermediate_files(image_masked_smoothed, f"imageData_masked_smoothed")

        return 0

    def _write_intermediate_files(self, imageData: vtk.vtkImageData, output_prefix: str) -> None:
        """Write intermediate files to output directory."""
        if not self.write_intermediate_files:
            return
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
