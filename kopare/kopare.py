"""Command-line entry point and processing scaffold for kopare."""

from __future__ import annotations

import argparse
import logging
from ngawari import fIO
from ngawari import vtkfilters
from spydcmtk import spydcm
import sys
from pathlib import Path
from typing import Any
import numpy as np

from kopare.kopare_masking import mask_external_air


THIS_DIR = Path(__file__).parent
DEFAULT_PARAMETER_FILE = THIS_DIR / "kopare_parameters.json"
LOGGER_NAME = "kopare"


def configure_logging(verbose: bool = False) -> None:
    """Configure terminal logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class kopare_main:
    """Main processing class for a kopare run."""

    def __init__(
                    self,
                    input_directory: Path,
                    output_dir: Path,
                    parameters: dict[str, Any],
                ) -> None:
        ##
        self.input_directory = input_directory
        self.output_dir = output_dir
        self.parameters = parameters
        self.logger = logging.getLogger(LOGGER_NAME)
        self.VERBOSE = logging.getLogger().level == logging.DEBUG


    def run(self) -> int:
        """Run the processing pipeline."""
        self.logger.info("Starting processing")
        self.logger.info("Input directory: %s", self.input_directory)
        self.logger.info("Output directory: %s", self.output_dir)
        self.logger.info("Loaded %d parameter entries", len(self.parameters))

        if not self.input_directory:
            self.logger.error("No input files found to process.")
            return 1

        # TODO: Process the input directory
        try:
            dcmSeries = spydcm.dcmTK.DicomSeries.setFromDirectory(self.input_directory, HIDE_PROGRESSBAR=not self.VERBOSE)
        except ValueError:
            raise ValueError(f"ERROR: More than one DICOM series found in input directory: {self.input_directory}")

        self.logger.info(f"Processing directory {self.input_directory}. Found {len(dcmSeries)} DICOMs")

        vtiDict = dcmSeries.buildVTIDict()
        if len(vtiDict) != 1:
            raise ValueError(f"ERROR: More than one volume data found in input directory: {self.input_directory}")
        imageData = list(vtiDict.values())[0]
        if len(dcmSeries) == 1:
            # FIXME: this is an ugly work around - to fix in spydcmtk
            A3 = vtkfilters.getArrayAsNumpy(imageData, "PixelData", RETURN_3D=True)
            self.logger.info(f"THIS WILL NOT WORK - FIXME {A3.shape}")

        fOut = fIO.writeVTKFile(imageData, self.output_dir / "imageData_original.mha")
        self.logger.info(f"Wrote original image to {fOut}")

        medFilterSize = self.parameters["Median_filter_size"]
        mask, image_med, threshold = mask_external_air(imageData, medFilterSize, "PixelData")
        self.logger.info(f"External air threshold: {threshold}")
        vtkfilters.setArrayFromNumpy(imageData, mask, "Labels", IS_3D=True, SET_SCALAR=True)

        # A_masked = vtkfilters.getArrayAsNumpy(imageData, "PixelData", RETURN_3D=True) * ~mask
        # vtkfilters.setArrayFromNumpy(imageData, A_masked, "PixelData", IS_3D=True, SET_SCALAR=True)
        fOut = fIO.writeVTKFile(image_med, self.output_dir / f"imageData_med-filter_{medFilterSize}.mha")
        self.logger.info(f"Wrote med-filtered image to {fOut}")
        fOut = fIO.writeVTKFile(imageData, self.output_dir / "imageData_masked.mha")
        self.logger.info(f"Wrote masked image to {fOut}")

        return 0

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
    logger = logging.getLogger(LOGGER_NAME)

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
        parameters=parameters,
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
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input DICOM files.",
    )
    parser.add_argument(
        "-p",
        "--parameter-file",
        type=Path,
        default=DEFAULT_PARAMETER_FILE,
        help=f"Path to JSON parameter file (default: {DEFAULT_PARAMETER_FILE}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to '<input_dir>_output'.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging.",
    )
    return parser.parse_args(argv)


# =========================================================================================
# Main Execution
# =========================================================================================
if __name__ == "__main__":
    sys.exit(main())
