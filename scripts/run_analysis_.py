#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Import required modules
from src.analysis.data_processing import load_data

# Import the refactored analysis functions
from standard_analysis import run_standard_analysis
from loocv import run_loocv


def main():
    """Main function to control analysis workflow."""
    parser = argparse.ArgumentParser(description='Run cell lysis analysis')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to input data file (default: data/scFv/scfv_lysis.xlsx)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output directory (default: results/analysis)')
    parser.add_argument('--loocv', action='store_true',
                        help='Run Leave-One-Out Cross-Validation')
    parser.add_argument('--standard', action='store_true',
                        help='Run standard analysis on full dataset')
    args = parser.parse_args()

    # Set paths
    data_path = args.data
    if data_path is not None:
        data_path = Path(data_path)
    else:
        data_path = PROJECT_ROOT / "data" / "scFv" / "scfv_lysis.xlsx"

    output_path = args.output
    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = PROJECT_ROOT / "results" / "analysis"

    # Make sure directories exist
    (PROJECT_ROOT / "data" / "scFv").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default to running both analyses if none specified
    if not args.loocv and not args.standard:
        args.loocv = True
        args.standard = True

    start_time = time.time()
    print("=== Cell Lysis Analysis Pipeline ===")

    try:
        # Load data
        print(f"\nLoading data from {data_path}...")
        df_raw = load_data(data_path)
        print(f"Loaded {len(df_raw)} rows of data")

        # Run requested analyses
        if args.standard:
            print("\n=== Running Standard Analysis ===")
            standard_output = output_path / "standard"
            run_standard_analysis(df_raw, standard_output)

        if args.loocv:
            print("\n=== Running LOOCV Analysis ===")
            loocv_output = output_path / "loocv"
            run_loocv(df_raw, loocv_output)

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print(f"\n=== Analysis complete ({end_time - start_time:.2f} seconds) ===")


if __name__ == "__main__":
    main()