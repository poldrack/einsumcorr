"""Command-line interface for einsumcorr."""

import argparse
import sys
import numpy as np
from .optcorr import optcorr


def main():
    """Main entry point for the einsumcorr CLI."""
    parser = argparse.ArgumentParser(
        description="Compute columnwise correlations using Einstein summation notation."
    )
    
    parser.add_argument(
        "input1",
        nargs="?",
        help="First input CSV file containing matrix data"
    )
    
    parser.add_argument(
        "input2",
        nargs="?",
        help="Optional second input CSV file for cross-correlation"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file to save correlation matrix (CSV format)"
    )
    
    parser.add_argument(
        "--delimiter", "-d",
        default=",",
        help="Delimiter for CSV files (default: comma)"
    )
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if not args.input1:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load first matrix
        x = np.loadtxt(args.input1, delimiter=args.delimiter)
        
        # Load second matrix if provided
        y = None
        if args.input2:
            y = np.loadtxt(args.input2, delimiter=args.delimiter)
        
        # Compute correlation
        result = optcorr(x, y)
        
        # Output results
        if args.output:
            np.savetxt(args.output, result, delimiter=args.delimiter, fmt="%.6f")
            print(f"Correlation matrix saved to {args.output}")
        else:
            # Print to stdout
            np.set_printoptions(precision=6, suppress=True)
            print("Correlation matrix:")
            print(result)
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)