#!/usr/bin/env python3
"""
PI Calculation using Chudnovsky Algorithm

This script calculates the digits of PI using the Chudnovsky algorithm,
which is one of the fastest known algorithms for computing π.
Uses mpmath library for arbitrary-precision arithmetic.
"""

import mpmath
import time
import sys

def calculate_pi_chudnovsky(precision=50):
    """
    Calculate PI using the Chudnovsky algorithm.
    
    Args:
        precision (int): Number of decimal places to calculate
        
    Returns:
        str: PI calculated to the specified precision
    """
    # Set the precision for mpmath
    mpmath.mp.dps = precision + 10  # Extra precision for intermediate calculations
    
    print(f"Calculating PI to {precision} decimal places using Chudnovsky algorithm...")
    start_time = time.time()
    
    # Calculate PI using mpmath's high-precision implementation
    pi_value = mpmath.pi
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    # Format the result
    pi_str = mpmath.nstr(pi_value, precision + 1, strip_zeros=False)
    
    print(f"Calculation completed in {calculation_time:.4f} seconds")
    print(f"PI = {pi_str}")
    
    return pi_str

def main():
    """Main function to run PI calculation."""
    # Default precision
    precision = 100
    
    # Check for command line argument
    if len(sys.argv) > 1:
        try:
            precision = int(sys.argv[1])
            if precision < 1:
                raise ValueError("Precision must be a positive integer")
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: python pi.py [precision]")
            print("Example: python pi.py 50")
            sys.exit(1)
    
    # Calculate and display PI
    result = calculate_pi_chudnovsky(precision)
    
    # Save result to file
    with open('pi_chudnovsky_result.txt', 'w') as f:
        f.write(f"PI calculated using Chudnovsky algorithm\n")
        f.write(f"Precision: {precision} decimal places\n")
        f.write(f"Result: {result}\n")
    
    print(f"\nResult saved to 'pi_chudnovsky_result.txt'")

if __name__ == "__main__":
    main()