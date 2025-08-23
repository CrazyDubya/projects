#!/usr/bin/env python3
"""
PI Calculation using Gauss-Legendre Algorithm

This script implements the Gauss-Legendre algorithm to compute the digits of PI.
Known for its rapid quadratic convergence - doubles the number of correct digits
at each iteration.
"""

import mpmath
import time
import sys

def calculate_pi_gauss_legendre(precision=50, max_iterations=10):
    """
    Calculate PI using the Gauss-Legendre algorithm.
    
    Args:
        precision (int): Number of decimal places to calculate
        max_iterations (int): Maximum number of iterations
        
    Returns:
        str: PI calculated to the specified precision
    """
    # Set the precision for mpmath
    mpmath.mp.dps = precision + 20  # Extra precision for intermediate calculations
    
    print(f"Calculating PI to {precision} decimal places using Gauss-Legendre algorithm...")
    start_time = time.time()
    
    # Initialize variables
    a = mpmath.mpf(1)
    b = mpmath.mpf(1) / mpmath.sqrt(2)
    t = mpmath.mpf(1) / mpmath.mpf(4)
    p = mpmath.mpf(1)
    
    print(f"Starting iterations (max: {max_iterations}):")
    
    for i in range(max_iterations):
        # Store previous values
        a_prev = a
        
        # Update values according to Gauss-Legendre algorithm
        a = (a + b) / 2
        b = mpmath.sqrt(a_prev * b)
        t = t - p * (a_prev - a) ** 2
        p = 2 * p
        
        # Calculate current approximation of PI
        pi_approx = (a + b) ** 2 / (4 * t)
        
        print(f"Iteration {i+1}: PI ≈ {mpmath.nstr(pi_approx, min(20, precision + 1))}")
        
        # Check for convergence (when change is smaller than desired precision)
        if i > 0:
            if abs(pi_approx - prev_pi) < mpmath.mpf(10) ** (-(precision + 5)):
                print(f"Converged after {i+1} iterations")
                break
        
        prev_pi = pi_approx
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    # Final calculation
    pi_final = (a + b) ** 2 / (4 * t)
    pi_str = mpmath.nstr(pi_final, precision + 1, strip_zeros=False)
    
    print(f"\nCalculation completed in {calculation_time:.4f} seconds")
    print(f"PI = {pi_str}")
    
    return pi_str

def main():
    """Main function to run PI calculation."""
    # Default values
    precision = 100
    max_iterations = 10
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        try:
            precision = int(sys.argv[1])
            if precision < 1:
                raise ValueError("Precision must be a positive integer")
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: python pi2.py [precision] [max_iterations]")
            print("Example: python pi2.py 50 8")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            max_iterations = int(sys.argv[2])
            if max_iterations < 1:
                raise ValueError("Max iterations must be a positive integer")
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: python pi2.py [precision] [max_iterations]")
            sys.exit(1)
    
    # Calculate and display PI
    result = calculate_pi_gauss_legendre(precision, max_iterations)
    
    # Save result to file
    with open('pi_gauss_legendre_result.txt', 'w') as f:
        f.write(f"PI calculated using Gauss-Legendre algorithm\n")
        f.write(f"Precision: {precision} decimal places\n")
        f.write(f"Max iterations: {max_iterations}\n")
        f.write(f"Result: {result}\n")
    
    print(f"\nResult saved to 'pi_gauss_legendre_result.txt'")

if __name__ == "__main__":
    main()