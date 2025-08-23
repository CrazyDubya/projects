#!/usr/bin/env python3
"""
PI Calculation using Multiple Algorithms with Parallel Processing

This script demonstrates the use of multiple algorithms and parallel processing
to calculate PI. It combines the Chudnovsky and Gauss-Legendre algorithms
and utilizes multiprocessing to enhance performance and efficiency.
"""

import mpmath
import time
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def chudnovsky_worker(precision):
    """Worker function for Chudnovsky algorithm calculation."""
    mpmath.mp.dps = precision + 10
    pi_value = mpmath.pi
    return ("Chudnovsky", mpmath.nstr(pi_value, precision + 1, strip_zeros=False), time.time())

def gauss_legendre_worker(args):
    """Worker function for Gauss-Legendre algorithm calculation."""
    precision, max_iterations = args
    mpmath.mp.dps = precision + 20
    
    # Gauss-Legendre algorithm implementation
    a = mpmath.mpf(1)
    b = mpmath.mpf(1) / mpmath.sqrt(2)
    t = mpmath.mpf(1) / mpmath.mpf(4)
    p = mpmath.mpf(1)
    
    for i in range(max_iterations):
        a_prev = a
        a = (a + b) / 2
        b = mpmath.sqrt(a_prev * b)
        t = t - p * (a_prev - a) ** 2
        p = 2 * p
        
        if i > 0:
            pi_approx = (a + b) ** 2 / (4 * t)
            if abs(pi_approx - prev_pi) < mpmath.mpf(10) ** (-(precision + 5)):
                break
        prev_pi = (a + b) ** 2 / (4 * t)
    
    pi_final = (a + b) ** 2 / (4 * t)
    return ("Gauss-Legendre", mpmath.nstr(pi_final, precision + 1, strip_zeros=False), time.time())

def monte_carlo_worker(args):
    """Worker function for Monte Carlo estimation (for demonstration)."""
    precision, num_samples = args
    import random
    
    inside_circle = 0
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    
    pi_estimate = 4 * inside_circle / num_samples
    return ("Monte Carlo", f"{pi_estimate:.{min(10, precision)}f}", time.time())

def calculate_pi_parallel(precision=50, max_iterations=8, monte_carlo_samples=1000000):
    """
    Calculate PI using multiple algorithms in parallel.
    
    Args:
        precision (int): Number of decimal places to calculate
        max_iterations (int): Max iterations for Gauss-Legendre
        monte_carlo_samples (int): Number of samples for Monte Carlo method
        
    Returns:
        dict: Results from all algorithms
    """
    print(f"Calculating PI to {precision} decimal places using multiple algorithms...")
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Using parallel processing with {min(3, mp.cpu_count())} workers\n")
    
    start_time = time.time()
    results = {}
    
    # Prepare tasks for parallel execution
    tasks = [
        ("Chudnovsky", chudnovsky_worker, precision),
        ("Gauss-Legendre", gauss_legendre_worker, (precision, max_iterations)),
        ("Monte Carlo", monte_carlo_worker, (precision, monte_carlo_samples))
    ]
    
    # Execute tasks in parallel
    with ProcessPoolExecutor(max_workers=min(3, mp.cpu_count())) as executor:
        # Submit all tasks
        future_to_algorithm = {}
        for alg_name, worker_func, args in tasks:
            future = executor.submit(worker_func, args)
            future_to_algorithm[future] = alg_name
        
        # Collect results as they complete
        for future in as_completed(future_to_algorithm):
            algorithm_name = future_to_algorithm[future]
            try:
                alg_name, result, end_time = future.result()
                calc_time = end_time - start_time
                results[alg_name] = {
                    'result': result,
                    'time': calc_time
                }
                print(f"✓ {alg_name} completed in {calc_time:.4f} seconds")
                print(f"  PI = {result[:50]}{'...' if len(result) > 50 else ''}\n")
            except Exception as e:
                print(f"✗ {algorithm_name} failed: {e}\n")
    
    total_time = time.time() - start_time
    print(f"All calculations completed in {total_time:.4f} seconds total")
    
    return results

def compare_results(results, precision):
    """Compare results from different algorithms."""
    if len(results) < 2:
        print("Need at least 2 results to compare")
        return
    
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    
    # Use the highest precision result as reference (typically Chudnovsky or Gauss-Legendre)
    reference = None
    ref_name = None
    
    # Find the most accurate reference (exclude Monte Carlo for high precision)
    for name, data in results.items():
        if name != "Monte Carlo":
            if reference is None or len(data['result']) > len(reference):
                reference = data['result']
                ref_name = name
    
    if reference is None:
        reference = list(results.values())[0]['result']
        ref_name = list(results.keys())[0]
    
    print(f"Using {ref_name} as reference")
    print(f"Reference: {reference}\n")
    
    for name, data in results.items():
        if name == ref_name:
            continue
        
        result = data['result']
        time_taken = data['time']
        
        # Count matching digits
        matching_digits = 0
        min_len = min(len(reference), len(result))
        
        for i in range(min_len):
            if reference[i] == result[i]:
                matching_digits += 1
            else:
                break
        
        print(f"{name}:")
        print(f"  Result: {result}")
        print(f"  Time: {time_taken:.4f} seconds")
        print(f"  Matching digits: {matching_digits}/{min_len}")
        print(f"  Accuracy: {matching_digits/min_len*100:.2f}%\n")

def main():
    """Main function to run parallel PI calculations."""
    # Default values
    precision = 50
    max_iterations = 8
    monte_carlo_samples = 1000000
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            precision = int(sys.argv[1])
            if precision < 1:
                raise ValueError("Precision must be a positive integer")
        except ValueError as e:
            print(f"Error: {e}")
            print("Usage: python py-multi.py [precision] [max_iterations] [monte_carlo_samples]")
            print("Example: python py-multi.py 100 8 5000000")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            max_iterations = int(sys.argv[2])
        except ValueError:
            print("Invalid max_iterations, using default: 8")
    
    if len(sys.argv) > 3:
        try:
            monte_carlo_samples = int(sys.argv[3])
        except ValueError:
            print("Invalid monte_carlo_samples, using default: 1000000")
    
    # Run parallel calculations
    results = calculate_pi_parallel(precision, max_iterations, monte_carlo_samples)
    
    # Compare results
    compare_results(results, precision)
    
    # Save comprehensive results to file
    with open('pi_multimethod_results.txt', 'w') as f:
        f.write("PI Multi-Method Calculation Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Precision: {precision} decimal places\n")
        f.write(f"Max iterations (Gauss-Legendre): {max_iterations}\n")
        f.write(f"Monte Carlo samples: {monte_carlo_samples}\n")
        f.write(f"CPU cores used: {min(3, mp.cpu_count())}\n\n")
        
        for name, data in results.items():
            f.write(f"{name} Algorithm:\n")
            f.write(f"  Result: {data['result']}\n")
            f.write(f"  Time: {data['time']:.4f} seconds\n\n")
    
    print(f"Comprehensive results saved to 'pi_multimethod_results.txt'")

if __name__ == "__main__":
    # Ensure multiprocessing works correctly on all platforms
    mp.set_start_method('spawn', force=True)
    main()