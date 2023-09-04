import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pyximport;

pyximport.install()


# Define matrix multiplication function (optimized, possibly in Cython)
def multiply_matrices(a, b):
    return np.dot(a, b)


# List of matrix sizes (N x N) for benchmarking
matrix_sizes = [1000, 2000, 3000, 4000, 5000]
runtimes = []


# Measure the runtime for each matrix size using multiprocessing
def measure_runtime(size):
    matrix_a = np.random.rand(size, size)
    matrix_b = np.random.rand(size, size)

    start_time = time.time()  # Start the timer
    result = multiply_matrices(matrix_a, matrix_b)  # Perform matrix multiplication
    end_time = time.time()  # Stop the timer

    execution_time = end_time - start_time
    return size, execution_time


if __name__ == "__main__":
    # Use a Pool for parallel processing
    with Pool() as pool:
        runtimes = pool.map(measure_runtime, matrix_sizes)

    # Extract results
    sizes, execution_times = zip(*runtimes)

    # Create a Matplotlib plot to visualize the data
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, execution_times, marker='o', linestyle='-')
    plt.title('Matrix Multiplication Runtime vs. Matrix Size')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)

    # Display the plot (you can also save it as an image)
    plt.show()

