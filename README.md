READ sample1 first before going through kmeans file. 

# Parallel K-Means Distance & Clustering (C++ / OpenMP)

This project implements a simple **parallel K-means-style assignment step** in C++ using **OpenMP**.

Given:
- an input CSV file with **N data points**
- each point having 8 numerical features
- a requested number of clusters **K** to be passed through command.

the program:

1. Loads all data points from a CSV file.
2. Randomly picks **K** points as initial centroids.
3. Computes the **Euclidean distance** from every point to every centroid **in parallel**.
4. Assigns each point to its **closest centroid**.
5. Prints a summary of:
   - the first 100 clustered points (with their distances to all centroids),
   - timing information for each phase.
6. Writes the **full clustering result** to `clustering_results.csv`.

> ⚠️ This is **only the assignment step** (no iterative centroid update). It’s suitable for demonstrating parallelism and measuring performance, not for full K-means convergence.

---

## Features

- Uses **OpenMP** for parallel distance computation.
- Reads data from a CSV file with **8 float columns**.
- Random initialization of centroids from actual data points.
- Prints:
  - initial random centroids,
  - first 100 clustered points,
  - detailed timing summary.
- Outputs all results to `clustering_results.csv`.

---

## Usage
./kmeans_cluster <input_file> <number_clusters>

## Example 
./kmeans_cluster generated_dataset.csv 5


```bash
g++ -O2 -fopenmp -o kmeans_cluster main.cpp
