Parallel K-Means Clustering with Best-of-N Restarts (C++ / OpenMP)

This project implements a parallel K-means clustering algorithm in C++ using OpenMP, with:

Support for 8-dimensional data

Multiple random restarts (best of N runs)

A custom accuracy metric (average distance to centroid)

Detailed CSV outputs for analysis and reporting

💡 If you also have sample1.cpp in this repo:
Read / run sample1 first – it demonstrates K-means on a tiny static dataset (2D, a few points) to understand the algorithm before jumping into the full 8D / large-dataset version.

1. What the Program Does

Given:

an input CSV file with N data points

each point having 8 numerical features (no header expected)

a number of clusters K (passed as a command-line argument)

the program:

Loads all data points from the CSV file into memory.

Runs K-means clustering multiple times (MAX_RUNS, default 10) with different random initial centroids.

For each run:

Randomly picks K data points as initial centroids.

Iteratively does:

Assignment step: assign each point to its closest centroid (squared Euclidean distance, no sqrt in comparison).

Update step: recompute centroids as the mean of points in each cluster.

Stops when no point changes cluster or reaches MAX_ITERS (default 200).

Computes run accuracy = average Euclidean distance of all points to their assigned centroid.

Keeps the best run (the one with lowest average distance).

Recomputes all point-to-centroid distances for the best clustering.

Prints:

Best overall accuracy

Best centroids

Cluster sizes

First 100 points with distances and cluster IDs

Timing summary (loading + clustering)

Writes multiple CSV files:

clustering_results.csv

best_cluster_points.csv

top10_per_cluster.csv

cluster_centroids.csv

2. Building the Program

You need:

A C++ compiler with OpenMP support (e.g. g++, clang++ with appropriate flags).

On Linux:

g++ -O2 -fopenmp -o kmeans_cluster main.cpp


On macOS with Homebrew gcc (recommended):

g++-13 -O2 -fopenmp -o kmeans_cluster main.cpp


(Replace g++-13 with your installed GCC version.)

If you use Apple clang, OpenMP is not enabled by default; you’ll need libomp and extra flags.

3. Usage

Basic syntax:

./kmeans_cluster <input_file> <number_clusters>


Example:

./kmeans_cluster generated_dataset.csv 5


Where:

generated_dataset.csv is your 8D dataset.

5 means K=5 clusters.

4. Key Parameters in the Code

Inside the code you’ll see:

const int D = 8;       // Number of features (dimensions)
const int MAX_RUNS = 10;
const int MAX_ITERS = 200;


You can tune:

MAX_RUNS
Number of random initializations. Higher = more chance to escape bad local minima, but more computation.

MAX_ITERS
Maximum K-means iterations per run. Stops earlier if no assignment changes.

Progress is printed every 5 runs:

if (completed_runs % 5 == 0) {
    // prints progress, best accuracy, elapsed time, ETA
}