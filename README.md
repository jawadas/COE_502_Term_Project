🚀 Parallel K-Means Clustering with Best-of-N Restarts (C++ / OpenMP)

This project implements a parallel K-means clustering algorithm in C++ using OpenMP, supporting:

✅ 8-dimensional datasets

✅ Multiple random restarts (best-of-N strategy)

✅ Custom accuracy metric (average point–centroid distance)

✅ Highly optimized assignment + update steps

✅ Rich CSV outputs for analysis and visualization

💡 Tip: If this repo includes sample1.cpp, run that first.
It demonstrates the algorithm on a tiny 2D dataset so you clearly see the K-means process before moving to the full 8D version.

📌 What the Program Does

Given:

A CSV file with N rows × 8 columns (floats, no header)

A target number of clusters K

The program performs the following:

1. Data Loading

Reads all rows into memory

Measures load time

2. Best-of-N K-Means (Parallel)

Runs K-means MAX_RUNS times (default: 10), each with different random initial centroids.

Each run performs:

Assignment Step (Parallel)

Compute squared Euclidean distance to each centroid

Assign each point to the nearest cluster

Update Step

Recompute centroids as the mean of all assigned points

Stopping Conditions

K-means stops early if:

No point changes cluster
or

MAX_ITERS iterations reached (default: 200)

Run Accuracy

Accuracy = average Euclidean distance of all points to their final centroid.

The run with the lowest average distance becomes the best clustering.

3. Final Reporting

For the best run:

Prints:

Best accuracy

Best centroids

Cluster sizes

First 100 points with distances & cluster IDs

Full timing breakdown

Computes all point-to-centroid distances

Generates multiple CSV outputs:

clustering_results.csv

best_cluster_points.csv

top10_per_cluster.csv

cluster_centroids.csv

cluster_statistics.csv

🛠️ Building the Program

You need a compiler with OpenMP support.

Linux
g++ -O2 -fopenmp -o kmeans_cluster main.cpp

macOS (recommended: Homebrew GCC)

Apple Clang does not support OpenMP.

Install GCC:

brew install gcc


Compile:

g++-13 -O2 -fopenmp -o kmeans_cluster main.cpp


(Replace g++-13 with whichever GCC version you have.)

▶️ Usage
./kmeans_cluster <input_file> <number_clusters>


Example:

./kmeans_cluster generated_dataset.csv 5


Where:

generated_dataset.csv → dataset with 8 columns

5 → number of clusters (K)

⚙️ Key Tunable Parameters

Inside the code:

const int D = 8;          // Number of features
const int MAX_RUNS = 10;  // Random restarts
const int MAX_ITERS = 200; // Max K-means iterations

Notes & Recommendations

Use larger MAX_RUNS (50–100) for more stable clustering, especially with noisy data.

For very large datasets, increase parallelism via:

export OMP_NUM_THREADS=8

The code uses squared distance for comparisons (faster), and sqrt distance only for final accuracy.