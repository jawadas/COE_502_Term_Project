#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>  
#include <omp.h>
#include <cstdlib>  
#include <ctime>    
#include <sstream>  


int main(int argc, char* argv[]) {
    // Start timing
    double start_time = omp_get_wtime();
    
    if (argc < 3) {
        std::cerr << "Please use correct command format: " << argv[0] << " <input_file> <number_clusters>\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    if (!in.is_open()) {
        std::cerr << "Error: could not open file " << argv[1] << "\n";
        return 1;
    }

    int K = std::stoi(argv[2]);
    if (K <= 0) {
        std::cerr << "Error: number of clusters must be positive.\n";
        return 1;
    }

    const int D = 8; // 8 features in the dataset
    std::vector<std::array<float, D>> data;
    
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
    
        std::stringstream ss(line);
        std::array<float, D> point;
    
        bool valid_row = true;
        for (int d = 0; d < D; d++) {
            std::string value;
            if (!std::getline(ss, value, ',')) {
                if (d == 0 && ss.eof()) {
                    valid_row = false;
                    break;
                }
                std::cerr << "Error: line has fewer than " << D << " columns.\n";
                return 1;
            }
            point[d] = std::stof(value);
        }
    
        if (valid_row) {
            data.push_back(point);
        }
    }
    in.close();

    int N = static_cast<int>(data.size());
    if (N == 0) {
        std::cerr << "Error: no data points read from file.\n";
        return 1;
    }

    std::cout << "Loaded " << N << " points with " << D << " dimensions from file.\n";
    
    double load_time = omp_get_wtime();
    std::cout << "Data loading time: " << (load_time - start_time) << " seconds\n";

    // Prepare cluster and centroids
    std::vector<int> cluster(N);
    std::vector<std::array<float, D>> centroids(K);
    std::vector<int> centroid_source_index(K);

    // Randomly initialize centroids from existing data points
    srand(static_cast<unsigned>(time(nullptr)));

    for (int c = 0; c < K; c++) {
        int idx = rand() % N;
        centroids[c] = data[idx];
        centroid_source_index[c] = idx;
    }

    // Print chosen centroids (all 8 dimensions)
    std::cout << "\nInitial centroids chosen randomly:\n";
    for (int c = 0; c < K; c++) {
        std::cout << "  C" << c << " = (";
        for (int d = 0; d < D; d++) {
            std::cout << centroids[c][d];
            if (d < D - 1) std::cout << ", ";
        }
        std::cout << ") from data point index " << centroid_source_index[c] << "\n";
    }
    std::cout << "\n";

    double init_time = omp_get_wtime();
    std::cout << "Centroid initialization time: " << (init_time - load_time) << " seconds\n\n";

    // Store distances for all points (no printing during parallel computation)
    std::vector<std::vector<float>> all_distances(N, std::vector<float>(K));

    double compute_start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float bestDist = INFINITY;
        int bestCluster = -1;

        // Calculate Euclidean distance across all 8 dimensions
        for (int c = 0; c < K; c++) {
            float sum_sq = 0.0f;
            
            for (int d = 0; d < D; d++) {
                float diff = data[i][d] - centroids[c][d];
                sum_sq += diff * diff;
            }
            
            float dist = std::sqrt(sum_sq);
            all_distances[i][c] = dist;

            if (dist < bestDist) {
                bestDist = dist;
                bestCluster = c;
            }
        }

        cluster[i] = bestCluster;
    }

    double compute_end = omp_get_wtime();

    // Now print results (optional - comment out for large datasets)
    std::cout << "Clustering complete. Printing first 100 results...\n\n";
    std::cout << std::fixed << std::setprecision(3);

    // Print header with all 8 dimensions
    std::cout << "Point\t";
    for (int d = 0; d < D; d++) {
        std::cout << "D" << d << "\t";
    }
    for (int c = 0; c < K; c++) {
        std::cout << "DistC" << c << "\t";
    }
    std::cout << "Cluster\n";

    // Print only first 100 points (or all if N < 100)
    int print_limit = (N < 100) ? N : 100;
    
    #pragma omp critical
    {
        for (int i = 0; i < print_limit; i++) {
            std::cout << "P" << i << "\t";
            
            // Print all 8 dimensions
            for (int d = 0; d < D; d++) {
                std::cout << data[i][d] << "\t";
            }

            // Print distances to all centroids
            for (int c = 0; c < K; c++) {
                std::cout << all_distances[i][c] << "\t";
            }

            std::cout << "C" << cluster[i] << "\n";
        }
    }

    if (N > 100) {
        std::cout << "... (showing first 100 of " << N << " points)\n";
    }

    double total_time = compute_end - start_time;
    
    std::cout << "\n";
    std::cout << "=== Timing Summary ===" << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Data loading time:        " << (load_time - start_time) << " seconds" << "\n";
    std::cout << "Initialization time:      " << (init_time - load_time) << " seconds" << "\n";
    std::cout << "Clustering computation:   " << (compute_end - compute_start) << " seconds" << "\n";
    std::cout << "Total elapsed time:       " << total_time << " seconds" << "\n";
    std::cout << std::endl;

    // Write results to file for large datasets
    std::cout << "\nWriting full results to 'clustering_results.csv'...\n";
    
    #pragma omp critical
    {
        std::ofstream out("clustering_results.csv");
        out << std::fixed << std::setprecision(3);
        
        // Header
        out << "Point,";
        for (int d = 0; d < D; d++) {
            out << "D" << d << ",";
        }
        for (int c = 0; c < K; c++) {
            out << "DistC" << c << ",";
        }
        out << "Cluster\n";

        // All data
        for (int i = 0; i < N; i++) {
            out << i << ",";
            for (int d = 0; d < D; d++) {
                out << data[i][d] << ",";
            }
            for (int c = 0; c < K; c++) {
                out << all_distances[i][c] << ",";
            }
            out << cluster[i] << "\n";
        }
        out.close();
    }
    
    std::cout << "Results written successfully.\n";

    return 0;
}