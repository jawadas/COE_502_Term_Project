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
#include <limits>
#include <algorithm>

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

    // Best-of-1000 tracking (shared across all threads)
    const int MAX_RUNS = 10;
    const int MAX_ITERS = 200;

    double best_overall_accuracy = std::numeric_limits<double>::infinity();
    std::vector<std::array<float, D>> best_centroids(K);
    std::vector<int> best_cluster(N, -1);

    // Thread-safe random number generation - one seed per thread
    int num_threads = omp_get_max_threads();
    std::vector<unsigned int> seeds(num_threads);
    for (int i = 0; i < num_threads; i++) {
        seeds[i] = static_cast<unsigned>(time(nullptr)) + i * 1000;
    }

    std::cout << "Running K-means with " << num_threads << " parallel threads...\n";
    std::cout << "Progress will be shown every 5 runs...\n";
    std::cout << std::flush;

    double compute_start = omp_get_wtime();

    // Progress tracking
    int completed_runs = 0;

    // ============================================================================
    // MAIN PARALLELIZATION: Parallelize the outer run loop
    // Each thread independently runs multiple K-means iterations
    // ============================================================================
    #pragma omp parallel for schedule(dynamic, 5)
    for (int run = 0; run < MAX_RUNS; ++run) {
        int tid = omp_get_thread_num();
        
        // Thread-local copies (each thread needs its own workspace)
        std::vector<int> local_cluster(N, -1);
        std::vector<std::array<float, D>> local_centroids(K);

        // ---------- Random initialization of centroids for this run ----------
        for (int c = 0; c < K; c++) {
            int idx = rand_r(&seeds[tid]) % N;
            local_centroids[c] = data[idx];
        }

        int iter;
        for (iter = 0; iter < MAX_ITERS; ++iter) {
            // ---------------- Assignment step ----------------
            int changes = 0;

            for (int i = 0; i < N; i++) {
                float bestDistSq = INFINITY;  // Use squared distance (no sqrt needed)
                int bestCluster = -1;

                // Calculate squared Euclidean distance (avoid sqrt for comparison)
                for (int c = 0; c < K; c++) {
                    float sum_sq = 0.0f;
                    
                    // Calculate distance without nested parallelism
                    for (int d = 0; d < D; d++) {
                        float diff = data[i][d] - local_centroids[c][d];
                        sum_sq += diff * diff;
                    }

                    if (sum_sq < bestDistSq) {
                        bestDistSq = sum_sq;
                        bestCluster = c;
                    }
                }

                if (bestCluster != local_cluster[i]) {
                    changes++;
                }
                local_cluster[i] = bestCluster;
            }

            // ---------------- Update step ----------------
            std::vector<std::array<float, D>> sum(K);
            std::vector<int> count(K, 0);

            // Initialize sums to 0
            for (int c = 0; c < K; ++c) {
                for (int d = 0; d < D; ++d) {
                    sum[c][d] = 0.0f;
                }
            }

            // Accumulate sums
            for (int i = 0; i < N; ++i) {
                int c = local_cluster[i];
                if (c < 0) continue;
                count[c]++;
                for (int d = 0; d < D; ++d) {
                    sum[c][d] += data[i][d];
                }
            }

            // Compute new centroids
            for (int c = 0; c < K; ++c) {
                if (count[c] > 0) {
                    for (int d = 0; d < D; ++d) {
                        local_centroids[c][d] = sum[c][d] / count[c];
                    }
                }
            }

            // Stopping condition: no changes in assignments
            if (changes == 0) {
                break;
            }
        } // end inner K-means iterations

        // ---------------- Accuracy of this run ----------------
        // Accuracy = average distance of all data points from their centroids
        double total_dist = 0.0;

        for (int i = 0; i < N; ++i) {
            int c = local_cluster[i];
            float sum_sq = 0.0f;
            
            for (int d = 0; d < D; ++d) {
                float diff = data[i][d] - local_centroids[c][d];
                sum_sq += diff * diff;
            }
            total_dist += std::sqrt(sum_sq);
        }

        double local_accuracy = total_dist / static_cast<double>(N);

        // Update global best solution (needs synchronization)
        #pragma omp critical
        {
            completed_runs++;
            
            if (local_accuracy < best_overall_accuracy) {
                best_overall_accuracy = local_accuracy;
                best_centroids = local_centroids;
                best_cluster = local_cluster;
            }
            
            // Progress print every 50 runs
            if (completed_runs % 5 == 0) {
                double elapsed = omp_get_wtime() - compute_start;
                double estimated_total = (elapsed / completed_runs) * MAX_RUNS;
                double remaining = estimated_total - elapsed;
                
                std::cout << "Progress: " << completed_runs << " / " << MAX_RUNS 
                          << " runs (" << (completed_runs * 100 / MAX_RUNS) << "%) "
                          << "| Best accuracy: " << best_overall_accuracy 
                          << " | Elapsed: " << elapsed << "s"
                          << " | ETA: " << remaining << "s\n";
                std::cout << std::flush;
            }
        }
    } // end parallel runs

    double compute_end = omp_get_wtime();

    // ---------------- Recompute distances for BEST clustering for output ----------------
    std::vector<std::vector<float>> all_distances(N, std::vector<float>(K));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < K; ++c) {
            float sum_sq = 0.0f;
            
            #pragma omp simd reduction(+:sum_sq)
            for (int d = 0; d < D; ++d) {
                float diff = data[i][d] - best_centroids[c][d];
                sum_sq += diff * diff;
            }
            all_distances[i][c] = std::sqrt(sum_sq);
        }
    }

    // ---------------- Print best centroids & accuracy ----------------
    std::cout << "\n=== Best Clustering over " << MAX_RUNS << " runs ===\n";
    std::cout << "Best overall accuracy (average distance to centroid): "
              << best_overall_accuracy << "\n\n";

    std::cout << "Centroids of the BEST cluster configuration:\n";
    for (int c = 0; c < K; ++c) {
        std::cout << "  C" << c << " = (";
        for (int d = 0; d < D; ++d) {
            std::cout << best_centroids[c][d];
            if (d < D - 1) std::cout << ", ";
        }
        std::cout << ")\n";
    }
    std::cout << "\n";

    // ---------------- Cluster statistics ----------------
    std::vector<int> cluster_sizes(K, 0);
    for (int i = 0; i < N; ++i) {
        cluster_sizes[best_cluster[i]]++;
    }
    
    std::cout << "Cluster sizes:\n";
    for (int c = 0; c < K; ++c) {
        std::cout << "  C" << c << ": " << cluster_sizes[c] << " points\n";
    }
    std::cout << "\n";

    // ---------------- Final reporting ----------------
    std::cout << "Clustering complete. Printing first 100 results of BEST clustering...\n\n";
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

    int print_limit = std::min(N, 100);

    for (int i = 0; i < print_limit; i++) {
        std::cout << "P" << i << "\t";

        for (int d = 0; d < D; d++) {
            std::cout << data[i][d] << "\t";
        }

        for (int c = 0; c < K; c++) {
            std::cout << all_distances[i][c] << "\t";
        }

        std::cout << "C" << best_cluster[i] << "\n";
    }

    if (N > 100) {
        std::cout << "... (showing first 100 of " << N << " points)\n";
    }

    double total_time = compute_end - start_time;

    std::cout << "\n";
    std::cout << "=== Timing Summary ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Data loading time:              " << (load_time - start_time)       << " seconds\n";
    std::cout << "K-means (" << MAX_RUNS << " runs) time:   " 
              << (compute_end - compute_start) << " seconds\n";
    std::cout << "Total elapsed time:             " << total_time                    << " seconds\n";
    std::cout << "Average time per run:           " << (compute_end - compute_start) / MAX_RUNS << " seconds\n";
    std::cout << "Threads used:                   " << num_threads << "\n";
    std::cout << std::endl;

    // ---------------- Write ALL clustering results to file ----------------
    std::cout << "\n=== Writing Results to CSV Files ===\n";
    
    // 1. Write complete clustering results
    std::cout << "Writing complete clustering results to 'clustering_results.csv'...\n";
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
            out << best_cluster[i] << "\n";
        }
        out.close();
    }
    std::cout << "  ✓ Complete results written (" << N << " points)\n";

    // 2. Write best representative points per cluster (closest to centroids)
    std::cout << "\nWriting best representative points to 'best_cluster_points.csv'...\n";
    {
        std::ofstream out("best_cluster_points.csv");
        out << std::fixed << std::setprecision(3);

        // Header
        out << "Cluster,Point_Index,Distance_to_Centroid,";
        for (int d = 0; d < D; d++) {
            out << "D" << d;
            if (d < D - 1) out << ",";
        }
        out << "\n";

        // For each cluster, find the point closest to the centroid
        std::vector<int> best_point_per_cluster(K, -1);
        std::vector<float> best_distance_per_cluster(K, INFINITY);

        for (int i = 0; i < N; ++i) {
            int c = best_cluster[i];
            float dist = all_distances[i][c];
            
            if (dist < best_distance_per_cluster[c]) {
                best_distance_per_cluster[c] = dist;
                best_point_per_cluster[c] = i;
            }
        }

        // Write best point for each cluster
        for (int c = 0; c < K; ++c) {
            if (best_point_per_cluster[c] >= 0) {
                int idx = best_point_per_cluster[c];
                out << c << "," << idx << "," << best_distance_per_cluster[c] << ",";
                for (int d = 0; d < D; ++d) {
                    out << data[idx][d];
                    if (d < D - 1) out << ",";
                }
                out << "\n";
            }
        }
        out.close();
    }
    std::cout << "  ✓ Best representative points written (" << K << " points, one per cluster)\n";

    // 3. Write top 10 best points per cluster
    std::cout << "\nWriting top 10 points per cluster to 'top10_per_cluster.csv'...\n";
    {
        std::ofstream out("top10_per_cluster.csv");
        out << std::fixed << std::setprecision(3);

        // Header
        out << "Cluster,Rank,Point_Index,Distance_to_Centroid,";
        for (int d = 0; d < D; d++) {
            out << "D" << d;
            if (d < D - 1) out << ",";
        }
        out << "\n";

        // For each cluster
        for (int c = 0; c < K; ++c) {
            // Collect all points in this cluster with their distances
            std::vector<std::pair<float, int>> cluster_points; // (distance, point_index)
            
            for (int i = 0; i < N; ++i) {
                if (best_cluster[i] == c) {
                    cluster_points.push_back({all_distances[i][c], i});
                }
            }

            // Sort by distance (closest first)
            std::sort(cluster_points.begin(), cluster_points.end());

            // Write top 10 (or fewer if cluster is small)
            int top_n = std::min(10, static_cast<int>(cluster_points.size()));
            for (int rank = 0; rank < top_n; ++rank) {
                int idx = cluster_points[rank].second;
                float dist = cluster_points[rank].first;
                
                out << c << "," << (rank + 1) << "," << idx << "," << dist << ",";
                for (int d = 0; d < D; ++d) {
                    out << data[idx][d];
                    if (d < D - 1) out << ",";
                }
                out << "\n";
            }
        }
        out.close();
    }
    std::cout << "  ✓ Top 10 points per cluster written\n";

    // 4. Write cluster centroids
    std::cout << "\nWriting cluster centroids to 'cluster_centroids.csv'...\n";
    {
        std::ofstream out("cluster_centroids.csv");
        out << std::fixed << std::setprecision(3);

        // Header
        out << "Cluster,Size,";
        for (int d = 0; d < D; d++) {
            out << "D" << d;
            if (d < D - 1) out << ",";
        }
        out << "\n";

        // Write each centroid
        for (int c = 0; c < K; ++c) {
            out << c << "," << cluster_sizes[c] << ",";
            for (int d = 0; d < D; ++d) {
                out << best_centroids[c][d];
                if (d < D - 1) out << ",";
            }
            out << "\n";
        }
        out.close();
    }
    std::cout << "  ✓ Cluster centroids written (" << K << " centroids)\n";

    // 5. Write cluster summary statistics
    std::cout << "\nWriting cluster statistics to 'cluster_statistics.csv'...\n";
    {
        std::ofstream out("cluster_statistics.csv");
        out << std::fixed << std::setprecision(3);

        // Header
        out << "Cluster,Size,Avg_Distance,Min_Distance,Max_Distance,Std_Distance\n";

        // Calculate statistics for each cluster
        for (int c = 0; c < K; ++c) {
            std::vector<float> distances_in_cluster;
            
            for (int i = 0; i < N; ++i) {
                if (best_cluster[i] == c) {
                    distances_in_cluster.push_back(all_distances[i][c]);
                }
            }

            if (distances_in_cluster.empty()) continue;

            // Calculate statistics
            float sum = 0.0f, min_d = INFINITY, max_d = -INFINITY;
            for (float d : distances_in_cluster) {
                sum += d;
                min_d = std::min(min_d, d);
                max_d = std::max(max_d, d);
            }
            float avg = sum / distances_in_cluster.size();

            // Calculate standard deviation
            float var_sum = 0.0f;
            for (float d : distances_in_cluster) {
                float diff = d - avg;
                var_sum += diff * diff;
            }
            float std_dev = std::sqrt(var_sum / distances_in_cluster.size());

            out << c << "," << distances_in_cluster.size() << "," 
                << avg << "," << min_d << "," << max_d << "," << std_dev << "\n";
        }
        out.close();
    }
    std::cout << "  ✓ Cluster statistics written\n";

    // Final summary
    std::cout << "\n=== All Results Successfully Written ===\n";
    std::cout << "Files created:\n";
    std::cout << "  1. clustering_results.csv       - All " << N << " points with full details\n";
    std::cout << "  2. best_cluster_points.csv      - Best representative point per cluster\n";
    std::cout << "  3. top10_per_cluster.csv        - Top 10 closest points per cluster\n";
    std::cout << "  4. cluster_centroids.csv        - Centroid coordinates for all clusters\n";
    std::cout << "  5. cluster_statistics.csv       - Statistical summary per cluster\n";
    std::cout << "\n";

    return 0;
}