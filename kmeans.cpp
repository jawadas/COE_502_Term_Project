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
using namespace std;

int main(int argc, char* argv[]) {
    omp_set_dynamic(0);
    // Start timing
    double start_time = omp_get_wtime();
    
    if (argc < 3) {
        cerr << "Please use correct command format: " << argv[0] << " <input_file> <number_clusters>\n";
        return 1;
    }

    ifstream in(argv[1]);
    if (!in.is_open()) {
        cerr << "Error: could not open file " << argv[1] << "\n";
        return 1;
    }

    int K = stoi(argv[2]);
    if (K <= 0) {
        cerr << "Error: number of clusters must be positive.\n";
        return 1;
    }

    const int D = 8; // 8 features in the dataset
    vector<array<float, D>> data;

    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        array<float, D> point;

        bool valid_row = true;
        for (int d = 0; d < D; d++) {
            string value;
            if (!getline(ss, value, ',')) {
                if (d == 0 && ss.eof()) {
                    valid_row = false;
                    break;
                }
                cerr << "Error: line has fewer than " << D << " columns.\n";
                return 1;
            }
            point[d] = stof(value);
        }

        if (valid_row) {
            data.push_back(point);
        }
    }
    in.close();

    int N = static_cast<int>(data.size());
    if (N == 0) {
        cerr << "Error: no data points read from file.\n";
        return 1;
    }

    cout << "Loaded " << N << " points with " << D << " dimensions from file.\n";

    double load_time = omp_get_wtime();
    cout << "Data loading time: " << (load_time - start_time) << " seconds\n";

    // Best-of-1000 tracking (shared across all threads)
    const int MAX_RUNS = 10;
    const int MAX_ITERS = 2000;

    double best_overall_accuracy = numeric_limits<double>::infinity();
    vector<array<float, D>> best_centroids(K);
    vector<int> best_cluster(N, -1);

    // Thread-safe random number generation - one seed per thread
    int num_threads = omp_get_max_threads();
    vector<unsigned int> seeds(num_threads);
    for (int i = 0; i < num_threads; i++) {
        seeds[i] = 1234u + i * 1000;
    }

    cout << "Running K-means with " << num_threads << " parallel threads...\n";
    cout << "Progress will be shown every 5 runs...\n";
    cout << flush;

    double compute_start = omp_get_wtime();

    // Progress tracking
    int completed_runs = 0;

    // ============================================================================
    // MAIN PARALLELIZATION: Parallelize the outer run loop
    // Each thread independently runs multiple K-means iterations
    // ============================================================================
    #pragma omp parallel for schedule(dynamic) proc_bind(spread)
    for (int run = 0; run < MAX_RUNS; ++run) {
        int tid = omp_get_thread_num();
        
        // Thread-local copies (each thread needs its own workspace)
        vector<int> local_cluster(N, -1);
        vector<array<float, D>> local_centroids(K);

        // ---------- Random initialization of centroids for this run ----------
        for (int c = 0; c < K; c++) {
            int idx = rand_r(&seeds[tid]) % N;
            local_centroids[c] = data[idx];
        }

        int iter;
        for (iter = 0; iter < MAX_ITERS; ++iter) {
           // -------------------- ASSIGNMENT STEP (PARALLEL) --------------------
int changes = 0;

for (int i = 0; i < N; i++) {
    float bestDist = INFINITY;
    int bestCluster = -1;

    // Loop over clusters
    for (int c = 0; c < K; c++) {
        float sum_sq = 0.0f;

        // SIMD vectorization: compute squared distance in parallel
        #pragma omp simd reduction(+:sum_sq)
        for (int d = 0; d < D; d++) {
            float diff = data[i][d] - local_centroids[c][d];
            sum_sq += diff * diff;
        }

        if (sum_sq < bestDist) {
            bestDist = sum_sq;
            bestCluster = c;
        }
    }

    if (bestCluster != local_cluster[i]) {
        changes++;
    }
    local_cluster[i] = bestCluster;
}
            // ---------------- Update step ----------------
            vector<array<float, D>> sum(K);
            vector<int> count(K, 0);

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
            total_dist += sqrt(sum_sq);
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
                
                cout << "Progress: " << completed_runs << " / " << MAX_RUNS 
                          << " runs (" << (completed_runs * 100 / MAX_RUNS) << "%) "
                          << "| Best accuracy: " << best_overall_accuracy 
                          << " | Elapsed: " << elapsed << "s"
                          << " | ETA: " << remaining << "s\n";
                cout << flush;
            }
        }
    } // end parallel runs

    double compute_end = omp_get_wtime();

    // ---------------- Recompute distances for BEST clustering for output ----------------
    vector<vector<float>> all_distances(N, vector<float>(K));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int c = 0; c < K; ++c) {
            float sum_sq = 0.0f;
            
            #pragma omp simd reduction(+:sum_sq)
            for (int d = 0; d < D; ++d) {
                float diff = data[i][d] - best_centroids[c][d];
                sum_sq += diff * diff;
            }
            all_distances[i][c] = sqrt(sum_sq);
        }
    }

    // ---------------- Print best centroids & accuracy ----------------
    cout << "\n=== Best Clustering over " << MAX_RUNS << " runs ===\n";
    cout << "Best overall accuracy (average distance to centroid): "
              << best_overall_accuracy << "\n\n";

    cout << "Centroids of the BEST cluster configuration:\n";
    for (int c = 0; c < K; ++c) {
        cout << "  C" << c << " = (";
        for (int d = 0; d < D; ++d) {
            cout << best_centroids[c][d];
            if (d < D - 1) cout << ", ";
        }
        cout << ")\n";
    }
    cout << "\n";

    // ---------------- Cluster statistics ----------------
    vector<int> cluster_sizes(K, 0);
    for (int i = 0; i < N; ++i) {
        cluster_sizes[best_cluster[i]]++;
    }
    
    cout << "Cluster sizes:\n";
    for (int c = 0; c < K; ++c) {
        cout << "  C" << c << ": " << cluster_sizes[c] << " points\n";
    }
    cout << "\n";

    // ---------------- Final reporting ----------------
    cout << "Clustering complete. Printing first 100 results of BEST clustering...\n\n";
    cout << fixed << setprecision(3);

    // Print header with all 8 dimensions
    cout << "Point\t";
    for (int d = 0; d < D; d++) {
        cout << "D" << d << "\t";
    }
    for (int c = 0; c < K; c++) {
        cout << "DistC" << c << "\t";
    }
    cout << "Cluster\n";

    int print_limit = min(N, 100);

    for (int i = 0; i < print_limit; i++) {
        cout << "P" << i << "\t";

        for (int d = 0; d < D; d++) {
            cout << data[i][d] << "\t";
        }

        for (int c = 0; c < K; c++) {
            cout << all_distances[i][c] << "\t";
        }

        cout << "C" << best_cluster[i] << "\n";
    }

    if (N > 100) {
        cout << "... (showing first 100 of " << N << " points)\n";
    }

    double total_time = compute_end - start_time;

    cout << "\n";
    cout << "=== Timing Summary ===\n";
    cout << fixed << setprecision(6);
    cout << "Data loading time:              " << (load_time - start_time)       << " seconds\n";
    cout << "K-means (" << MAX_RUNS << " runs) time:   " 
              << (compute_end - compute_start) << " seconds\n";
    cout << "Total elapsed time:             " << total_time                    << " seconds\n";
    cout << "Average time per run:           " << (compute_end - compute_start) / MAX_RUNS << " seconds\n";
    cout << "Threads used:                   " << num_threads << "\n";
    cout << endl;

    // ---------------- Write ALL clustering results to file ----------------
    cout << "\n=== Writing Results to CSV Files ===\n";
    
    // 1. Write complete clustering results
    cout << "Writing complete clustering results to 'clustering_results.csv'...\n";
    {
        ofstream out("clustering_results.csv");
        out << fixed << setprecision(3);

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
    cout << "  ✓ Complete results written (" << N << " points)\n";

    // 2. Write best representative points per cluster (closest to centroids)
    cout << "\nWriting best representative points to 'best_cluster_points.csv'...\n";
    {
        ofstream out("best_cluster_points.csv");
        out << fixed << setprecision(3);

        // Header
        out << "Cluster,Point_Index,Distance_to_Centroid,";
        for (int d = 0; d < D; d++) {
            out << "D" << d;
            if (d < D - 1) out << ",";
        }
        out << "\n";

        // For each cluster, find the point closest to the centroid
        vector<int> best_point_per_cluster(K, -1);
        vector<float> best_distance_per_cluster(K, INFINITY);

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
    cout << "  ✓ Best representative points written (" << K << " points, one per cluster)\n";

    // 3. Write top 10 best points per cluster
    cout << "\nWriting top 10 points per cluster to 'top10_per_cluster.csv'...\n";
    {
        ofstream out("top10_per_cluster.csv");
        out << fixed << setprecision(3);

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
            vector<pair<float, int>> cluster_points; // (distance, point_index)
            
            for (int i = 0; i < N; ++i) {
                if (best_cluster[i] == c) {
                    cluster_points.push_back({all_distances[i][c], i});
                }
            }

            // Sort by distance (closest first)
            sort(cluster_points.begin(), cluster_points.end());

            // Write top 10 (or fewer if cluster is small)
            int top_n = min(10, static_cast<int>(cluster_points.size()));
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
    cout << "  ✓ Top 10 points per cluster written\n";

    // 4. Write cluster centroids
    cout << "\nWriting cluster centroids to 'cluster_centroids.csv'...\n";
    {
        ofstream out("cluster_centroids.csv");
        out << fixed << setprecision(3);

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
    cout << "  ✓ Cluster centroids written (" << K << " centroids)\n";

    // 5. Write cluster summary statistics
    cout << "\nWriting cluster statistics to 'cluster_statistics.csv'...\n";
    {
        ofstream out("cluster_statistics.csv");
        out << fixed << setprecision(3);

        // Header
        out << "Cluster,Size,Avg_Distance,Min_Distance,Max_Distance,Std_Distance\n";

        // Calculate statistics for each cluster
        for (int c = 0; c < K; ++c) {
            vector<float> distances_in_cluster;
            
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
                min_d = min(min_d, d);
                max_d = max(max_d, d);
            }
            float avg = sum / distances_in_cluster.size();

            // Calculate standard deviation
            float var_sum = 0.0f;
            for (float d : distances_in_cluster) {
                float diff = d - avg;
                var_sum += diff * diff;
            }
            float std_dev = sqrt(var_sum / distances_in_cluster.size());

            out << c << "," << distances_in_cluster.size() << "," 
                << avg << "," << min_d << "," << max_d << "," << std_dev << "\n";
        }
        out.close();
    }
    cout << "  ✓ Cluster statistics written\n";

    // Final summary
    cout << "\n=== All Results Successfully Written ===\n";
    cout << "Files created:\n";
    cout << "  1. clustering_results.csv       - All " << N << " points with full details\n";
    cout << "  2. best_cluster_points.csv      - Best representative point per cluster\n";
    cout << "  3. top10_per_cluster.csv        - Top 10 closest points per cluster\n";
    cout << "  4. cluster_centroids.csv        - Centroid coordinates for all clusters\n";
    cout << "  5. cluster_statistics.csv       - Statistical summary per cluster\n";
    cout << "\n";

    return 0;
}