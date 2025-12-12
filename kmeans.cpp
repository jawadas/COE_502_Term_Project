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
        cerr << "Please use correct command format: " << argv[0]
             << " <input_file> <number_clusters>\n";
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
    data.reserve(1000000);

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

    // Best-of tracking (shared across all threads)
    const int MAX_RUNS = 100;
    const int MAX_ITERS = 2000;

    double best_overall_accuracy = numeric_limits<double>::infinity();
    vector<array<float, D>> best_centroids(K);
    vector<int> best_cluster(N, -1);

    // Thread-safe random number generation - one seed per thread
    int num_threads = omp_get_max_threads();
    vector<unsigned int> seeds(num_threads);

    // CHANGE #1: time-based seeds (not fixed), still unique per thread
    unsigned int base_seed = static_cast<unsigned int>(time(nullptr));
    for (int i = 0; i < num_threads; i++) {
        seeds[i] = base_seed + static_cast<unsigned int>(i) * 1000u;
    }

    cout << "Running K-means with " << num_threads << " parallel threads...\n";
    cout << "Progress will be shown every 5 runs...\n";
    cout << flush;

    double compute_start = omp_get_wtime();

    // Progress tracking
    int completed_runs = 0;

    // ============================
    // Timing accumulators (GLOBAL)
    // ============================
    double total_assign_time = 0.0;
    double total_update_time = 0.0;
    long long total_assign_steps = 0;
    long long total_update_steps = 0;

    // ============================================================================
    // MAIN PARALLELIZATION: Parallelize the outer run loop
    // ============================================================================
    #pragma omp parallel for schedule(dynamic) proc_bind(spread)
    for (int run = 0; run < MAX_RUNS; ++run) {
        int tid = omp_get_thread_num();

        // Thread-local copies (each thread needs its own workspace)
        vector<int> local_cluster(N, -1);
        vector<array<float, D>> local_centroids(K);

        // Thread-local timing (accumulate per run)
        double local_assign_time = 0.0;
        double local_update_time = 0.0;
        long long local_assign_steps = 0;
        long long local_update_steps = 0;

        // ---------- Random initialization of centroids for this run ----------
        for (int c = 0; c < K; c++) {
            int idx = rand_r(&seeds[tid]) % N;
            local_centroids[c] = data[idx];
        }

        // CHANGE #2: allocate ONCE per run, reuse every iteration
        vector<array<float, D>> sum(K);
        vector<int> count(K);

        int iter;
        for (iter = 0; iter < MAX_ITERS; ++iter) {

            // -------------------- ASSIGNMENT STEP (TIMED) --------------------
            double t_assign0 = omp_get_wtime();

            int changes = 0;

            for (int i = 0; i < N; i++) {
                float bestDist = INFINITY;
                int bestCluster = -1;

                // Loop over clusters
                for (int c = 0; c < K; c++) {
                    float sum_sq = 0.0f;

                    // SIMD hint (may not do much at -O0, but harmless)
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

            double t_assign1 = omp_get_wtime();
            local_assign_time += (t_assign1 - t_assign0);
            local_assign_steps++;

            // -------------------- UPDATE STEP (TIMED) --------------------
            double t_update0 = omp_get_wtime();

            // Reset buffers (reuse, no re-allocation)
            fill(count.begin(), count.end(), 0);
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

            double t_update1 = omp_get_wtime();
            local_update_time += (t_update1 - t_update0);
            local_update_steps++;

            // Stopping condition: no changes in assignments
            if (changes == 0) {
                break;
            }
        } // end inner K-means iterations

        // ---------------- Accuracy of this run ----------------
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

        // Update global best solution + progress + timing totals
        #pragma omp critical
        {
            completed_runs++;

            total_assign_time += local_assign_time;
            total_update_time += local_update_time;
            total_assign_steps += local_assign_steps;
            total_update_steps += local_update_steps;

            if (local_accuracy < best_overall_accuracy) {
                best_overall_accuracy = local_accuracy;
                best_centroids = local_centroids;
                best_cluster = local_cluster;
            }

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

    cout << "Point\t";
    for (int d = 0; d < D; d++) cout << "D" << d << "\t";
    for (int c = 0; c < K; c++) cout << "DistC" << c << "\t";
    cout << "Cluster\n";

    int print_limit = min(N, 100);
    for (int i = 0; i < print_limit; i++) {
        cout << "P" << i << "\t";
        for (int d = 0; d < D; d++) cout << data[i][d] << "\t";
        for (int c = 0; c < K; c++) cout << all_distances[i][c] << "\t";
        cout << "C" << best_cluster[i] << "\n";
    }
    if (N > 100) cout << "... (showing first 100 of " << N << " points)\n";

    double total_time = compute_end - start_time;

    cout << "\n=== Timing Summary ===\n";
    cout << fixed << setprecision(6);
    cout << "Data loading time:              " << (load_time - start_time) << " seconds\n";
    cout << "K-means (" << MAX_RUNS << " runs) time:   " << (compute_end - compute_start) << " seconds\n";
    cout << "Total elapsed time:             " << total_time << " seconds\n";
    cout << "Average time per run:           " << (compute_end - compute_start) / MAX_RUNS << " seconds\n";
    cout << "Threads used:                   " << num_threads << "\n";

    double timed_total = total_assign_time + total_update_time;

    double avg_assign_per_step = (total_assign_steps > 0)
        ? (total_assign_time / static_cast<double>(total_assign_steps))
        : 0.0;

    double avg_update_per_step = (total_update_steps > 0)
        ? (total_update_time / static_cast<double>(total_update_steps))
        : 0.0;

    cout << "\n--- Detailed Phase Timing (SUM over all runs/iterations across threads) ---\n";
    cout << "Assignment total time:          " << total_assign_time << " seconds\n";
    cout << "Update total time:              " << total_update_time << " seconds\n";
    cout << "Timed (assign+update) total:    " << timed_total << " seconds\n";
    cout << "Assignment % (timed parts):     "
         << (timed_total > 0 ? (100.0 * total_assign_time / timed_total) : 0.0) << "%\n";
    cout << "Update % (timed parts):         "
         << (timed_total > 0 ? (100.0 * total_update_time / timed_total) : 0.0) << "%\n";
    cout << "Assignment steps timed:         " << total_assign_steps << "\n";
    cout << "Update steps timed:             " << total_update_steps << "\n";

    cout << "\n--- Averages (per step, averaged over ALL runs/iterations) ---\n";
    cout << "Avg time per A-step:            " << avg_assign_per_step << " seconds\n";
    cout << "Avg time per U-step:            " << avg_update_per_step << " seconds\n";
    cout << endl;

    // ---------------- Write ALL clustering results to file ----------------
    cout << "\n=== Writing Results to CSV Files ===\n";

    cout << "Writing complete clustering results to 'clustering_results.csv'...\n";
    {
        ofstream out("clustering_results.csv");
        out << fixed << setprecision(3);

        out << "Point,";
        for (int d = 0; d < D; d++) out << "D" << d << ",";
        for (int c = 0; c < K; c++) out << "DistC" << c << ",";
        out << "Cluster\n";

        for (int i = 0; i < N; i++) {
            out << i << ",";
            for (int d = 0; d < D; d++) out << data[i][d] << ",";
            for (int c = 0; c < K; c++) out << all_distances[i][c] << ",";
            out << best_cluster[i] << "\n";
        }
        out.close();
    }
    cout << "  ✓ Complete results written (" << N << " points)\n";

    cout << "\nWriting best representative points to 'best_cluster_points.csv'...\n";
    {
        ofstream out("best_cluster_points.csv");
        out << fixed << setprecision(3);

        out << "Cluster,Point_Index,Distance_to_Centroid,";
        for (int d = 0; d < D; d++) {
            out << "D" << d;
            if (d < D - 1) out << ",";
        }
        out << "\n";

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

    cout << "\nWriting top 10 points per cluster to 'top10_per_cluster.csv'...\n";
    {
        ofstream out("top10_per_cluster.csv");
        out << fixed << setprecision(3);

        out << "Cluster,Rank,Point_Index,Distance_to_Centroid,";
        for (int d = 0; d < D; d++) {
            out << "D" << d;
            if (d < D - 1) out << ",";
        }
        out << "\n";

        for (int c = 0; c < K; ++c) {
            vector<pair<float, int>> cluster_points;
            for (int i = 0; i < N; ++i) {
                if (best_cluster[i] == c) {
                    cluster_points.push_back({all_distances[i][c], i});
                }
            }

            sort(cluster_points.begin(), cluster_points.end());

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

    cout << "\nWriting cluster centroids to 'cluster_centroids.csv'...\n";
    {
        ofstream out("cluster_centroids.csv");
        out << fixed << setprecision(3);

        out << "Cluster,Size,";
        for (int d = 0; d < D; d++) {
            out << "D" << d;
            if (d < D - 1) out << ",";
        }
        out << "\n";

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

    cout << "\nWriting cluster statistics to 'cluster_statistics.csv'...\n";
    {
        ofstream out("cluster_statistics.csv");
        out << fixed << setprecision(3);

        out << "Cluster,Size,Avg_Distance,Min_Distance,Max_Distance,Std_Distance\n";

        for (int c = 0; c < K; ++c) {
            vector<float> distances_in_cluster;
            for (int i = 0; i < N; ++i) {
                if (best_cluster[i] == c) {
                    distances_in_cluster.push_back(all_distances[i][c]);
                }
            }

            if (distances_in_cluster.empty()) continue;

            float sumd = 0.0f, min_d = INFINITY, max_d = -INFINITY;
            for (float dval : distances_in_cluster) {
                sumd += dval;
                min_d = min(min_d, dval);
                max_d = max(max_d, dval);
            }
            float avg = sumd / distances_in_cluster.size();

            float var_sum = 0.0f;
            for (float dval : distances_in_cluster) {
                float diff = dval - avg;
                var_sum += diff * diff;
            }
            float std_dev = sqrt(var_sum / distances_in_cluster.size());

            out << c << "," << distances_in_cluster.size() << ","
                << avg << "," << min_d << "," << max_d << "," << std_dev << "\n";
        }
        out.close();
    }
    cout << "  ✓ Cluster statistics written\n";

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
