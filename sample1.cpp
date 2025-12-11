#include <iostream>
#include <cmath>
#include <omp.h>
using namespace std;

// Full K-means on a static dataset:
// - N points, 2 clusters, 2D
// - Assignment (parallel with OpenMP)
// - Update (parallel with OpenMP using local accumulators)
// - Repeat until no point changes cluster membership

int main() {
    const int N = 20;   // number of points
    const int K = 2;    // number of clusters
    const int D = 2;    // dimensions

    // Data points: (x, y)
    float data[N][D] = {
        {0.127f, 0.0f},
        {0.678f, 0.0f},
        {1.739f, 2.0f},
        {1.805f, 0.0f},
        {2.679f, 0.0f},
        {2.747f, 5.0f},
        {3.260f, 0.0f},
        {3.522f, 1.0f},
        {3.562f, 0.0f},
        {3.685f, 7.0f},
        {3.732f, 0.0f},
        {4.098f, 0.0f},
        {4.335f, 0.0f},
        {4.626f, 1.0f},
        {4.895f, 2.2f},
        {5.095f, 3.1f},
        {5.640f, 2.5f},
        {8.274f, 0.0f},
        {8.396f, 5.0f},
        {9.062f, 0.0f}
    };

    // Initial centroids
    float centroids[K][D] = {
        {1.0f, 1.0f},   // C0
        {2.5f, 1.5f}    // C1
    };

    int assignment[N];      // cluster index for each point
    const int MAX_ITERS = 100;

    // Initialize assignments to -1 so first iteration always counts as "changed"
    for (int i = 0; i < N; ++i) {
        assignment[i] = -1;
    }

    cout << "Initial centroids:\n";
    for (int c = 0; c < K; ++c) {
        cout << "  C" << c << " = ("
                  << centroids[c][0] << ", "
                  << centroids[c][1] << ")\n";
    }
    cout << "\n";

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        cout << "=== Iteration " << iter << " ===\n";

        // ---------------- A-step: assign points to nearest centroid (parallel) ----------------
        int changes = 0;

        #pragma omp parallel for reduction(+:changes)
        for (int i = 0; i < N; i++) {
            float x = data[i][0];
            float y = data[i][1];

            // distance to C0 (squared)
            float dx0 = x - centroids[0][0];
            float dy0 = y - centroids[0][1];
            float dist0 = dx0 * dx0 + dy0 * dy0;

            // distance to C1 (squared)
            float dx1 = x - centroids[1][0];
            float dy1 = y - centroids[1][1];
            float dist1 = dx1 * dx1 + dy1 * dy1;

            int old_cluster = assignment[i];
            int new_cluster = (dist0 <= dist1) ? 0 : 1;

            if (new_cluster != old_cluster) {
                changes++;
            }

            assignment[i] = new_cluster;

            // OPTIONAL: debug print per thread (comment out if too noisy)
            /*
            int tid = omp_get_thread_num();
            #pragma omp critical
            {
                cout << "[Assign][Thread " << tid << "] "
                          << "Point " << i << " (" << x << ", " << y << ") "
                          << "dist0=" << dist0 << " dist1=" << dist1
                          << " -> Cluster " << assignment[i] << "\n";
            }
            */
        }

        // Print assignments for this iteration (serial, ordered)
        for (int i = 0; i < N; ++i) {
            cout << "Point " << i << " ("
                      << data[i][0] << ", " << data[i][1]
                      << ") -> Cluster " << assignment[i] << "\n";
        }
        cout << "Changed assignments this iteration: " << changes << "\n";

        // ---------------- U-step: recompute centroids as mean of cluster points (parallel) ----------------
        float sum[K][D] = {0};   // global sums of coordinates per cluster
        int count[K] = {0};      // global counts per cluster

        #pragma omp parallel
        {
            // Thread-local accumulators
            float local_sum[K][D] = {0};
            int local_count[K] = {0};

            // Each thread processes a chunk of points
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                int c = assignment[i];   // cluster index for this point
                local_count[c]++;
                for (int d = 0; d < D; ++d) {
                    local_sum[c][d] += data[i][d];
                }
            }

            // Merge local sums into global sums (short critical section)
            #pragma omp critical
            {
                for (int c = 0; c < K; ++c) {
                    count[c] += local_count[c];
                    for (int d = 0; d < D; ++d) {
                        sum[c][d] += local_sum[c][d];
                    }
                }
            }
        }

        // Now compute new centroids from global sum & count (can also be parallel)
        #pragma omp parallel for
        for (int c = 0; c < K; ++c) {
            if (count[c] > 0) {
                for (int d = 0; d < D; ++d) {
                    centroids[c][d] = sum[c][d] / count[c];
                }
            }
        }

        cout << "Updated centroids:\n";
        for (int c = 0; c < K; ++c) {
            cout << "  C" << c << " = ("
                      << centroids[c][0] << ", "
                      << centroids[c][1] << ")"
                      << "  [count = " << count[c] << "]\n";
        }
        cout << "\n";

        // ---------------- Stopping condition: no changes ----------------
        if (changes == 0) {
            cout << "Converged after " << iter << " iterations.\n";
            break;
        }

        if (iter == MAX_ITERS - 1) {
            cout << "Reached max iterations without full convergence.\n";
        }
    }

    return 0;
}
