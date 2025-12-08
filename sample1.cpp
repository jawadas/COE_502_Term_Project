#include <iostream>
#include <cmath>
#include <omp.h>
// <READ THIS> //
// in this sample code, I am creating static dataset and initializing static centroids. 
//I am then performing Assignment-Step only, pragma is used to ensure each thread calcules result independantly
// without interleaving and the critical section ensures the output is produced by one thread only. 

int main() {
    // Starting dataset: 4 points, 2 clusters, and 2 features (dimensions)
    const int N = 4;   // number of static points
    const int K = 2;   // number of clusters
    const int D = 2;   // dimensions

    // Data points: (x, y)
    float data[N][D] = {
        {1.0, 1.0},
        {1.5, 2.0},
        {5.0, 7.0},
        {6.0, 8.0}
    };

    // Initial centroids
    float centroids[K][D] = {
        {1.0, 1.0},   // C0
        {5.0, 7.0}    // C1
    };

    int assignment[N]; // cluster index for each point

    // --- A-step: assign each point to nearest centroid (parallel)
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {

        // Get thread ID
        int tid = omp_get_thread_num();

        float x = data[i][0];
        float y = data[i][1];

        // distance to cluster 0 (squared)
        float dx0 = x - centroids[0][0];
        float dy0 = y - centroids[0][1];
        float dist0 = dx0 * dx0 + dy0 * dy0;

        // distance to cluster 1 (squared)
        float dx1 = x - centroids[1][0];
        float dy1 = y - centroids[1][1];
        float dist1 = dx1 * dx1 + dy1 * dy1;

        // Assign cluster
        if (dist0 <= dist1)
            assignment[i] = 0;
        else
            assignment[i] = 1;

        // CRITICAL section for clean printing
        #pragma omp critical
        {
            std::cout << "[Thread " << tid << "] "
                      << "Point " << i
                      << " (" << x << ", " << y << ") "
                      << "dist0=" << dist0 << " dist1=" << dist1
                      << " -> Cluster " << assignment[i] << "\n";
        }
    }

    return 0;
}
