#include "../../../hnswlib/henn.h"

#include <iostream>
#include <random>
#include <chrono>
#include <iomanip> // For std::setprecision and std::fixed

using namespace std;
using namespace hnswlib;
using namespace henn;

float *gen_points(int size, int dim)
{
    std::mt19937 rng(std::random_device{}());
    std::exponential_distribution<> distro(200); // or other distros
    float *data = new float[dim * size];
    for (int i = 0; i < dim * size; i++)
    {
        data[i] = distro(rng);
    }
    return data;
}

double run_random_queries(HierarchicalNSW<float> *hnsw, L2Space &space, int dim)
{
    int queries_size = 10;

    float *queries = getRanges(queries_size, dim);

    double time = 0;
    for (int i = 0; i < queries_size; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto knnResult = hnsw->searchKnn(queries + i * dim, 1, nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        time += elapsed.count();
    }

    delete[] queries;

    return time / queries_size;
}

/*
 *   arg1: min logn
 *   arg2: max logn
 *   arg3: dim
 */
int main(int argc, char *argv[])
{
    // default values
    int min_logn = stoi(argv[1]);
    int max_logn = stoi(argv[2]);
    int dim = stoi(argv[3]);

    int repeat = 4;
    int M = 4; // 16
    vector<long> henn_hops;
    vector<long> henn_dists;
    vector<long> hnsw_hops;
    vector<long> hnsw_dists;
    vector<int> logns;
    L2Space space(dim);
    for (int logn = min_logn; logn <= max_logn; logn++)
    {
        int size = pow(2, logn);
        long henn_hop = 0;
        long henn_dist = 0;
        long hnsw_hop = 0;
        long hnsw_dist = 0;
        for (int i = 0; i < repeat; i++)
        {
            cout << "Size: " << size << " repeat: " << i + 1 << endl;
            long tmp_hop;
            long tmp_dist;
            float *data;
            if (i % 2 == 0) // cancel out the effect of cache
            {
                // HENN
                data = gen_points(size, dim);
                auto henn_layers = buildBestWorstLayers(data, size, dim, &space, M, true);
                auto henn = buildHENN(henn_layers, data, size, dim, &space);
                run_random_queries(henn, space, dim);
                tmp_hop = henn->metric_hops;
                tmp_dist = henn->metric_distance_computations;
                henn_hop = tmp_hop > henn_hop ? tmp_hop : henn_hop;      // worst case time for HENN
                henn_dist = tmp_dist > henn_dist ? tmp_dist : henn_dist; // worst case time for HENN

                // HNSW
                data = gen_points(size, dim);
                auto hnsw_layers = buildBestWorstLayers(data, size, dim, &space, M, false);
                auto hnsw = buildHENN(hnsw_layers, data, size, dim, &space);
                run_random_queries(hnsw, space, dim);
                tmp_hop = hnsw->metric_hops;
                tmp_dist = hnsw->metric_distance_computations;
                hnsw_hop = tmp_hop > hnsw_hop ? tmp_hop : hnsw_hop;      // worst case time for HNSW
                hnsw_dist = tmp_dist > hnsw_dist ? tmp_dist : hnsw_dist; // worst case time for HNSW
            }
            else
            {
                // HNSW
                data = gen_points(size, dim);
                auto hnsw_layers = buildBestWorstLayers(data, size, dim, &space, M, false);
                auto hnsw = buildHENN(hnsw_layers, data, size, dim, &space);
                run_random_queries(hnsw, space, dim);
                tmp_hop = hnsw->metric_hops;
                tmp_dist = hnsw->metric_distance_computations;
                hnsw_hop = tmp_hop > hnsw_hop ? tmp_hop : hnsw_hop;      // worst case time for HNSW
                hnsw_dist = tmp_dist > hnsw_dist ? tmp_dist : hnsw_dist; // worst case time for HNSW

                // HENN
                data = gen_points(size, dim);
                auto henn_layers = buildBestWorstLayers(data, size, dim, &space, M, true);
                auto henn = buildHENN(henn_layers, data, size, dim, &space);
                run_random_queries(henn, space, dim);
                tmp_hop = henn->metric_hops;
                tmp_dist = henn->metric_distance_computations;
                henn_hop = tmp_hop > henn_hop ? tmp_hop : henn_hop;      // worst case time for HENN
                henn_dist = tmp_dist > henn_dist ? tmp_dist : henn_dist; // worst case time for HENN
            }

            delete[] data;

            // cout << "HENN time: " << std::setprecision(7) << std::fixed << henn_time / (i + 1) << " seconds" << endl;
            // cout << "HNSW time: " << hnsw_time / (i + 1) << " seconds" << endl;
        }

        cout << "HENN hops: " << henn_hop << "-" << henn_dist << " seconds" << endl;
        cout << "HNSW hops: " << hnsw_hop << "-" << hnsw_dist << " seconds" << endl;

        henn_hops.push_back(henn_hop);
        henn_dists.push_back(henn_dist);
        hnsw_hops.push_back(hnsw_hop);
        hnsw_dists.push_back(hnsw_dist);
        logns.push_back(logn);
    }

    // Print CSV format of henn_times and hnsw_times
    cout << "\n\nCSV Output:\n";
    cout << "logn,dim,HENN_hop,HNSW_hop,HENN_dist,HNSW_dist\n";
    for (size_t i = 0; i < logns.size(); ++i)
    {
        cout << logns[i] << "," << dim << "," << std::setprecision(7) << std::fixed << henn_hops[i] << "," << hnsw_hops[i] << "," << henn_dists[i] << "," << hnsw_dists[i] << "\n";
    }

    return 0;
}