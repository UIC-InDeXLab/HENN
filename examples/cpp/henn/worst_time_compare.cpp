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
    vector<double> henn_times;
    vector<double> hnsw_times;
    vector<int> logns;
    L2Space space(dim);
    for (int logn = min_logn; logn <= max_logn; logn++)
    {
        int size = pow(2, logn);
        double henn_time = 0;
        double hnsw_time = 0;
        for (int i = 0; i < repeat; i++)
        {
            cout << "Size: " << size << " repeat: " << i + 1 << endl;
            double tmp;
            float *data;
            if (i % 2 == 0) // cancel out the effect of cache
            {
                // HENN
                data = gen_points(size, dim);
                auto henn_layers = buildBestWorstLayers(data, size, dim, &space, M, true);
                auto henn = buildHENN(henn_layers, data, size, dim, &space);
                tmp = run_random_queries(henn, space, dim);
                henn_time = tmp > henn_time ? tmp : henn_time; // worst case time for HENN
                // HNSW
                data = gen_points(size, dim);
                auto hnsw_layers = buildBestWorstLayers(data, size, dim, &space, M, false);
                auto hnsw = buildHENN(hnsw_layers, data, size, dim, &space);
                tmp = run_random_queries(hnsw, space, dim);
                hnsw_time = tmp > hnsw_time ? tmp : hnsw_time; // worst case time for HNSW
            }
            else
            {
                // HNSW
                data = gen_points(size, dim);
                auto hnsw_layers = buildBestWorstLayers(data, size, dim, &space, M, false);
                auto hnsw = buildHENN(hnsw_layers, data, size, dim, &space);
                tmp = run_random_queries(hnsw, space, dim);
                hnsw_time = tmp > hnsw_time ? tmp : hnsw_time; // worst case time for HNSW
                // HENN
                data = gen_points(size, dim);
                auto henn_layers = buildBestWorstLayers(data, size, dim, &space, M, true);
                auto henn = buildHENN(henn_layers, data, size, dim, &space);
                tmp = run_random_queries(henn, space, dim);
                henn_time = tmp > henn_time ? tmp : henn_time; // worst case time for HENN
            }

            delete[] data;

            // cout << "HENN time: " << std::setprecision(7) << std::fixed << henn_time / (i + 1) << " seconds" << endl;
            // cout << "HNSW time: " << hnsw_time / (i + 1) << " seconds" << endl;
        }

        cout << "HENN time: " << std::setprecision(7) << std::fixed << henn_time << " seconds" << endl;
        cout << "HNSW time: " << hnsw_time << " seconds" << endl;

        henn_times.push_back(henn_time);
        hnsw_times.push_back(hnsw_time);
        logns.push_back(logn);
    }

    // Print CSV format of henn_times and hnsw_times
    cout << "\n\nCSV Output:\n";
    cout << "logn,dim,HENN,HNSW\n";
    for (size_t i = 0; i < logns.size(); ++i)
    {
        cout << logns[i] << "," << dim << "," << std::setprecision(7) << std::fixed << henn_times[i] << "," << hnsw_times[i] << "\n";
    }

    return 0;
}