#include "../../../hnswlib/henn.h"

#include <iostream>
#include <random>
#include <chrono>
#include <iomanip> // For std::setprecision and std::fixed
#include <fstream>

using namespace std;
using namespace hnswlib;
using namespace henn;

float *gen_points(int size, int dim, int seed, int lambda)
{
    // std::mt19937 rng(std::random_device{}());
    mt19937 rng(seed);
    std::exponential_distribution<> distro(lambda); // or other distros
    // uniform_real_distribution<float> distro(0.0f, 1.0f); // Uniform distribution
    float *data = new float[dim * size];
    for (int i = 0; i < dim * size; i++)
    {
        data[i] = distro(rng);
    }
    return data;
}

float *get_queries(int ranges_size, int dim)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<> distro(0.0f, 1.0f); // or other distros

    float *ranges = new float[dim * ranges_size];

    for (int i = 0; i < ranges_size; ++i)
    {
        float value = static_cast<float>(i) / (ranges_size - 1);
        for (int j = 0; j < dim; ++j)
        {
            ranges[i * dim + j] = distro(rng);
        }
    }

    return ranges;
}

float l2_distance(const float *a, const float *b, size_t dim)
{
    float sum = 0;
    for (size_t i = 0; i < dim; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

vector<vector<float>> get_ground_truth(float *queries, int num_queries, float *points, int size, int dim, int k)
{
    // Brute-force
    cout << "Computing ground truth..." << endl;
    vector<vector<float>> ground_truth(num_queries);
    for (size_t i = 0; i < num_queries; ++i)
    {
        vector<pair<float, size_t>> dists;
        for (size_t j = 0; j < size; ++j)
        {
            float dist = l2_distance(queries + i * dim, points + j * dim, dim);
            dists.emplace_back(dist, j);
        }
        sort(dists.begin(), dists.end());
        for (int j = 0; j < k; ++j)
            ground_truth[i].push_back(dists[j].second);
    }
    cout << "Ground truth computed." << endl;
    return ground_truth;
}

float compute_recall(const vector<vector<float>> &gt, const vector<vector<float>> &ann, int k)
{
    assert(gt.size() == ann.size());
    size_t match = 0, total = gt.size() * k;
    for (size_t i = 0; i < gt.size(); ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (find(ann[i].begin(), ann[i].end(), gt[i][j]) != ann[i].end())
                match++;
        }
    }
    return float(match) / total;
}

HierarchicalNSW<float> *buildHENN(float *data, int size, int dim, SpaceInterface<float> *space)
{
    cout << "Building HNSW..." << endl;
    auto henn_layers = buildBestWorstLayers(data, size, dim, space, 4, false);
    auto henn = buildHENN(henn_layers, data, size, dim, space);
    cout << "HNSW built." << endl;
    return henn;
}

void write_to_file(string address, vector<tuple<int, double, float>> &time_recall)
{
    std::ofstream outfile(address);
    if (!outfile)
    {
        std::cerr << "Failed to open file for writing!" << std::endl;
    }

    outfile << "ef,Time,Recall" << std::endl;
    for (const auto &entry : time_recall)
    {
        auto [ef, time, recall] = entry;
        outfile << ef << "," << time << "," << recall << std::endl;
    }

    outfile.close();
    std::cout << "Data written to results.csv" << std::endl;
}

/*
 * Get time vs recall trends by varying 'ef' parameter.
 */
int main(int argc, char *argv[])
{
    int dim = 32;
    int size = 5000;
    int num_queries = 100;
    int k = 10;
    int repeats = 10;
    int lambda = atoi(argv[1]);

    vector<tuple<int, double, float>> time_recall;

    for (int r = 0; r < repeats; r++)
    {
        cout << "Repeat " << r + 1 << " of " << repeats << endl;
        float *data = gen_points(size, dim, r, lambda);
        float *queries = get_queries(num_queries, dim);
        HierarchicalNSW<float> hnsw(new L2Space(dim), size);

        for (int i = 0; i < size; i++)
        {
            hnsw.addPoint(data + i * dim, i);
        }

        // HierarchicalNSW<float> *hnsw = buildHENN(data, size, dim, new L2Space(dim));

        vector<vector<float>> ground_truth = get_ground_truth(queries, num_queries, data, size, dim, k);

        for (int ef = 10; ef <= 300; ef += 20)
        {
            hnsw.setEf(ef);
            auto start = chrono::high_resolution_clock::now();

            vector<vector<float>> ann_results(num_queries);
            for (int j = 0; j < num_queries; j++)
            {
                auto res = hnsw.searchKnn(queries + j * dim, k);
                while (!res.empty())
                {
                    ann_results[j].push_back(res.top().second);
                    res.pop();
                }
            }

            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            auto time = chrono::duration_cast<chrono::microseconds>(elapsed).count() / num_queries;

            float recall = compute_recall(ground_truth, ann_results, k);

            cout << "ef = " << ef
                 << " | Avg Time per Query = " << time << " microseconds"
                 << " | Recall@10 = " << recall << endl;

            time_recall.push_back({ef, time, recall});
        }
    }

    std::string filename = "../examples/reports/time_recall_hnsw_lambda=" + std::to_string(lambda) + ".csv";
    write_to_file(filename, time_recall);
    return 0;
}
