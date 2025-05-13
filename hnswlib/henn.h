#pragma once

#include <vector>
#include <random>
#include <unordered_set>
#include <hnswlib/hnswlib.h>
#include <iostream>
#include <cmath>

using namespace std;
using namespace hnswlib;

/*
 * Search for "TODO: fine-tune" in this file to find the parameters that can be fine-tuned.
 */

namespace henn
{
    typedef unsigned int tableint;

    /*
     *  Draw random samples with replacement.
     */
    std::pair<std::vector<int>, float *> sample(
        const float *points,
        const std::vector<int> cur_indices,
        size_t numPoints,
        size_t dim,
        size_t sampleSize)
    {
        if (numPoints == 0 || sampleSize == 0)
        {
            return {{}, nullptr};
        }

        std::vector<int> indices(sampleSize);

        float *sampledFlatArray = new float[sampleSize * dim];

        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, numPoints - 1);

        for (size_t i = 0; i < sampleSize; ++i)
        {
            size_t idx = dist(gen); // Random index
            for (size_t d = 0; d < dim; ++d)
            {
                sampledFlatArray[i * dim + d] = points[idx * dim + d];
                indices[i] = cur_indices[idx];
            }
        }

        return {indices, sampledFlatArray};
    }

    /*
     *  It generates a set of random points in the range [0, 1] and returns them.
     */
    float *getRanges(int ranges_size, int dim)
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

    /*
     *  Get the value k which is important during the processing for finding an approximated epsilon net.
     *  Fine-tune this value to get a good epsilon net.
     */
    int getK(int numPoints)
    {
        int k = floor(log2(numPoints)) / 4;
        k += 3;
        k = max(k, 2);
        // TODO: fine-tune
        return min(k, numPoints);
    }

    vector<float> getKthDistances(const float *ranges, int ranges_size, const float *points, int numPoints, int dim, int k, SpaceInterface<float> *space)
    {
        HierarchicalNSW<float> hnsw(space, numPoints);
        for (int i = 0; i < numPoints; i++)
        {
            hnsw.addPoint(points + i * dim, i);
        }

        vector<float> kthDistances(ranges_size);
        for (int i = 0; i < ranges_size; ++i)
        {
            auto knnResult = hnsw.searchKnn(ranges + i * dim, k);
            kthDistances[i] = knnResult.top().first;
        }

        return kthDistances;
    }

    /*
     * return {best_pair, worst_pair}. The best pair is the best epsilon net found after maxTries.
     * the worst pair is the worst sample found after maxTries.
     */
    pair<pair<vector<int>, float *>, pair<vector<int>, float *>> getBestWorstSample(
        const float *points,
        const vector<int> indices,
        int numPoints,
        int dim,
        int maxTries,
        SpaceInterface<float> *space,
        int M = 1)
    {
        if (numPoints == 0)
        {
            throw std::invalid_argument("Point set or k cannot be zero.");
        }

        int k = getK(numPoints);

        // cout << "k is set to " << k << " num points: " << numPoints << endl;

        int ranges_size = 400; // TODO: fine-tune
        float *ranges = getRanges(ranges_size, dim);

        // cout << "Ranges size: " << ranges_size << endl;

        vector<float> kthDistances = getKthDistances(ranges, ranges_size, points, numPoints, dim, k, space);

        int max_hit = -1;
        int min_hit = ranges_size + 1;
        pair<vector<int>, float *> best_epsnet;
        pair<vector<int>, float *> worst_epsnet;

        for (size_t attempt = 0; attempt < maxTries; ++attempt)
        {
            size_t sampleSize = numPoints / pow(2, M);
            auto smpl = sample(points, indices, numPoints, dim, sampleSize);
            float *epsilonNet = smpl.second;

            int hits = 0;
            for (size_t i = 0; i < ranges_size; ++i)
            {
                bool hit = false;
                for (size_t j = 0; j < sampleSize; ++j)
                {
                    float distToSample = space->get_dist_func()(
                        ranges + i * dim, epsilonNet + j * dim, space->get_dist_func_param());
                    if (distToSample <= kthDistances[i])
                    {
                        hit = true;
                        break;
                    }
                }

                if (hit)
                {
                    hits++;
                }
            }

            if (hits <= max_hit && hits >= min_hit)
            {
                delete[] epsilonNet;
            }

            if (hits > max_hit)
            {
                max_hit = hits;
                best_epsnet = smpl;
            }
            if (hits < min_hit)
            {
                min_hit = hits;
                worst_epsnet = smpl;
            }
        }
        // cout << "Max hits " << static_cast<float>(max_hit) / ranges_size << " Min hits: " << static_cast<float>(min_hit) / ranges_size << endl;

        return {best_epsnet, worst_epsnet};
    }

    /*
     *  It build a hierarchy of layers. If isBest is true, it builds the best found epsilon net at each layer.
     *  If isBest is false, it builds the worst case random sample found at each layer.
     */
    unordered_map<int, int> buildBestWorstLayers(
        const float *points,
        size_t numPoints,
        size_t dim,
        SpaceInterface<float> *space,
        size_t M = 1,
        bool isBest = true)
    {
        // TODO: fine-tune M
        int L = static_cast<int>(floor(log2(numPoints) / M));

        unordered_map<int, int> indexToLayer;

        vector<int> cur_layer(numPoints);
        iota(cur_layer.begin(), cur_layer.end(), 0);

        for (int i = 0; i < numPoints; i++)
        {
            indexToLayer[i] = 0;
        }

        auto tmp = points;
        int size = numPoints;
        pair<vector<int>, float *> epsnet;
        for (int i = 1; i <= L; i++)
        {
            if (isBest)
            {
                epsnet = getBestWorstSample(tmp, cur_layer, size, dim, 10000, space, M).first;
            }
            else
            {
                epsnet = getBestWorstSample(tmp, cur_layer, size, dim, 10000, space, M).second;
            }

            tmp = epsnet.second;
            size = size / pow(2, M);
            cur_layer = epsnet.first;
            for (int j = 0; j < epsnet.first.size(); j++)
            {
                indexToLayer[epsnet.first[j]] = i;
            }
        }
        return indexToLayer;
    }

    /*
     *  Given a set of layers, it generates an HierarchicalNSW object, adding each point to the corresponding layer.
     */
    template <typename dist_t>
    hnswlib::HierarchicalNSW<dist_t> *buildHENN(
        std::unordered_map<int, int> layers,
        float *points,
        size_t numPoints,
        size_t dim,
        hnswlib::SpaceInterface<dist_t> *space)
    {
        std::vector<int> alreadyAdded;
        hnswlib::HierarchicalNSW<dist_t> *hnsw = new hnswlib::HierarchicalNSW<dist_t>(space, numPoints);
        for (const auto &[index, layer] : layers)
        {
            if (std::find(alreadyAdded.begin(), alreadyAdded.end(), index) != alreadyAdded.end())
            {
                continue;
            }
            alreadyAdded.push_back(index);
            hnsw->addPoint(points + index * dim, index, layer);
        }
        return hnsw;
    }

    /*
     *  Call this function to build the HENN Index. See examples/cpp/henn/henn_time_recall.cpp for usage.
     */
    HierarchicalNSW<float> *buildHENN(float *data, int size, int dim, SpaceInterface<float> *space, int M)
    {
        cout << "Building HENN..." << endl;
        auto henn_layers = buildBestWorstLayers(data, size, dim, space, M, true);
        auto henn = buildHENN(henn_layers, data, size, dim, space);
        cout << "HENN built." << endl;
        return henn;
    }
}