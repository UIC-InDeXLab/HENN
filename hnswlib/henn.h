#pragma once

#include <vector>
#include <random>
#include <unordered_set>
// #include <hnswlib/hnswlib.h>
#include "hnswlib.h"
#include <iostream>
#include <cmath>
#include <thread>

#include <mutex>
#include <atomic>
#include <exception>
#include <iterator>

using namespace std;
using namespace hnswlib;

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template <class Function>
inline void Parallel(size_t start, size_t end, size_t numThreads, Function fn)
{
    if (numThreads <= 0)
    {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1)
    {
        for (size_t id = start; id < end; id++)
        {
            fn(id, 0);
        }
    }
    else
    {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId)
        {
            threads.push_back(std::thread([&, threadId]
                                          {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                } }));
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        if (lastException)
        {
            std::rethrow_exception(lastException);
        }
    }
}

template <class Function>
inline void ParallelMap(const std::unordered_map<int, int> &data, size_t numThreads, Function fn)
{
    if (numThreads <= 0)
    {
        numThreads = std::thread::hardware_concurrency();
    }

    // Convert keys to a vector for indexed access
    std::vector<int> keys;
    keys.reserve(data.size());
    for (const auto &kv : data)
    {
        keys.push_back(kv.first);
    }

    size_t total = keys.size();

    if (numThreads == 1 || total < numThreads)
    {
        for (size_t i = 0; i < total; i++)
        {
            int key = keys[i];
            fn(key, data.at(key), 0);
        }
    }
    else
    {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(0);
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId)
        {
            threads.emplace_back([&, threadId]
                                 {
                while (true) {
                    size_t index = current.fetch_add(1);
                    if (index >= total) break;

                    int key = keys[index];
                    try {
                        fn(key, data.at(key), threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = total;
                        break;
                    }
                } });
        }

        for (auto &t : threads)
            t.join();

        if (lastException)
            std::rethrow_exception(lastException);
    }
}

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
        // std::mt19937 rng(std::random_device{}());
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
        int k = floor(log2(numPoints));
        k += 1;
        k = max(k, 4);
        // k = min(k, 5);
        // TODO: fine-tune
        return min(k, numPoints);
    }

    vector<float> getKthDistances(const float *ranges, int ranges_size, const float *points, int numPoints, int dim, int k, SpaceInterface<float> *space)
    {
        HierarchicalNSW<float> hnsw(space, numPoints);
        // for (int i = 0; i < numPoints; i++)
        // {
        //     cout << "Adding point " << i << " of " << numPoints << "\r";
        //     hnsw.addPoint(points + i * dim, i);
        // }
        cout << "Adding points to HNSW index..." << endl;
        Parallel(0, numPoints, 60, [&](size_t row, size_t threadId)
                 {
            size_t id = row;
            hnsw.addPoint((void *)(points + dim * row), (size_t)id); });

        // hnsw.saveIndex("henn_index.bin");
        hnsw.setEf(400);

        vector<float> kthDistances(ranges_size);
        for (int i = 0; i < ranges_size; ++i)
        {
            cout << "Searching for point " << i + 1 << " of " << ranges_size << "\r";
            auto knnResult = hnsw.searchKnn(ranges + i * dim, k);
            kthDistances[i] = knnResult.top().first;
        }
        cout << endl;

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

        cout << "k is set to " << k << " num points: " << numPoints << endl;

        int ranges_size = 800; // TODO: fine-tune
        float *ranges = getRanges(ranges_size, dim);

        cout << "Ranges size: " << ranges_size << endl;

        vector<float> kthDistances = getKthDistances(ranges, ranges_size, points, numPoints, dim, k, space);

        int max_hit = -1;
        int min_hit = ranges_size + 1;
        pair<vector<int>, float *> best_epsnet;
        pair<vector<int>, float *> worst_epsnet;

        for (size_t attempt = 0; attempt < maxTries; ++attempt)
        {
            cout << "Attempt " << attempt + 1 << " of " << maxTries << "\r" << flush;
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
        cout << endl;
        cout << "Max hits " << static_cast<float>(max_hit) / ranges_size << " Min hits: " << static_cast<float>(min_hit) / ranges_size << endl;

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

        cout << "L is set to " << L << endl;

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
                epsnet = getBestWorstSample(tmp, cur_layer, size, dim, 200, space, M).first;
            }
            else
            {
                epsnet = getBestWorstSample(tmp, cur_layer, size, dim, 200, space, M).second;
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

        // int count = 0;
        // for (const auto &[index, layer] : layers)
        // {
        //     count += 1;
        //     cout << "Adding to layer " << count << "\r" << flush;
        //     if (std::find(alreadyAdded.begin(), alreadyAdded.end(), index) != alreadyAdded.end())
        //     {
        //         continue;
        //     }
        //     alreadyAdded.push_back(index);
        //     hnsw->addPoint(points + index * dim, index, layer);
        // }
        cout << "Adding points to HENN index..." << endl;
        ParallelMap(layers, 60, [&](int index, int layer, size_t threadId)
                    {
            // if (std::find(alreadyAdded.begin(), alreadyAdded.end(), index) != alreadyAdded.end())
            // {
            //     return;
            // }
            // alreadyAdded.push_back(index);
            hnsw->addPoint(points + index * dim, index, layer); });

        cout << "Number of points in hnsw: " << hnsw->cur_element_count << endl;

        cout << endl;
        return hnsw;
    }

    /*
     *  Call this function to build the HENN Index. See examples/cpp/henn/henn_time_recall.cpp for usage.
     */
    HierarchicalNSW<float> *buildHENN(float *data, int size, int dim, SpaceInterface<float> *space, int M, bool best = true)
    {
        cout << "Building HENN..." << endl;
        auto henn_layers = buildBestWorstLayers(data, size, dim, space, M, best);
        auto henn = buildHENN(henn_layers, data, size, dim, space);
        cout << "HENN built." << endl;
        return henn;
    }
}