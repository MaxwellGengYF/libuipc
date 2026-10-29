#pragma once
/********************************************************************
 * @file bench_stats.h
 * @brief Small statistics helpers for §7.1 (median / P95 / agreement).
 *********************************************************************/
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace uipc::benchmark
{
/// Linear-interpolated percentile of an already sorted or unsorted sample.
inline double percentile(std::vector<double> sample, double p)
{
    if(sample.empty())
        return 0.0;
    std::sort(sample.begin(), sample.end());
    if(sample.size() == 1u)
        return sample.front();
    const double rank = p / 100.0 * static_cast<double>(sample.size() - 1u);
    const auto   lo   = static_cast<std::size_t>(std::floor(rank));
    const auto   hi   = static_cast<std::size_t>(std::ceil(rank));
    const double t    = rank - static_cast<double>(lo);
    return sample[lo] * (1.0 - t) + sample[hi] * t;
}

inline double median(std::vector<double> sample)
{
    return percentile(std::move(sample), 50.0);
}

/// Coefficient of variation in percent (agreement check of §7.2: > 3 % = unstable).
inline double cov_percent(const std::vector<double>& sample)
{
    if(sample.size() < 2u)
        return 0.0;
    double sum = 0.0;
    for(double v : sample)
        sum += v;
    const double mean = sum / static_cast<double>(sample.size());
    if(mean == 0.0)
        return 0.0;
    double acc = 0.0;
    for(double v : sample)
        acc += (v - mean) * (v - mean);
    const double sd = std::sqrt(acc / static_cast<double>(sample.size() - 1u));
    return sd / mean * 100.0;
}
}// namespace uipc::benchmark
