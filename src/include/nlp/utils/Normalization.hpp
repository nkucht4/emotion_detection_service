#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

inline std::vector<float> softmax(const std::vector<float>& logits){
    if (logits.empty()) return {};

    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exps;
    exps.reserve(logits.size());
    float sum = 0.0f;

    for (float l : logits) {
        float e = std::exp(l - max_logit);
        exps.push_back(e);
        sum += e;
    }

    for (auto& e : exps) e /= sum;

    return exps;
}