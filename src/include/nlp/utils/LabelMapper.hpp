#pragma once
#include <vector>
#include <string>
#include <unordered_map>

class LabelMapper {
private:
    std::vector<std::string> labels;

public:
    LabelMapper(const std::vector<std::string>& labels_) : labels(labels_) {}

    std::unordered_map<std::string, float> map(const std::vector<float>& scores) const{
        std::unordered_map<std::string, float> result;
        if (scores.size() != labels.size())
            throw std::runtime_error("Score vector size does not match number of labels");

        for (size_t i = 0; i < labels.size(); ++i) {
            result[labels[i]] = scores[i];
        }
        return result;
    }
};