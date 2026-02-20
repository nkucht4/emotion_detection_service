#pragma once
#include <string>
#include <vector>

class IPreprocessor {
public:
    virtual ~IPreprocessor() = default;

    virtual std::vector<std::string> preprocess(std::string text) = 0;
};