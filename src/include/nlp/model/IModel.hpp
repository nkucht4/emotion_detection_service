#pragma once

#include <vector>
#include <string>

class IModel{
public:
    virtual ~IModel() = default;

    virtual void load(const std::string& path) = 0;

    virtual std::vector<float> predict(const std::string& input_text) = 0;
};