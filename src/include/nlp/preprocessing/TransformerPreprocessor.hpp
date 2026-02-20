#pragma once

#include "IPreprocessor.hpp"
#include <string>
#include <vector>
#include <memory>

class TransformerPreprocessor : public IPreprocessor{
public:
    std::vector<std::string> preprocess(std::string text) override;
};