#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>

class IPreprocessor{
public:
    virtual ~IPreprocessor() = default;

    virtual std::string preprocessToString(std::string text) = 0;

    virtual Eigen::VectorXf preprocessToVector(std::string text) = 0;
};