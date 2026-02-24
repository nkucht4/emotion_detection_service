#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

struct ServerConfig{
    std::string host;
    uint16_t port;
};

struct NLPConfig{
    std::string vocab_path;
    std::string idf_path;
    std::string model_path;
    std::vector<std::string> labels;
};

class ConfigReader{
public:
    static ServerConfig readServerConfig(const std::string& path){
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot open server config file: " + path);
        nlohmann::json j; file >> j;
        return ServerConfig{ j.at("host").get<std::string>(), 
                             static_cast<uint16_t>(std::stoi(j.at("port").get<std::string>())) };
    }

    static NLPConfig readNLPConfig(const std::string& path){
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Cannot open NLP config file: " + path);
        nlohmann::json j; file >> j;
        return NLPConfig{
            j.at("vocab_path").get<std::string>(),
            j.at("idf_path").get<std::string>(),
            j.at("model_path").get<std::string>(),
            j.at("labels").get<std::vector<std::string>>()
        };
    }
};