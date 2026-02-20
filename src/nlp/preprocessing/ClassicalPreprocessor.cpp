#include "nlp/preprocessing/ClassicalPreprocessor.hpp"

#include <cctype>

std::vector<std::string> ClassicalPreprocessor::preprocess(std::string text){
    lowercase(text);
    removePunctuation(text);
    
    std::vector<std::string> tokens = tokenize(text);

    removeStopwords(tokens);
    stem(tokens);

    return tokens;
}

void ClassicalPreprocessor::lowercase(std::string &text){
    for (char& c : text)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

void ClassicalPreprocessor::removePunctuation(std::string &text){
    std::string result;
    result.reserve(text.size());

    for (unsigned char c : text){
        if (std::ispunct(c) && c != '!' && c != '?'){
            continue;
        }
        result.push_back(static_cast<char>(c));
    }

    text = std::move(result);
}

std::vector<std::string> ClassicalPreprocessor::tokenize(std::string_view text){
    std::vector<std::string> tokens;
    size_t start = 0;
    const size_t n = text.size();

    while (start < n){
        while (start < n && std::isspace(static_cast<unsigned char>(text[start]))) start++;
        if (start >= n) break;

        size_t end = start;
        while (end < n && !std::isspace(static_cast<unsigned char>(text[end]))) end++;

        std::string token{text.substr(start, end - start)};
        
        if (!token.empty()) {
            tokens.push_back(token);
        }

        start = end;
    }

    return tokens;
}

void ClassicalPreprocessor::removeStopwords(std::vector<std::string> &tokens){
    std::vector<std::string> clean;
    clean.reserve(tokens.size());

    for (auto &token : tokens) {
        if (token.empty()) continue;

        if (stopwords.find(token) == stopwords.end()) {
            clean.push_back(token);
        }
    }
    tokens = std::move(clean);
}

void ClassicalPreprocessor::stem(std::vector<std::string> &tokens) {
    std::vector<std::string> stemmed;
    stemmed.reserve(tokens.size());

    for (std::string &token : tokens){
        if (token.size() <= 3) {
            stemmed.push_back(token);
            continue;
        }

        bool suffix_removed = false;

        for (const auto &suffix : suffixes){
            if (token.size() > suffix.size() + 1){ 
                if (token.compare(token.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    token.erase(token.size() - suffix.size());
                    suffix_removed = true;
                    break; 
                }
            }
        }

        stemmed.push_back(token);
    }

    tokens = std::move(stemmed);
}
