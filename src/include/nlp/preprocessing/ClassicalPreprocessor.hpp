#pragma once

#include "IPreprocessor.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>

class ClassicalPreprocessor : public IPreprocessor{
public:
    ClassicalPreprocessor(const std::string &vocab_file, const std::string &idf_file);

    std::string preprocessToString(std::string text) override;

    Eigen::VectorXf preprocessToVector(std::string text) override;

private:
    std::unordered_map<std::string, size_t> vocab;
    Eigen::VectorXf idf; 

    inline static const std::unordered_set<std::string> stopwords{
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "with", "to", "of",
        "for", "at", "by", "from", "up", "down", "out", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "can", "will", "just", "should", "now", "am", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "he", "she", "it", "they", "them", "his", "her", "its", "their", "you",
        "your", "i", "me", "my", "we", "us", "our"
    };

    inline static const std::unordered_set<std::string> suffixes{
        "ing", "ed", "ly", "s", "es", "er", "est", "ment", "ness", "ful", "less", "able"
    };

    void loadVocabulary(const std::string &path);
    void loadIDF(const std::string &path);

    void lowercase(std::string &text);
    void removePunctuation(std::string &text);
    std::vector<std::string> tokenize(std::string_view text);
    void removeStopwords(std::vector<std::string> &tokens);
    void stem(std::vector<std::string> &tokens);
    std::string join(std::vector<std::string> &tokens);

    std::vector<std::string> generateNgrams(const std::vector<std::string> &tokens, int n);

    Eigen::VectorXf computeTFIDF(const std::vector<std::string> &tokens);
};