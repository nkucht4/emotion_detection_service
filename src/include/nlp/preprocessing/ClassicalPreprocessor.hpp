#pragma once

#include "IPreprocessor.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>

class ClassicalPreprocessor : public IPreprocessor {
public:
    std::vector<std::string> preprocess(std::string text) override;

private:
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

    void lowercase(std::string &text);

    void removePunctuation(std::string &text);

    std::vector<std::string> tokenize(std::string_view text);

    void removeStopwords(std::vector<std::string> &tokens);

    void stem(std::vector<std::string> &tokens);
};