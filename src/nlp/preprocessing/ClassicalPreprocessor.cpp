#include "nlp/preprocessing/ClassicalPreprocessor.hpp"

#include <cctype>
#include <fstream>

ClassicalPreprocessor::ClassicalPreprocessor(
    const std::string &vocab_file, const std::string &idf_file){
        loadVocabulary(vocab_file);
        loadIDF(idf_file);
}

void ClassicalPreprocessor::loadVocabulary(const std::string &path){
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open vocab file");
    std::string token;
    size_t idx = 0;
    while (file >> token) {
        vocab[token] = idx++;
    }
}

void ClassicalPreprocessor::loadIDF(const std::string &path){
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open idf file");
    std::vector<float> values;
    float val;
    while (file >> val) {
        values.push_back(val);
    }
    idf = Eigen::Map<Eigen::VectorXf>(values.data(), values.size());
}

Eigen::VectorXf ClassicalPreprocessor::preprocessToVector(std::string text){
    lowercase(text);
    removePunctuation(text);
    
    std::vector<std::string> tokens = tokenize(text);
    removeStopwords(tokens);
    stem(tokens);

    std::vector<std::string> all_tokens = tokens;
    auto bigrams = generateNgrams(tokens, 2);
    all_tokens.insert(all_tokens.end(), bigrams.begin(), bigrams.end());

    return computeTFIDF(all_tokens);;
}

std::string ClassicalPreprocessor::preprocessToString(std::string text){
    lowercase(text);
    removePunctuation(text);
    
    std::vector<std::string> tokens = tokenize(text);
    removeStopwords(tokens);
    stem(tokens);

    return join(tokens);
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

std::string ClassicalPreprocessor::join(std::vector<std::string> &tokens){
    std::string res{};
    if (!tokens.empty()){
        for (auto token : tokens){
            res.append(token);
            res.append(" ");
        }
        res.pop_back();
    }
    return res;
}

std::vector<std::string> ClassicalPreprocessor::generateNgrams(
    const std::vector<std::string>& tokens, int n){
    std::vector<std::string> ngrams;
    if (tokens.size() < static_cast<size_t>(n)) return ngrams;

    for (size_t i = 0; i <= tokens.size() - n; ++i) {
        std::string gram = tokens[i];
        for (int j = 1; j < n; ++j) {
            gram += " " + tokens[i + j];
        }
        ngrams.push_back(gram);
    }
    return ngrams;
}

Eigen::VectorXf ClassicalPreprocessor::computeTFIDF(const std::vector<std::string> &tokens){
    if (vocab.empty() || idf.size() != vocab.size())
        return Eigen::VectorXf::Zero(vocab.size());
        
    Eigen::VectorXf tfidf = Eigen::VectorXf::Zero(vocab.size());
    std::unordered_map<size_t, int> term_count;

    for (auto &token : tokens) {
        auto it = vocab.find(token);
        if (it != vocab.end()) 
            term_count[it->second]++;
    }

    int total_terms = tokens.empty() ? 1 : static_cast<int>(tokens.size());

    for (auto &[idx, count] : term_count) {
        tfidf[idx] = (count / float(total_terms)) * idf[idx];
    }

    return tfidf;
}