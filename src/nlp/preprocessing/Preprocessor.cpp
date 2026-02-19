#include "preprocessing/Preprocessor.hpp"

#include <cctype>

std::vector<std::string> Preprocessor::preprocessFully(std::string text){
    lowercase(text);
    removePunctuation(text);
    
    std::vector<std::string> tokens = tokenize(text);

    removeStopwords(tokens);
    stem(tokens);

    return tokens;
}

void Preprocessor::lowercase(std::string &text){
    for (char& c : text)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

void Preprocessor::removePunctuation(std::string &text){
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

std::vector<std::string> Preprocessor::tokenize(std::string_view text){
    std::vector<std::string> tokens;
    tokens.reserve(10);

    unsigned int token_start = 0;
    size_t n = text.size();

    for (size_t i = 0; i < n; i++){
        if (std::isspace(static_cast<unsigned char>(text[i]))){
            if (token_start != i){
                tokens.push_back(std::string{text.substr(token_start, i - token_start)});
            }
            token_start = i + 1;
        }
    }

    if (token_start < n)
        tokens.push_back(std::string{text.substr(token_start, n)});

    return tokens;
}

void Preprocessor::removeStopwords(std::vector<std::string> &tokens){
    size_t current_index = 0;
    size_t n = tokens.size();

    for (size_t i = 0; i < n; i++){
        if (stopwords.find(tokens[i]) == stopwords.end()){
            tokens[current_index++] = std::move(tokens[i]);
        }
    }

    tokens.resize(current_index);
}

void Preprocessor::stem(std::vector<std::string> &tokens){

}