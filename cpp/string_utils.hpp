#pragma once

#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

inline std::string_view trim_view(std::string_view value) noexcept {
    auto first = value.begin();
    while (first != value.end() && std::isspace(static_cast<unsigned char>(*first)) != 0) {
        ++first;
    }
    auto last = value.end();
    while (last != first && std::isspace(static_cast<unsigned char>(*(last - 1))) != 0) {
        --last;
    }
    return {first, last};
}

inline std::string trim_copy(std::string_view value) {
    const std::string_view trimmed = trim_view(value);
    return std::string(trimmed);
}

inline std::vector<std::string_view> split_view_preserving_empty(std::string_view value, char delimiter) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (true) {
        const std::size_t pos = value.find(delimiter, start);
        if (pos == std::string_view::npos) {
            parts.push_back(value.substr(start));
            break;
        }
        parts.push_back(value.substr(start, pos - start));
        start = pos + 1;
    }
    return parts;
}
