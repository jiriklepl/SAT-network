#include "mask.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <bit>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

std::size_t word_count(unsigned width) {
    return (static_cast<std::size_t>(width) + 63) / 64;
}

std::uint64_t high_word_mask(unsigned width) {
    const unsigned used = width % 64;
    if (used == 0) return ~std::uint64_t{0};
    return (std::uint64_t{1} << used) - 1;
}

void require_same_width(const PackedMask &left, const PackedMask &right) {
    if (left.width() != right.width()) {
        throw std::runtime_error("PackedMask width mismatch");
    }
}

}  // namespace

PackedMask::PackedMask(unsigned width) : width_(width), words_(word_count(width), 0) {}

PackedMask::PackedMask(unsigned width, std::vector<std::uint64_t> words) : width_(width), words_(std::move(words)) {
    if (words_.size() != word_count(width_)) {
        throw std::runtime_error("PackedMask word count does not match width");
    }
    trim_unused_bits();
}

unsigned PackedMask::width() const {
    return width_;
}

std::span<const std::uint64_t> PackedMask::words() const {
    return words_;
}

bool PackedMask::is_zero() const {
    return std::ranges::all_of(words_, [](std::uint64_t word) { return word == 0; });
}

bool PackedMask::test(std::size_t idx) const {
    if (idx >= width_) {
        throw std::out_of_range("PackedMask bit index out of range");
    }
    return ((words_[idx / 64] >> (idx % 64)) & std::uint64_t{1}) != 0;
}

void PackedMask::set(std::size_t idx) {
    if (idx >= width_) {
        throw std::out_of_range("PackedMask bit index out of range");
    }
    words_[idx / 64] |= std::uint64_t{1} << (idx % 64);
}

std::vector<std::size_t> PackedMask::set_bit_indices() const {
    std::vector<std::size_t> indices;
    for (std::size_t word_idx = 0; word_idx < words_.size(); ++word_idx) {
        std::uint64_t word = words_[word_idx];
        while (word != 0) {
            const unsigned bit = std::countr_zero(word);
            const std::size_t idx = word_idx * 64 + bit;
            if (idx < width_) indices.push_back(idx);
            word &= word - 1;
        }
    }
    return indices;
}

void PackedMask::trim_unused_bits() {
    if (!words_.empty()) {
        words_.back() &= high_word_mask(width_);
    }
}

bool operator==(const PackedMask &left, const PackedMask &right) {
    return left.width_ == right.width_ && left.words_ == right.words_;
}

bool operator!=(const PackedMask &left, const PackedMask &right) {
    return !(left == right);
}

PackedMask operator&(const PackedMask &left, const PackedMask &right) {
    require_same_width(left, right);
    PackedMask result(left.width_);
    for (std::size_t idx = 0; idx < result.words_.size(); ++idx) {
        result.words_[idx] = left.words_[idx] & right.words_[idx];
    }
    result.trim_unused_bits();
    return result;
}

PackedMask operator|(const PackedMask &left, const PackedMask &right) {
    require_same_width(left, right);
    PackedMask result(left.width_);
    for (std::size_t idx = 0; idx < result.words_.size(); ++idx) {
        result.words_[idx] = left.words_[idx] | right.words_[idx];
    }
    result.trim_unused_bits();
    return result;
}

PackedMask operator^(const PackedMask &left, const PackedMask &right) {
    require_same_width(left, right);
    PackedMask result(left.width_);
    for (std::size_t idx = 0; idx < result.words_.size(); ++idx) {
        result.words_[idx] = left.words_[idx] ^ right.words_[idx];
    }
    result.trim_unused_bits();
    return result;
}

PackedMask all_ones(unsigned width) {
    const PackedMask result(width);
    std::vector<std::uint64_t> words(word_count(width), ~std::uint64_t{0});
    return {width, std::move(words)};
}

PackedMask single_bit(std::size_t idx, unsigned width) {
    PackedMask result(width);
    result.set(idx);
    return result;
}
