#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

class PackedMask {
public:
    PackedMask() = default;
    explicit PackedMask(unsigned width);
    PackedMask(unsigned width, std::vector<std::uint64_t> words);

    unsigned width() const;
    std::span<const std::uint64_t> words() const;
    std::uint64_t word(std::size_t word_idx) const noexcept;
    bool is_zero() const;
    std::size_t count() const;
    bool test(std::size_t idx) const;
    void set(std::size_t idx);
    void push_back(bool bit);
    std::vector<std::size_t> set_bit_indices() const;
    PackedMask slice(std::size_t start, std::size_t count) const;

    PackedMask operator~() const;
    PackedMask &operator&=(const PackedMask &other);
    PackedMask &operator|=(const PackedMask &other);
    PackedMask &operator^=(const PackedMask &other);

    friend bool operator==(const PackedMask &left, const PackedMask &right);
    friend bool operator!=(const PackedMask &left, const PackedMask &right);
    friend PackedMask operator&(PackedMask left, const PackedMask &right);
    friend PackedMask operator|(PackedMask left, const PackedMask &right);
    friend PackedMask operator^(PackedMask left, const PackedMask &right);

private:
    unsigned width_ = 0;
    std::vector<std::uint64_t> words_;

    void trim_unused_bits();
};

PackedMask all_ones(unsigned width);
PackedMask single_bit(std::size_t idx, unsigned width);
