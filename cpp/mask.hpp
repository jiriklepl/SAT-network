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
    bool is_zero() const;
    bool test(std::size_t idx) const;
    void set(std::size_t idx);
    std::vector<std::size_t> set_bit_indices() const;

    friend bool operator==(const PackedMask &left, const PackedMask &right);
    friend bool operator!=(const PackedMask &left, const PackedMask &right);
    friend PackedMask operator&(const PackedMask &left, const PackedMask &right);
    friend PackedMask operator|(const PackedMask &left, const PackedMask &right);
    friend PackedMask operator^(const PackedMask &left, const PackedMask &right);

private:
    unsigned width_ = 0;
    std::vector<std::uint64_t> words_;

    void trim_unused_bits();
};

PackedMask all_ones(unsigned width);
PackedMask single_bit(std::size_t idx, unsigned width);
