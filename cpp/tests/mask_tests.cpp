#include "mask.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

TEST_CASE("PackedMask constructs zero and all-ones masks across word boundaries") {
    PackedMask zero_0(0);
    REQUIRE(zero_0.width() == 0);
    REQUIRE(zero_0.words().empty());
    REQUIRE(zero_0.is_zero());

    PackedMask ones_5 = all_ones(5);
    REQUIRE(ones_5.width() == 5);
    REQUIRE(ones_5.words().size() == 1);
    REQUIRE(ones_5.words()[0] == 0x1f);

    PackedMask ones_64 = all_ones(64);
    REQUIRE(ones_64.words().size() == 1);
    REQUIRE(ones_64.words()[0] == ~std::uint64_t{0});

    PackedMask ones_65 = all_ones(65);
    REQUIRE(ones_65.words().size() == 2);
    REQUIRE(ones_65.words()[0] == ~std::uint64_t{0});
    REQUIRE(ones_65.words()[1] == 1);
}

TEST_CASE("PackedMask sets and tests bits across words") {
    PackedMask mask(130);
    mask.set(0);
    mask.set(63);
    mask.set(64);
    mask.set(129);

    REQUIRE(mask.test(0));
    REQUIRE(mask.test(63));
    REQUIRE(mask.test(64));
    REQUIRE(mask.test(129));
    REQUIRE_FALSE(mask.test(65));
    REQUIRE_THROWS_AS(mask.set(130), std::out_of_range);
    REQUIRE_THROWS_AS(mask.test(130), std::out_of_range);
    REQUIRE(mask.set_bit_indices() == std::vector<std::size_t>{0, 63, 64, 129});
}

TEST_CASE("PackedMask bitwise operations preserve width and trim high bits") {
    PackedMask left = single_bit(0, 65) | single_bit(64, 65);
    PackedMask right = single_bit(1, 65) | single_bit(64, 65);

    PackedMask both = left & right;
    REQUIRE(both.set_bit_indices() == std::vector<std::size_t>{64});

    PackedMask either = left | right;
    REQUIRE(either.set_bit_indices() == std::vector<std::size_t>{0, 1, 64});

    PackedMask xor_mask = left ^ right;
    REQUIRE(xor_mask.set_bit_indices() == std::vector<std::size_t>{0, 1});
    REQUIRE(xor_mask.words()[1] == 0);
}

TEST_CASE("PackedMask rejects bitwise operations with mismatched widths") {
    REQUIRE_THROWS(operator&(PackedMask(5), PackedMask(6)));
    REQUIRE_THROWS(operator|(PackedMask(5), PackedMask(6)));
    REQUIRE_THROWS(operator^(PackedMask(5), PackedMask(6)));
}
