#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static inline void sleek_search_error(const char* message)
{
  std::fprintf(stderr, "ERROR: %s\n", message);
  std::exit(EXIT_FAILURE);
}

static inline long double sleek_parse_budget(int& argc, char** argv)
{
  long double budget = -1;
  int positional = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--l2-budget") != 0) {
      argv[positional++] = argv[i];
      continue;
    }
    if (budget >= 0 || ++i == argc) sleek_search_error("expected one --l2-budget value");
    char* end;
    budget = std::strtold(argv[i], &end);
    if (*end || end == argv[i] || !std::isfinite(budget) || budget < 0)
      sleek_search_error("L2 budget must be finite and nonnegative");
  }
  argc = positional;
  argv[argc] = nullptr;
  return budget;
}

// A negative marker is distinct from the positive byte count in legacy files.
// These prefixes identify the version and scalar width before the native payload.
template <typename F>
static constexpr long long sleek_search_marker() { return -1000 - int(sizeof(F)); }

template <typename F>
static long sleek_read_search_prefix(FILE* file, long long& size, F& selected)
{
  if (!file || std::fread(&size, sizeof(size), 1, file) != 1)
    sleek_search_error("missing compressed header");
  if (size > 0) return 0;
  if (size != sleek_search_marker<F>() ||
      std::fread(&selected, sizeof(F), 1, file) != 1 ||
      !std::isnormal(selected) || selected <= 0 ||
      std::fread(&size, sizeof(size), 1, file) != 1 || size <= 0 || size % sizeof(F))
    sleek_search_error("invalid searched stream or wrong precision");
  return sizeof(long long) + sizeof(F);
}

template <typename F>
static long double sleek_search_cap(const char* arg, const F* input, long long count)
{
  char* end;
  long double cap = std::strtold(arg, &end);
  if (*end || end == arg || !std::isfinite(cap) || cap <= 0)
    sleek_search_error("L-inf epsilon must be positive and finite");
#ifdef ARTIFACT
  const auto range = std::minmax_element(input, input + count);
  cap *= static_cast<long double>(*range.second) - *range.first;
#endif
  return cap;
}

template <typename F, typename RoundTrip>
static void sleek_search_compress(const F* input, long long count, long double cap, long double budget,
                                  unsigned char* encoded, RoundTrip round_trip, const char* path)
{
  if (!std::isfinite(cap) || cap < 0) sleek_search_error("invalid absolute L-inf cap");
  // Limit trials to the exponent range supported by sleek_error_bound.
  constexpr int largest = std::numeric_limits<F>::max_exponent - std::numeric_limits<F>::digits - 1;
  F first = std::ldexp(F(1), largest);
  if (cap < std::ldexp(1.L, largest + 1)) {
    first = static_cast<F>(cap);
    if (first > cap) first = std::nextafter(first, F(0));
    first = std::max(first, std::numeric_limits<F>::min());
  }
  std::vector<F> decoded(count);
  std::vector<unsigned char> best;
  F selected = 0;
  // Scale by a power of two before squaring, including on systems where
  // long double has the same exponent range as double.
  int scale = 0;
  const long double fraction = std::frexp(budget, &scale);
  const long double limit = fraction * fraction;
  int trials = 0;
  auto passes = [&](F epsilon) {
    ++trials;
    long long size = round_trip(epsilon, decoded.data());
    long double sum = 0, correction = 0;
    for (long long i = 0; i < count; ++i) {
      if (input[i] == decoded[i] || (std::isnan(input[i]) && std::isnan(decoded[i]))) continue;
      long double error = std::fabs(static_cast<long double>(input[i]) - decoded[i]);
      if (!std::isfinite(error) || error > cap || error > budget) return false;
      const long double scaled = std::ldexp(error, -scale);
      const long double square = scaled * scaled;
      const long double next = sum + square;
      // Compensated summation retains small residuals alongside large ones.
      correction += sum >= square ? (sum - next) + square : (square - next) + sum;
      sum = next;
    }
    if (correction > limit - sum) return false;
    selected = epsilon;
    best.assign(encoded, encoded + size);
    return true;
  };

  if (!passes(first)) {
    // Only the exponent affects the codec. Bracket from the conservative
    // B/sqrt(N) estimate, then bisect distinct quantizer settings.
    int high = std::ilogb(first);
    const int minimum = std::numeric_limits<F>::min_exponent - 1;
    long double estimate = budget / std::sqrt(static_cast<long double>(count));
    int low = minimum;
    if (estimate >= std::numeric_limits<F>::min())
      low = std::max(minimum, std::min(high - 1, std::ilogb(estimate)));
    if (low == high || !passes(std::ldexp(F(1), low))) {
      high = low;
      if (low == minimum || !passes(std::numeric_limits<F>::min()))
        sleek_search_error("no passing epsilon found within supported precision");
      low = minimum;
    }
    while (high - low > 1) {
      int mid = low + (high - low) / 2;
      if (passes(std::ldexp(F(1), mid))) low = mid;
      else high = mid;
    }
  }

  const long long marker = sleek_search_marker<F>();
  FILE* out = std::fopen(path, "wb");
  if (!out || std::fwrite(&marker, sizeof(marker), 1, out) != 1 ||
      std::fwrite(&selected, sizeof(F), 1, out) != 1 ||
      std::fwrite(best.data(), 1, best.size(), out) != best.size() || std::fclose(out))
    sleek_search_error("could not write searched stream");
  const auto bytes = sizeof(marker) + sizeof(F) + best.size();
  std::printf("absolute L-inf cap: %.*Lg\n", std::numeric_limits<long double>::max_digits10, cap);
  std::printf("selected epsilon: %.17g\nsearch trials: %d\nencoded size: %zu bytes\n",
              double(selected), trials, bytes);
  std::printf("ratio: %.3fx\n", double(count) * sizeof(F) / bytes);
}
