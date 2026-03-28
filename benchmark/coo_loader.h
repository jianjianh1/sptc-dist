#ifndef SPTC_COO_LOADER_H
#define SPTC_COO_LOADER_H

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/// Sparse tensor in COO (coordinate) format, up to rank 4.
struct COOTensor {
  int rank = 0;
  std::vector<std::array<std::size_t, 4>> indices;
  std::vector<double> values;
  std::array<std::size_t, 4> shape = {0, 0, 0, 0};  // max_index + 1 per dim
};

/// Load a sparse tensor from a space/tab-separated text file.
/// Each line: idx0 idx1 ... idxN value
/// Rank is auto-detected from the first non-empty line.
inline COOTensor load_coo(const std::string& filename) {
  COOTensor tensor;
  std::ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "ERROR: cannot open " << filename << "\n";
    return tensor;
  }

  std::string line;
  bool first = true;

  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);

    // On first line, detect rank by counting tokens
    if (first) {
      std::vector<std::string> tokens;
      std::string tok;
      while (iss >> tok) tokens.push_back(tok);
      tensor.rank = static_cast<int>(tokens.size()) - 1;  // last token is value
      if (tensor.rank < 1 || tensor.rank > 4) {
        std::cerr << "ERROR: unsupported rank " << tensor.rank << " in "
                  << filename << "\n";
        return tensor;
      }

      std::array<std::size_t, 4> idx = {0, 0, 0, 0};
      for (int d = 0; d < tensor.rank; ++d) {
        idx[d] = std::stoull(tokens[d]);
        if (idx[d] + 1 > tensor.shape[d]) tensor.shape[d] = idx[d] + 1;
      }
      double val = std::stod(tokens[tensor.rank]);
      tensor.indices.push_back(idx);
      tensor.values.push_back(val);
      first = false;
      continue;
    }

    std::array<std::size_t, 4> idx = {0, 0, 0, 0};
    for (int d = 0; d < tensor.rank; ++d) {
      iss >> idx[d];
      if (idx[d] + 1 > tensor.shape[d]) tensor.shape[d] = idx[d] + 1;
    }
    double val;
    iss >> val;
    tensor.indices.push_back(idx);
    tensor.values.push_back(val);
  }

  return tensor;
}

#endif  // SPTC_COO_LOADER_H
