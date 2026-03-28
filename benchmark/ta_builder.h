#ifndef SPTC_TA_BUILDER_H
#define SPTC_TA_BUILDER_H

#include <tiledarray.h>

#include <cmath>
#include <iostream>
#include <map>
#include <vector>

#include "coo_loader.h"

/// Build a TiledRange1 with uniform tile sizes for a dimension of given extent.
inline TA::TiledRange1 make_tr1(std::size_t extent, std::size_t tile_size) {
  std::vector<std::size_t> boundaries;
  for (std::size_t i = 0; i <= extent; i += tile_size)
    boundaries.push_back(i);
  if (boundaries.back() != extent) boundaries.push_back(extent);
  return TA::TiledRange1(boundaries.begin(), boundaries.end());
}

/// Build a TiledRange from per-dimension extents and tile sizes.
inline TA::TiledRange make_trange(const std::vector<std::size_t>& shape,
                                  const std::vector<std::size_t>& tile_sizes) {
  std::vector<TA::TiledRange1> tr1s;
  for (std::size_t d = 0; d < shape.size(); ++d)
    tr1s.push_back(make_tr1(shape[d], tile_sizes[d]));
  return TA::TiledRange(tr1s.begin(), tr1s.end());
}

/// Convert a COOTensor into a distributed TiledArray sparse array.
///
/// Three-pass algorithm:
///   Pass 1: Scan all entries, compute tile norms (all ranks, for SparseShape).
///   Pass 2: Create sparse shape and array (determines tile ownership).
///   Pass 3: Re-scan entries, only group entries for LOCAL tiles, set tiles.
inline TA::TSpArrayD build_sparse_array(TA::World& world,
                                        const COOTensor& coo,
                                        const std::vector<std::size_t>& tile_sizes,
                                        const std::string& label = "") {
  std::vector<std::size_t> shape(coo.shape.begin(),
                                 coo.shape.begin() + coo.rank);
  TA::TiledRange trange = make_trange(shape, tile_sizes);
  const auto& tiles_range = trange.tiles_range();

  // Pass 1: compute tile norms (all ranks need this for SparseShape)
  TA::Tensor<float> tile_norms(tiles_range, 0.0f);

  for (std::size_t i = 0; i < coo.values.size(); ++i) {
    std::vector<long> elem_idx(coo.rank);
    for (int d = 0; d < coo.rank; ++d)
      elem_idx[d] = static_cast<long>(coo.indices[i][d]);

    auto tidx = trange.element_to_tile(elem_idx);
    float val = static_cast<float>(coo.values[i]);
    tile_norms[tidx] += val * val;
  }
  tile_norms.inplace_unary([](float& x) { x = std::sqrt(x); });

  // Pass 2: create sparse shape and array (determines tile-to-rank mapping)
  TA::SparseShape<float> sp_shape(world, tile_norms, trange);
  TA::TSpArrayD array(world, trange, sp_shape);

  // Pass 3: re-scan entries, only collect entries for local non-zero tiles
  using ElemEntry = std::pair<std::vector<long>, double>;
  std::map<std::size_t, std::vector<ElemEntry>> tile_entries;

  for (std::size_t i = 0; i < coo.values.size(); ++i) {
    std::vector<long> elem_idx(coo.rank);
    for (int d = 0; d < coo.rank; ++d)
      elem_idx[d] = static_cast<long>(coo.indices[i][d]);

    auto tidx = trange.element_to_tile(elem_idx);
    auto ord = tiles_range.ordinal(tidx);

    if (array.is_zero(ord) || !array.is_local(ord)) continue;
    tile_entries[ord].emplace_back(std::move(elem_idx), coo.values[i]);
  }

  // Set each local tile
  for (auto& [ord, entries] : tile_entries) {
    auto tile_range = trange.make_tile_range(ord);
    TA::Tensor<double> tile(tile_range, 0.0);
    for (auto& [idx, val] : entries)
      tile[idx] = val;
    array.set(ord, std::move(tile));
  }

  // Set remaining local non-zero tiles (no COO entries) to zero
  for (auto it = array.begin(); it != array.end(); ++it) {
    auto ord = it.ordinal();
    if (!array.is_zero(ord) && array.is_local(ord) &&
        tile_entries.find(ord) == tile_entries.end()) {
      array.set(ord, TA::Tensor<double>(trange.make_tile_range(ord), 0.0));
    }
  }

  world.gop.fence();

  if (world.rank() == 0 && !label.empty()) {
    std::cerr << "  " << label << ": shape (";
    for (int d = 0; d < coo.rank; ++d) {
      if (d) std::cerr << ",";
      std::cerr << shape[d];
    }
    std::cerr << "), nnz=" << coo.values.size()
              << ", sparsity=" << (sp_shape.sparsity() * 100.0) << "%"
              << ", tiles=" << tiles_range.volume() << "\n";
  }

  return array;
}

#endif  // SPTC_TA_BUILDER_H
