#!/usr/bin/env python3
"""T0.2 — muda/cuda_tool/Eigen/CUB/CUDA-intrinsic API inventory generator.

Scans the reference CUDA backend (src/backends/cuda) and emits a table of every
distinct device-side API symbol that the LuisaCompute backend must replace.

The repository's CUDA backend was migrated off the third-party `muda` library to
an in-tree `cuda_tool` layer (namespace uipc::backend::cuda_tool); this script
scans for both spellings so the inventory stays valid across that boundary.

Usage:
    python scripts/luisa_inventory.py            # write docs/luisa_api_inventory.md
    python scripts/luisa_inventory.py --stdout   # print to stdout
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUDA_DIR = os.path.join(REPO, "src", "backends", "cuda")
OUT = os.path.join(REPO, "docs", "luisa_api_inventory.md")

SRC_EXT = (".cu", ".cuh", ".cpp", ".h", ".inl", ".hpp")

# symbol pattern -> category
PATTERNS: list[tuple[str, str]] = [
    (r"cuda_tool::[A-Za-z_][A-Za-z0-9_:]*", "cuda_tool"),
    (r"muda::[A-Za-z_][A-Za-z0-9_:]*", "muda"),
    (r"cub::[A-Za-z_][A-Za-z0-9_]*", "cub"),
    (r"Eigen::[A-Za-z_][A-Za-z0-9_]*", "eigen"),
    (
        r"\b(__syncthreads|__syncwarp|__shfl[a-z_]*|__ballot_sync|__any_sync|__all_sync|"
        r"__activemask|__popc|__popcll|__clz|__ffs|__brev|__threadfence[a-z_]*|"
        r"__fmaf_rn|__fma_rn|__reduce_add_sync|__reduce_min_sync|__reduce_max_sync|"
        r"__ldg|__ldca|__ldcg|__ldcs|__stcs|__stcg|__float_as_int|__int_as_float|"
        r"__double_as_longlong|__longlong_as_double|__hiloint2double|__double2hiint|"
        r"__nanosleep|__trap|__prof_trigger)\b",
        "intrinsic",
    ),
    (
        r"\b(atomicAdd|atomicSub|atomicExch|atomicMin|atomicMax|atomicInc|atomicDec|"
        r"atomicCAS|atomicAnd|atomicOr|atomicXor)\b",
        "cuda_atomic",
    ),
    (r"\b(__global__|__device__|__shared__|__constant__)\b", "qualifier"),
]

# LC replacement per distinct symbol (exact or prefix based). '*' = prefix match.
MAP: dict[str, tuple[str, str]] = {
    # ---- cuda_tool containers / views -------------------------------------
    "cuda_tool::BufferView": ("lc::LBufferView<T> (device rw view)", "yes"),
    "cuda_tool::CBufferView": ("lc::LCBufferView<T> (device ro view)", "yes"),
    "cuda_tool::VarView": ("lc::LVarView<T>", "yes"),
    "cuda_tool::CVarView": ("lc::LCVarView<T>", "yes"),
    "cuda_tool::Dense": ("lc::LDense<T> (read-write scalar)", "yes"),
    "cuda_tool::CDense": ("lc::LCDense<T> (read-only scalar)", "yes"),
    "cuda_tool::Dense2D": ("lc::LDense2D<T>", "yes"),
    "cuda_tool::CDense2D": ("lc::LCDense2D<T>", "yes"),
    "cuda_tool::DeviceVar": ("lc::LDeviceVar<T> (1-elem LBuffer)", "yes"),
    "cuda_tool::DeviceVector": ("lc::LDeviceBuffer<T>", "yes"),
    "cuda_tool::DeviceBuffer": ("lc::LDeviceBuffer<T>", "yes"),
    "cuda_tool::DeviceBuffer2D": ("lc::LDeviceBuffer2D<T>", "yes"),
    "cuda_tool::DeviceDenseVector": ("lc::LDeviceDenseVector<T>", "yes"),
    "cuda_tool::BufferLaunch": ("lc::LBufferLaunch (fill/copy/transform)", "yes"),
    "cuda_tool::Buffer2DView": ("lc::LBuffer2DView<T>", "yes"),
    "cuda_tool::CBuffer2DView": ("lc::LCBuffer2DView<T>", "yes"),
    "cuda_tool::Extent2D": ("lc::LExtent2D", "yes"),
    "cuda_tool::ViewerBase": ("lc::LViewerBase", "yes"),
    "cuda_tool::CBCOOVectorView": ("lc::linalg::LCBCOOVectorView<T,N> (T3.1)", "yes"),
    "cuda_tool::DeviceDenseMatrix": ("lc::linalg::LDenseMatrix<T> (T3.1)", "yes"),
    "cuda_tool::GraphCapture": ("lc::RuntimeDependencyGraph (CommandList replay)", "yes"),
    "cuda_tool::GraphWhile": ("lc::RuntimeDependencyGraph while-region", "yes"),
    "cuda_tool::GraphCapture::Result::Ok": ("lc::RuntimeDependencyGraph result enum", "yes"),
    "cuda_tool::GraphWhile::Result::Ok": ("lc::RuntimeDependencyGraph result enum", "yes"),
    "cuda_tool::GraphWhile::runtime_supported": ("lc::RuntimeDependencyGraph::supported()", "yes"),
    "cuda_tool::LaunchBase": ("lc::kernel::LaunchBase", "yes"),
    "cuda_tool::Device": ("luisa::compute::Device (LC)", "no"),
    "cuda_tool::check_cuda_error": ("lc::check_device_error (luisa::compute throw)", "yes"),
    "cuda_tool::wait_stream": ("LStream::synchronize()", "no"),
    "cuda_tool::debug_trap": ("$assert(false)", "no"),
    "cuda_tool::details::eigen": ("lc::math (LVec/LMat, T2.x)", "yes"),
    "muda::BufferLaunch": ("lc::LBufferLaunch (fill/copy/transform)", "yes"),
    "muda::DeviceVar": ("lc::LDeviceVar<T> (1-elem LBuffer)", "yes"),
    # ---- launch helpers ----------------------------------------------------
    "cuda_tool::best_block_dim": ("lc::kernel::block_dim()", "yes"),
    "cuda_tool::best_grid_dim": ("lc::kernel::grid_dim(n)", "yes"),
    "cuda_tool::default_block_dim": ("lc::kernel::kDefaultBlockDim", "yes"),
    "cuda_tool::atomic_add": ("lc::atomic_add(Expr<T>&, v)", "yes"),
    "cuda_tool::atomic_exch": ("lc::atomic_exch(Expr<T>&, v)", "yes"),
    "cuda_tool::wait_device": ("lc::wait_device() -> stream.synchronize()", "yes"),
    "cuda_tool::Stream": ("lc::LStream (wraps CommandList+Stream)", "yes"),
    "cuda_tool::Stream::Default": ("lc::device::default_stream()", "yes"),
    "cuda_tool::LinearSystemContext": ("lc::LLinearSystemContext", "yes"),
    # ---- device algorithms (replacements for CUB) --------------------------
    "cuda_tool::DeviceReduce": ("lc::linalg::reduce (Phase T3.2)", "yes"),
    "cuda_tool::DeviceScan": ("lc::linalg::inclusive_scan (T3.2)", "yes"),
    "cuda_tool::DeviceSelect": ("lc::linalg::select (T3.2)", "yes"),
    "cuda_tool::DevicePartition": ("lc::linalg::partition (T3.2)", "yes"),
    "cuda_tool::DeviceRadixSort": ("lc::linalg::radix_sort_by_key (T3.2)", "yes"),
    "cuda_tool::DeviceMergeSort": ("lc::linalg::merge_sort (T3.2)", "yes"),
    "cuda_tool::DeviceRunLength": ("lc::linalg::run_length_encode (T3.2)", "yes"),
    "cuda_tool::DeviceRunLengthEncode": ("lc::linalg::run_length_encode (T3.2)", "yes"),
    "cuda_tool::DeviceSpmv": ("lc::linalg::spmv (T3.4)", "yes"),
    # ---- matrices / vectors ------------------------------------------------
    "cuda_tool::DeviceTripletMatrix": ("lc::linalg::LTripletMatrix<T,N> (T3.1)", "yes"),
    "cuda_tool::DeviceDoubletVector": ("lc::linalg::LDoubletVector<T,N> (T3.1)", "yes"),
    "cuda_tool::DeviceBCOOMatrix": ("lc::linalg::LBCOOMatrix<T,N> (T3.1)", "yes"),
    "cuda_tool::DeviceBCOOVector": ("lc::linalg::LBCOOVector<T,N> (T3.1)", "yes"),
    "cuda_tool::DeviceBSRMatrix": ("lc::linalg::LBSRMatrix<T,N> (T3.1)", "yes"),
    "cuda_tool::DeviceCOOMatrix": ("lc::linalg::LCOOMatrix<T,N> (T3.1)", "yes"),
    "cuda_tool::TripletMatrixView": ("lc::linalg::LTripletMatrixView<T,N>", "yes"),
    "cuda_tool::CTripletMatrixView": ("lc::linalg::LCTripletMatrixView<T,N>", "yes"),
    "cuda_tool::DoubletVectorView": ("lc::linalg::LDoubletVectorView<T,N>", "yes"),
    "cuda_tool::CDoubletVectorView": ("lc::linalg::LCDoubletVectorView<T,N>", "yes"),
    "cuda_tool::BCOOMatrixView": ("lc::linalg::LBCOOMatrixView<T,N>", "yes"),
    "cuda_tool::CBCOOMatrixView": ("lc::linalg::LCTripletMatrixView<T,N>", "yes"),
    "cuda_tool::DenseVectorView": ("lc::linalg::LDenseVectorView<T>", "yes"),
    "cuda_tool::CDenseVectorView": ("lc::linalg::LCDenseVectorView<T>", "yes"),
    "cuda_tool::TripletMatrixViewer": ("lc::linalg::LTripletMatrixView<T,N>", "yes"),
    "cuda_tool::DoubletVectorViewer": ("lc::linalg::LDoubletVectorView<T,N>", "yes"),
    "cuda_tool::DenseVectorViewer": ("lc::linalg::LDenseVectorView<T>", "yes"),
    # ---- debug / logging ---------------------------------------------------
    "cuda_tool::Debug": ("lc::debug (sync callbacks, device debug)", "yes"),
    "cuda_tool::Debug::debug_sync_all": ("lc::debug::sync_all() -> stream.synchronize()", "yes"),
    "cuda_tool::Debug::set_sync_callback": ("lc::debug::set_sync_callback()", "yes"),
    "cuda_tool::KernelCout": ("lc::KernelCout (device_log capture)", "yes"),
    "cuda_tool::KernelCout::viewer": ("lc::KernelCout::viewer()", "yes"),
    "cuda_tool::LoggerViewer": ("lc::LoggerViewer", "yes"),
    # ---- cuda_tool::eigen --------------------------------------------------
    "cuda_tool::eigen": ("lc::math (LVec/LMat, T2.x)", "yes"),
    "cuda_tool::eigen::inverse": ("lc::math::inverse (T2.2)", "yes"),
    "cuda_tool::eigen::evd": ("lc::math::evd (T2.2)", "yes"),
    "cuda_tool::eigen::svd": ("lc::math::svd (T2.2)", "yes"),
    "cuda_tool::eigen::pd": ("lc::math::polar_decomposition (T2.2)", "yes"),
    "cuda_tool::eigen::atomic_add": ("lc::math::atomic_add (T2.2)", "yes"),
    "cuda_tool::eigen::AnalyticalInverse": ("lc::math::analytical_inverse (T2.2)", "yes"),
    # ---- intrinsics --------------------------------------------------------
    "__syncthreads": ("$barrier / block sync in DSL", "no"),
    "__syncwarp": ("$barrier / warp sync in DSL", "no"),
    "__shfl_down_sync": ("warp shuffle in DSL (shfl_down)", "yes (builtins)"),
    "__shfl_up_sync": ("warp shuffle in DSL (shfl_up)", "yes (builtins)"),
    "__shfl_xor_sync": ("warp shuffle in DSL (shfl_xor)", "yes (builtins)"),
    "__shfl_sync": ("warp shuffle in DSL (shfl)", "yes (builtins)"),
    "__ballot_sync": ("warp_active_bit_mask", "yes (builtins)"),
    "__any_sync": ("warp_active_any", "yes (builtins)"),
    "__all_sync": ("warp_active_all", "yes (builtins)"),
    "__activemask": ("warp_active_bit_mask / lane count", "yes (builtins)"),
    "__popc": ("luisa::compute::popcount", "yes (builtins)"),
    "__clz": ("luisa::compute::clz / count_leading_zeros", "yes (builtins)"),
    "__ffs": ("luisa::compute::find_first_set", "yes (builtins)"),
    "__brev": ("luisa::compute::reverse_bits", "yes (builtins)"),
    "__threadfence": ("device-fence in DSL", "yes (builtins)"),
    "__float_as_int": ("as<int>(float) bit-cast", "yes (builtins)"),
    "__int_as_float": ("as<float>(int) bit-cast", "yes (builtins)"),
    "__double_as_longlong": ("as<ulong>(double) bit-cast", "yes (builtins)"),
    "__longlong_as_double": ("as<double>(ulong) bit-cast", "yes (builtins)"),
    "__double2hiint": ("as<uint>(double) high word bit-cast", "yes (builtins)"),
    "__hiloint2double": ("as<double> from hi/lo words", "yes (builtins)"),
    "cub::WARP_SYNC": ("warp_active_count_bits / $barrier", "yes (builtins)"),
    "__trap": ("$assert(false)", "no"),
    # ---- cuda atomics ------------------------------------------------------
    "atomicAdd": ("LBufferView::atomic(i).fetch_add(v) / atomic_fetch_add", "no"),
    "atomicSub": ("LBufferView::atomic(i).fetch_sub(v)", "no"),
    "atomicExch": ("LBufferView::atomic(i).exchange(v)", "no"),
    "atomicMin": ("LBufferView::atomic(i).fetch_min(v)", "no"),
    "atomicMax": ("LBufferView::atomic(i).fetch_max(v)", "no"),
    "atomicInc": ("CAS loop / atomic_inc shim", "yes"),
    "atomicDec": ("CAS loop / atomic_dec shim", "yes"),
    "atomicCAS": ("LBufferView::atomic(i).compare_exchange(...)", "no"),
    "atomicAnd": ("LBufferView::atomic(i).fetch_and(v)", "no"),
    "atomicOr": ("LBufferView::atomic(i).fetch_or(v)", "no"),
    "atomicXor": ("LBufferView::atomic(i).fetch_xor(v)", "no"),
    # ---- CUB ---------------------------------------------------------------
    "cub::WarpReduce": ("lc::linalg::warp_reduce (warp_active_sum)", "yes"),
    "cub::WarpScan": ("lc::linalg::warp_scan", "yes"),
    "cub::BlockReduce": ("lc::linalg::block_reduce (Shared<T>)", "yes"),
    "cub::DeviceReduce": ("lc::linalg::reduce (T3.2)", "yes"),
    "cub::DeviceScan": ("lc::linalg::inclusive_scan (T3.2)", "yes"),
    "cub::DeviceSelect": ("lc::linalg::select (T3.2)", "yes"),
    "cub::DevicePartition": ("lc::linalg::partition (T3.2)", "yes"),
    "cub::DeviceRadixSort": ("lc::linalg::radix_sort_by_key (T3.2)", "yes"),
    "cub::DeviceMergeSort": ("lc::linalg::merge_sort (T3.2)", "yes"),
    "cub::DeviceRunLengthEncode": ("lc::linalg::run_length_encode (T3.2)", "yes"),
    "cub::DeviceSpmv": ("lc::linalg::spmv (T3.4)", "yes"),
    "cub::ShuffleIndex": ("warp shfl via block index remap", "yes (builtins)"),
    "cub::KeyValuePair": ("lc::KeyValuePair<T> shim", "yes"),
    # ---- Eigen -------------------------------------------------------------
    "Eigen::Vector": ("lc::math::LVec<T,N> (T2.1)", "yes"),
    "Eigen::Matrix": ("lc::math::LMat<T,R,C> (T2.1)", "yes"),
    "Eigen::Vector3": ("lc::math::LVec<T,3>", "yes"),
    "Eigen::Vector3f": ("lc::math::LVec<float,3>", "yes"),
    "Eigen::Matrix3d": ("lc::math::LMat<double,3,3>", "yes"),
    "Eigen::Array": ("lc::math::LArray/LVec", "yes"),
    "Eigen::Array3": ("lc::math::LVec<T,3>", "yes"),
    "Eigen::Array4": ("lc::math::LVec<T,4>", "yes"),
    "Eigen::Map": ("lc::math::LMap / raw view", "yes"),
    "Eigen::MatrixBase": ("lc::math::LMatBase", "yes"),
    "Eigen::VectorBlock": ("lc::math::LBlock (block())", "yes"),
    "Eigen::Dynamic": ("compile-time size instead (static_assert)", "no"),
    "Eigen::ColMajor": ("row-major fixed storage convention", "yes"),
    "Eigen::RowMajor": ("row-major fixed storage convention", "yes"),
    "Eigen::Stride": ("explicit stride in LMap", "yes"),
    "Eigen::Unaligned": ("ignored (no vectorized alignment in DSL)", "no"),
    "Eigen::AlignedBox": ("lc::math::LAABB", "yes"),
    "Eigen::SparseMatrix": ("host-only; container is lc::linalg (T3.1)", "yes"),
    "Eigen::Triplet": ("lc::linalg::LTriplet", "yes"),
    "Eigen::VectorX": ("dynamic -> std::vector / LBuffer (host)", "yes"),
    "Eigen::SelfAdjointEigenSolver": ("lc::math::evd (T2.2)", "yes"),
    # ---- qualifiers --------------------------------------------------------
    "__global__": ("Kernel1D/Callable built with device.compile<1>", "no"),
    "__device__": ("DSL lambda body / inline callable", "no"),
    "__shared__": ("Shared<T> in DSL", "no"),
    "__constant__": ("Constant<T> / captured value in DSL", "no"),
}


def category_of(text: str) -> str | None:
    for pat, cat in PATTERNS:
        if re.search(pat, text):
            return cat
    return None


def scan() -> tuple[Counter, dict[str, list[str]], dict[str, str]]:
    counts: Counter = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    cats: dict[str, str] = {}
    pats = [(re.compile(p), c) for p, c in PATTERNS]
    for dirpath, _dirnames, filenames in os.walk(CUDA_DIR):
        for fn in filenames:
            if not fn.endswith(SRC_EXT):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, REPO).replace("\\", "/")
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        for pat, cat in pats:
                            for m in pat.finditer(line):
                                sym = m.group(0)
                                # normalise qualified muda/cuda_tool names
                                if cat in ("cuda_tool", "muda"):
                                    sym = re.sub(r"<.*", "", sym).rstrip(":")
                                counts[(cat, sym)] += 1
                                cats[sym] = cat
                                if len(samples[sym]) < 3:
                                    samples[sym].append(f"{rel}:{lineno}")
            except OSError:
                continue
    return counts, samples, cats


def lookup(sym: str) -> tuple[str, str] | None:
    if sym in MAP:
        return MAP[sym]
    # prefix fallback (e.g. cuda_tool::eigen::details::foo)
    for key in sorted(MAP, key=len, reverse=True):
        if sym.startswith(key):
            return MAP[key]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    counts, samples, cats = scan()
    total = sum(counts.values())
    per_cat: Counter = Counter()
    for (cat, _sym), n in counts.items():
        per_cat[cat] += n

    unmapped = []
    lines: list[str] = []
    w = lines.append
    w("# libuipc — muda / cuda_tool / Eigen / CUB / CUDA-intrinsic API inventory")
    w("")
    w("> **T0.2 artifact.** Auto-generated by `scripts/luisa_inventory.py`.")
    w("> Regenerate with `python scripts/luisa_inventory.py`.")
    w("")
    w("## 0. Method")
    w("")
    w("Scans every `.cu/.cuh/.cpp/.h/.inl/.hpp` under `src/backends/cuda` and")
    w("extracts every distinct device-side API symbol that has no meaning inside a")
    w("LuisaCompute AST kernel. The reference backend was migrated off the external")
    w("`muda` library onto the in-tree `cuda_tool` layer (`uipc::backend::cuda_tool`),")
    w("so both spellings are scanned. Symbols are grouped by category and each is")
    w("mapped to its LuisaCompute replacement.")
    w("")
    w("## 1. Summary")
    w("")
    w(f"- Total device-API symbol occurrences: **{total}**")
    w(f"- Distinct symbols: **{len(counts)}**")
    w(f"- Unmapped symbols: **{len(unmapped)}**")
    w("")
    w("| Category | Occurrences |")
    w("|---|---:|")
    for cat in sorted(per_cat, key=lambda c: -per_cat[c]):
        w(f"| {cat} | {per_cat[cat]} |")
    w("")

    for cat in sorted(per_cat, key=lambda c: -per_cat[c]):
        w(f"## Category: `{cat}`")
        w("")
        w("| Symbol | Occurrences | Example site | LC replacement | Shim? |")
        w("|---|---:|---|---|---|")
        rows = [(sym, n) for (c, sym), n in counts.items() if c == cat]
        rows.sort(key=lambda r: (-r[1], r[0]))
        for sym, n in rows:
            mapping = lookup(sym)
            if mapping is None:
                unmapped.append(sym)
                repl, shim = "**UNMAPPED**", "?"
            else:
                repl, shim = mapping
            site = samples[sym][0] if samples.get(sym) else ""
            w(f"| `{sym}` | {n} | `{site}` | {repl} | {shim} |")
        w("")

    w("## 2. Unmapped symbols")
    w("")
    if unmapped:
        for sym in sorted(set(unmapped)):
            w(f"- `{sym}`")
    else:
        w("None — every scanned symbol has an LC replacement. ✅")
    w("")

    text = "\n".join(lines) + "\n"
    if args.stdout:
        sys.stdout.write(text)
    else:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {OUT}: {total} occurrences, {len(counts)} distinct, "
              f"{len(set(unmapped))} unmapped")
    return 0 if not unmapped else 2


if __name__ == "__main__":
    raise SystemExit(main())
