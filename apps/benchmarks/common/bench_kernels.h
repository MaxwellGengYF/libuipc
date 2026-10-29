#pragma once
/********************************************************************
 * @file bench_kernels.h
 * @brief Kernel-level device surface for **Tier 1** (bench_plan.md §4.2, K1..K10).
 *
 * Tier 0 (`BenchDevice`) measures *submission*: one trivial kernel, many ways of
 * issuing it.  Tier 1 measures the kernels a frame actually spends its time in, so it
 * needs a different surface: a problem (matrix, vector, buffer), the kernel under test,
 * and the work/byte counts that turn a wall time into a per-unit and a GB/s number.
 *
 * Same contract as Tier 0 (§6.1): the cases are written **once** against this header and
 * compiled into one binary per backend; only the implementation TU differs
 * (`luisa/bench_kernels_luisa.cpp`, later `cuda/bench_kernels_cuda.cu`) -- so "same
 * scenario, two backends" is enforced by the linker, not by discipline.  The surface is
 * concrete (pimpl, no vtable): a measured window is a tight loop over `spmv()` /
 * `blas()` calls, and a virtual call there would land inside the measurement.
 *
 * The backend implementations wrap the *product* kernels, not a re-implementation:
 * K1 calls `luisa_device::linalg::LSpmv` (the linear solver's SpMV), K2 calls
 * `linalg::LLinearSystemContext` (dot / nrm2 / axpy), K0 is a plain fill/copy so that a
 * bandwidth roofline exists for the others to be compared against.
 *
 * Sizing note (§4.2): every case uses a fixed problem size and a fixed repetition count,
 * and reports work units + bytes so that a block-size or layout change is comparable
 * *per unit of work* rather than per call.
 *********************************************************************/
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include <common/bench_device.h>  // DeviceInfo (backend / device_name / runtime_version / extra)

namespace uipc::benchmark
{
/**
 * @brief Per-backend kernel bench: problems, the kernels under test, checksums, sync.
 *
 * Lifecycle: `init()` once per process, then any number of `make_*()` (outside the
 * measured window), then the measured window is a loop over `spmv()` / `blas()` /
 * `bandwidth()` followed by `synchronize()`.  `checksum*()` reads back a scalar *after*
 * the window (a full read-back inside the window would measure the copy, not the kernel).
 */
class BenchKernels
{
  public:
    BenchKernels();
    ~BenchKernels();
    BenchKernels(const BenchKernels&) = delete;
    BenchKernels& operator=(const BenchKernels&) = delete;

    // ---- lifecycle ---------------------------------------------------------
    /// Bring the backend module up. `program_path` is argv[0].
    void init(std::string_view program_path);
    void shutdown();
    [[nodiscard]] const DeviceInfo& info() const noexcept;

    /// The block size the kernels below are traced with (OP-C1's knob); recorded so a
    /// sweep's rows are self-describing.
    [[nodiscard]] std::uint32_t block_dim() const noexcept;
    /// Commit and wait for everything issued so far (the only sync in a measured case).
    void synchronize();

    // ---- K1: SpMV, upper-packed symmetric 3x3-block BCOO -------------------
    /// The linear solver's hot kernel.  `entries_per_row` is an average density of the
    /// upper triangle; the builder guarantees `triplets >= rows` (a diagonal entry).
    struct SpmvSetup
    {
        std::uint64_t rows = 0;
        std::uint64_t triplets = 0;  ///< upper-packed block entries
        std::uint64_t dofs = 0;      ///< rows * 3
        std::uint64_t nnz = 0;       ///< scalar non-zeros *after* the symmetric mirror
        /// Bytes touched by one `y = A x`: the packed blocks + x + y.
        std::uint64_t bytes_per_call = 0;
    };
    SpmvSetup make_spmv(std::uint64_t rows, double entries_per_row, std::uint32_t seed);
    [[nodiscard]] const SpmvSetup& spmv_setup() const noexcept;

    enum class SpmvVariant
    {
        Sym,        ///< `sym_spmv`: the plain symmetric expansion
        Rbk,        ///< `rbk_spmv`: row-blocked, general (reads the matrix as COO)
        RbkSym,     ///< `rbk_sym_spmv`: row-blocked symmetric
        RbkSymDot,  ///< `rbk_sym_spmv_dot`: symmetric + fused `x^T (A x)`
        Count
    };
    [[nodiscard]] static const char* name(SpmvVariant v) noexcept;
    /// One SpMV.  The caller decides how many times to call it.
    void spmv(SpmvVariant variant);
    /// Sum of the result vector's first elements (correctness gate, after the window).
    [[nodiscard]] double spmv_checksum() const;

    // ---- K2: dense BLAS / reduction ---------------------------------------
    void make_dense(std::uint64_t n, std::uint32_t seed);
    [[nodiscard]] std::uint64_t dense_n() const noexcept;
    [[nodiscard]] std::uint64_t dense_bytes_per_call(/*see BlasOp*/ int op) const noexcept;
    enum class BlasOp
    {
        Dot,   ///< x . y          2n
        Nrm2,  ///< ||x||          1n
        Axpy,  ///< y += a x       3n
        Fill,  ///< x = a          1n
        Count
    };
    [[nodiscard]] static const char* name(BlasOp op) noexcept;
    void blas(BlasOp op);
    [[nodiscard]] double blas_checksum() const;

    // ---- K0: pure bandwidth (the roofline the kernels above are compared to) ----
    void make_bandwidth(std::uint64_t floats, std::uint32_t seed);
    [[nodiscard]] std::uint64_t bandwidth_floats() const noexcept;
    enum class BwOp
    {
        Fill,  ///< b[i] = 1          write n
        Copy,  ///< b[i] = a[i]       read n + write n
        Count
    };
    [[nodiscard]] static const char* name(BwOp op) noexcept;
    [[nodiscard]] std::uint64_t bandwidth_bytes_per_call(BwOp op) const noexcept;
    void bandwidth(BwOp op);
    [[nodiscard]] double bandwidth_checksum() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace uipc::benchmark
