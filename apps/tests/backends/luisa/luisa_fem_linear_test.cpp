/********************************************************************
 * @file luisa_fem_linear_test.cpp
 * @brief T5.4a — the FEM linear subsystem, the global linear system's
 * assembly/layout half and their fan-outs (LuisaCompute backend).
 *
 * Every oracle here is derived independently *in this file*: the reference
 * backend is CUDA-only and cannot be host-included, so nothing is copied from it.
 * The oracles are
 *  (a) the *ported* T5.2 constitution routines (`arap_3d_detail::tet_{gradient,
 *      hessian}`) evaluated on the host, used to build an independent dense
 *      Eigen model of the same element blocks in a *different* order and compared
 *      against the assembled device matrix through `MatrixConverter` + `Spmv`
 *      (the brief's headline validation),
 *  (b) hand-derived tuples for the dof layout and the extent budget (what a
 *      mixture of constitutions/bodies produces),
 *  (c) the frontend's own laws for the assembled *gradient* of a real scene
 *      (the kinetic `m (x - x~)`, T5.3 rule 1, plus the ARAP element gradients
 *      scattered at the element's global vertex indices, plus the test-local
 *      reporter's known contributions), and
 *  (d) the device-vs-host equivalence of the ported assembly routines
 *      (`double` on `storage::memory` vs `lcrt::Var<double>` inside a kernel),
 *      asserted *bitwise* because the arithmetic is identical (T2.1/T5.1/T5.3).
 *
 * ---------------------------------- T5.4a ----------------------------------
 * Coverage map (case -> oracle):
 *  1) the 5-tet mesh -> triplets -> MatrixConverter -> Spmv vs a dense
 *     Eigen A*x, plus the COO content (sorted, unique, exact block values)
 *     and the host-instantiation equality ............. fem_linear_assembly_five_tet
 *  2) the dof layout (uid order, offsets/counts) and the extent budget
 *     (kinetic | reporters | dytopo; gradient_only both ways), host-factored
 *     and through the real chain ...................... fem_linear_layout
 *  3) a scene through the public API: the new systems build and stay inert;
 *     the gradient-only assembly (real reporters + the mock reporter's
 *     slices) matches the frontend's laws ............ fem_linear_reporters
 *  4) every factored assembly routine: host `double` vs device DSL, bitwise
 *     .................................................. fem_linear_device_vs_host
 *
 * Shared-binary rules (cookbook §7.14 rule 6, §7.18 rule 6): every system
 * registered here is built for every scene of the whole test binary, so the
 * test-local reporter gates itself on a marker vertex attribute of *its own*
 * geometry and reports zero extents elsewhere; `find<T>()` returns a registered
 * but *invalidated* system, so "is it in play?" is asserted with `is_valid()`.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS FILE CANNOT REACH (the brief's §2; stated so nothing is implied)
 * ---------------------------------------------------------------------------
 * `GlobalLinearSystem::solve()` is T8.3 and `SimEngine` is not wired, so the
 * assemble -> solve path is not exercised here.  Exercised:
 * `GlobalLinearSystem::init()` (public in the port for exactly this reason — see
 * the header's deviation note) + `dof_count()`; `compute_gradient()` (the
 * *gradient-only* assembly: `FEMLinearSubsystem::Impl::assemble` ->
 * `_assemble_kinetic`/`_assemble_reporters` -> the reporter/constitution
 * fan-out); a *test-local* `GlobalLinearSystem::Impl` (`Impl::init` +
 * `_update_subsystem_extent`, i.e. `report_extent` with `gradient_only == false`);
 * and every shipped kernel body through its factored routine (host + device).
 * NOT reached: `solve()`/`spmv`/`apply_preconditioner`/`build_linear_system`'s
 * converter path/`distribute_solution` end to end (they need a solver),
 * `_assemble_linear_system` (same), the dytopo-effect branch (contact is
 * disabled in this backend's scenes) and `_assemble_preconditioner`'s global
 * branch (T8.3's type).
 *********************************************************************/
#include "luisa_test_shim.h"
#include "luisa_test_runtime.h"
#include <uipc/uipc.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/geometry/utils/factory.h>
#include <uipc/geometry/utils/label_surface.h>
#include <uipc/constitution/arap.h>
#include <uipc/constitution/stable_neo_hookean.h>
#include <uipc/constitution/elastic_moduli.h>
#include <type_define.h>
#include <frame_stream.h>
#include <linear_system/global_linear_system.h>
#include <linear_system/global_linear_system_function.h>
#include <linear_system/diag_linear_subsystem.h>
#include <finite_element/fem_linear_subsystem.h>
#include <finite_element/fem_linear_subsystem_reporter.h>
#include <finite_element/fem_linear_subsystem_function.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_elastics.h>
#include <finite_element/finite_element_animator.h>
#include <finite_element/constitutions/arap_function.h>
#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>
#include <finite_element/fem_utils.h>
#include <utils/offset_count_collection.h>
// The test target's include path puts src/backends/none first: the shell header
// must be qualified (cookbook §7.14 rule 7).
#include <backends/luisa/sim_engine.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using namespace uipc;
using namespace uipc::core;
using namespace uipc::geometry;
using namespace uipc::constitution;
namespace gg   = uipc::backend::luisa;
namespace lud  = uipc::backend::luisa::luisa_device;
namespace lps  = uipc::backend::luisa::luisa_device::math;
namespace lla  = uipc::backend::luisa::luisa_device::linalg;
namespace lcrt = ::luisa::compute;
namespace linf = uipc::backend::luisa::fem_linear_subsystem_detail;
namespace lgls = uipc::backend::luisa::global_linear_system_detail;

// The ported surface lives in `uipc::backend::luisa` (like the reference's does in
// `uipc::backend::cuda`), so a test body sees the same spellings a ported system
// does: `write_element_hessian`, `scatter_gradient_row`, `plan_extent`, ...
using namespace uipc::backend::luisa;

namespace t54
{
// ---------------------------------------------------------------------------
//  small helpers (the T5.1/T5.3 test files' idioms)
// ---------------------------------------------------------------------------
[[nodiscard]] inline double rel_err(double a, double b)
{
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a - b) / scale;
}

[[nodiscard]] inline std::string workspace_path(std::string_view name)
{
    auto p = std::filesystem::temp_directory_path() / name;
    std::filesystem::create_directories(p);
    return p.string() + "/";
}

/// Read an LC buffer back to the host (frame stream first: a command may still be
/// recorded in an open frame).
template <typename T>
std::vector<T> read_device(const lud::LCBufferView<T>& view)
{
    std::vector<T> host(view.size());
    gg::frame_stream().synchronize();
    if(!host.empty())
    {
        view.copy_to(span<T>{host});
        lud::main_stream().synchronize();
    }
    return host;
}

[[nodiscard]] std::vector<Vector3> read_vec3(const lud::LCBufferView<gg::LVector3>& view)
{
    auto                 raw = read_device(view);
    std::vector<Vector3> host;
    host.reserve(raw.size());
    for(auto& v : raw)
        host.push_back(gg::to_eigen(v));
    return host;
}

/// Host-view aliases (the T5.2 assembler case's idiom).
using HostDoublet  = lla::LDoubletVectorHostMemoryView<double, 3>;
using HostTriplet  = lla::LTripletMatrixHostMemoryView<double, 3, 3>;
using HostDense    = lla::LDenseVectorHostMemoryView<double>;

// device-view aliases for the test-local kernels
using DoubletDeviceView  = lla::LDoubletVectorDeviceView<lcrt::Var<double>, 3>;
using TripletDeviceView  = lla::LTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using CTripletDeviceView = lla::LCTripletMatrixDeviceView<lcrt::Var<double>, 3, 3>;
using DenseDeviceView    = lla::LDenseVectorDeviceView<lcrt::Var<double>>;
using CDenseDeviceView   = lla::LCDenseVectorDeviceView<lcrt::Var<double>>;
using CDoubletDeviceView = lla::LCDoubletVectorDeviceView<lcrt::Var<double>, 3>;

// ===========================================================================
// 1) the five-tet mesh (written down here: connectivity + rest positions)
// ===========================================================================
[[nodiscard]] inline lps::LVec<double, 3> vec3_at(const std::vector<Vector3>& v, int i)
{
    return lps::LVec<double, 3>{v[i](0), v[i](1), v[i](2)};
}

struct FiveTet
{
    std::vector<Vector3>  X;  // rest positions
    std::vector<Vector3>  x;  // deformed positions (the state the Hessians see)
    std::vector<Vector4i> tets;
    double                kappa = 1.0e6;
    double                dt    = 1.0e-2;
};

/// 8 vertices, 5 tets.  `tets[3]` and `tets[4]` are deliberately *descending* in
/// their global indices, which is the case the reference's `upper_LR` swap exists
/// for (T5.2's assembler case).
[[nodiscard]] inline FiveTet five_tet()
{
    FiveTet m;
    m.X = {
        {0.0, 0.0, 0.0},    // 0
        {1.0, 0.0, 0.0},    // 1
        {0.0, 1.0, 0.0},    // 2
        {0.0, 0.0, 1.0},    // 3
        {1.0, 1.0, 1.0},    // 4
        {2.0, 0.5, 0.25},   // 5
        {-0.25, 0.5, 1.5},  // 6
        {0.5, -0.25, -0.5}  // 7
    };
    m.tets = {{0, 1, 2, 3}, {0, 1, 2, 4}, {0, 2, 3, 6}, {3, 1, 2, 0}, {4, 3, 1, 7}};
    // a deformation that turns every Hessian block on (a small shear + stretch)
    m.x = {
        {0.05, -0.02, 0.01},
        {1.10, 0.05, -0.03},
        {-0.04, 1.05, 0.02},
        {0.03, 0.01, 0.95},
        {1.05, 1.10, 0.98},
        {2.05, 0.48, 0.29},
        {-0.28, 0.55, 1.45},
        {0.52, -0.20, -0.55},
    };
    return m;
}

[[nodiscard]] inline lps::LMat<double, 3, 3> element_Dm_inv(const FiveTet& m, SizeT e)
{
    const auto& t = m.tets[e];
    return fem::Dm_inv<double>(vec3_at(m.X, t(0)),
                               vec3_at(m.X, t(1)),
                               vec3_at(m.X, t(2)),
                               vec3_at(m.X, t(3)));
}

[[nodiscard]] inline double element_volume(const FiveTet& m, SizeT e)
{
    const auto& t  = m.tets[e];
    auto        Ds = fem::Ds<double>(vec3_at(m.X, t(0)),
                              vec3_at(m.X, t(1)),
                              vec3_at(m.X, t(2)),
                              vec3_at(m.X, t(3)));
    return lps::determinant(Ds) / 6.0;
}

/// The element's local 12x12 Hessian from the *ported* T5.2 ARAP routine (host
/// instantiation), with the rest metric derived here from the rest positions.
[[nodiscard]] inline lps::LMat<double, 12, 12> element_hessian(const FiveTet& m, SizeT e)
{
    return arap_3d_detail::tet_hessian<double>(m.kappa,
                                               m.dt,
                                               element_volume(m, e),
                                               vec3_at(m.x, m.tets[e](0)),
                                               vec3_at(m.x, m.tets[e](1)),
                                               vec3_at(m.x, m.tets[e](2)),
                                               vec3_at(m.x, m.tets[e](3)),
                                               element_Dm_inv(m, e));
}

[[nodiscard]] inline lps::LVec<double, 12> element_gradient(const FiveTet& m, SizeT e)
{
    return arap_3d_detail::tet_gradient<double>(m.kappa,
                                                m.dt,
                                                element_volume(m, e),
                                                vec3_at(m.x, m.tets[e](0)),
                                                vec3_at(m.x, m.tets[e](1)),
                                                vec3_at(m.x, m.tets[e](2)),
                                                vec3_at(m.x, m.tets[e](3)),
                                                element_Dm_inv(m, e));
}

// ===========================================================================
// 2) the five-tet assembly kernel (the ported fan-out write, on the device)
// ===========================================================================
auto& five_tet_assembly_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<LVector4i>,
                                   lcrt::Buffer<double>,  // element Hessians (flat 144)
                                   lcrt::Buffer<double>,  // element gradients (flat 12)
                                   lcrt::Buffer<std::int32_t>,  // triplet offsets
                                   lcrt::Buffer<std::int32_t>,  // doublet offsets
                                   lcrt::Buffer<std::int32_t>,  // rows
                                   lcrt::Buffer<std::int32_t>,  // cols
                                   lcrt::Buffer<double>,        // values
                                   lcrt::Buffer<std::int32_t>,  // doublet indices
                                   lcrt::Buffer<double>,        // doublet values
                                   uint32_t,                    // vertex count
                                   uint32_t,                    // total triplets
                                   uint32_t,                    // total doublets
                                   uint32_t>(                   // element count
            [](lcrt::BufferVar<LVector4i>    tets,
               lcrt::BufferVar<double>       hessians,
               lcrt::BufferVar<double>       gradients,
               lcrt::BufferVar<std::int32_t> triplet_offsets,
               lcrt::BufferVar<std::int32_t> doublet_offsets,
               lcrt::BufferVar<std::int32_t> rows,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       values,
               lcrt::BufferVar<std::int32_t> doublet_indices,
               lcrt::BufferVar<double>       doublet_values,
               lcrt::Var<uint32_t>           vertex_count,
               lcrt::Var<uint32_t>           total_triplets,
               lcrt::Var<uint32_t>           total_doublets,
               lcrt::Var<uint32_t>           n) noexcept
            {
                auto e = lcrt::dispatch_x();
                $if(e < n)
                {
                    const auto ee = lcrt::cast<std::int32_t>(e);
                    auto       A  = TripletDeviceView::from_handles_full(rows,
                                                                   cols,
                                                                   values,
                                                                   vertex_count,
                                                                   vertex_count,
                                                                   std::int32_t{0},
                                                                   total_triplets,
                                                                   total_triplets,
                                                                   std::int32_t{0},
                                                                   std::int32_t{0},
                                                                   vertex_count,
                                                                   vertex_count);
                    auto       G  = DoubletDeviceView::from_handles(doublet_indices,
                                                              doublet_values,
                                                              vertex_count,
                                                              total_doublets);
                    auto       tet = load_vec4i(tets->read(e));
                    auto       H   = lps::make_mat<lcrt::Var<double>, 12, 12>(
                        [&](int r, int c) noexcept
                        { return hessians->read(ee * 144 + c * 12 + r); });
                    auto g = lps::make_vec<lcrt::Var<double>, 12>(
                        [&](int i) noexcept { return gradients->read(ee * 12 + i); });

                    fem_linear_subsystem_detail::write_element_gradient(
                        G, doublet_offsets->read(e), tet, g);
                    fem_linear_subsystem_detail::write_element_hessian(
                        A, triplet_offsets->read(e), tet, H);
                };
            });
    return k;
}

// ===========================================================================
// 3) the test-local reporter: its own known extent + a known write into its
//    slices of the shared buffers (the T5.3 mock pattern)
// ===========================================================================
class T54LinearReporter final : public gg::FEMLinearSubsystemReporter
{
  public:
    using FEMLinearSubsystemReporter::FEMLinearSubsystemReporter;

    // Reserved uid 0x5435344C494E53 == "T54LINS" (7 bytes, > 2^32).  The uids in
    // use elsewhere are 671 (FEMExternalForceManager) and the test mocks
    // 0x5445535456455254, 0x543432414E494D, 0x543432464F5243, 0x54343344594E54
    // (T4.1-T4.3) and 0x54353346454D58 (T5.3) -- none collides.
    static constexpr U64   kUid              = 0x5435344C494E53ull;
    static constexpr SizeT kGradientCount    = 2;  // 2 doublets (vertices 0 and 1)
    static constexpr SizeT kHessianCount     = 3;  // a 2-stencil's upper triangle
    static constexpr const char* kMarker     = "t54a_linear_mock";

    static inline bool  enabled            = false;
    static inline int   init_count         = 0;
    static inline int   report_extent_count = 0;
    static inline int   assemble_count     = 0;
    static inline bool  last_gradient_only = false;
    static inline SizeT last_gradient_size = 0;
    static inline SizeT last_hessian_size  = 0;
    static inline SizeT last_reported_gradient = 0;
    static inline SizeT last_reported_hessian  = 0;
    static inline SizeT last_gradient_offset   = 0;
    static inline SizeT last_hessian_offset    = 0;

    static void reset()
    {
        enabled            = false;
        init_count         = 0;
        report_extent_count = 0;
        assemble_count     = 0;
        last_gradient_only = false;
        last_gradient_size = 0;
        last_hessian_size  = 0;
        last_reported_gradient = 0;
        last_reported_hessian  = 0;
        last_gradient_offset   = 0;
        last_hessian_offset    = 0;
    }

    /// The values the mock writes (they are part of the expected gradient of the
    /// scene test: 1.5 at vertex 0, 2.5 at vertex 1, and a known 6x6 block).
    static inline const std::vector<double> kGradientValues = {1.5, 1.5, 1.5, 2.5, 2.5, 2.5};
    [[nodiscard]] static lps::LMat<double, 6, 6> hessian_values()
    {
        lps::LMat<double, 6, 6> H;
        for(int c = 0; c < 6; ++c)
            for(int r = 0; r < 6; ++r)
                H(r, c) = 0.5 + 10.0 * r + c;
        return H;
    }

  protected:
    // The reporter base has no `get_uid` hook (its `m_index` is assigned by
    // FEMLinearSubsystem::Impl::init), so this is documentation, not an override:
    // the uid is reserved so no real system can collide with the mock.
    [[nodiscard]] U64 t54_reserved_uid() const noexcept { return kUid; }

    void do_build(BuildInfo& info) override
    {
        (void)info;
        m_method = &require<gg::FiniteElementMethod>();
    }

    void do_init(InitInfo& info) override
    {
        (void)info;
        ++init_count;
        // gate on this test's own marker attribute (cookbook §7.14 rule 6)
        enabled = false;
        for(auto& slot : world().scene().geometries())
        {
            if(auto* sc = slot->geometry().as<geometry::SimplicialComplex>())
            {
                if(sc->vertices().find<IndexT>(kMarker))
                {
                    enabled = true;
                    break;
                }
            }
        }
    }

    void do_report_extent(ReportExtentInfo& info) override
    {
        ++report_extent_count;
        last_gradient_only = info.gradient_only();
        if(!enabled)
        {
            info.gradient_count(0);
            info.hessian_count(0);
            return;
        }
        info.gradient_count(kGradientCount);
        // a gradient-only pass must report no Hessian blocks (the subsystem and
        // this reporter both assert it) -- exactly like the real reporters
        info.hessian_count(info.gradient_only() ? 0 : kHessianCount);
        last_reported_gradient = kGradientCount;
        last_reported_hessian  = info.gradient_only() ? 0 : kHessianCount;
    }

    void do_assemble(AssembleInfo& info) override
    {
        ++assemble_count;
        auto gradients = info.gradients();
        auto hessians  = info.hessians();
        last_gradient_size = static_cast<SizeT>(gradients.doublet_count());
        last_hessian_size  = static_cast<SizeT>(hessians.triplet_count());
        if(!enabled)
            return;

        last_gradient_offset = static_cast<SizeT>(gradients.doublet_index_offset());
        last_hessian_offset  = static_cast<SizeT>(hessians.triplet_index_offset());
        write_mock_gradient(gradients);
        if(!info.gradient_only())
            write_mock_hessian(hessians);
    }

  private:
    FiniteElementMethod* m_method = nullptr;
    void write_mock_gradient(lla::LDoubletVectorView<Float, 3> gradients);
    void write_mock_hessian(lla::LTripletMatrixView<Float, 3, 3> hessians);
};
}  // namespace t54

REGISTER_SIM_SYSTEM(t54::T54LinearReporter);

namespace t54
{
/// The mock reporter's write, through the *ported* fan-out routine (the same code
/// path a constitution's `do_compute_gradient_hessian` runs).
auto& mock_gradient_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // slice indices
                                   lcrt::Buffer<double>,        // slice values
                                   uint32_t>(                  // total segments
            [](lcrt::BufferVar<std::int32_t> indices,
               lcrt::BufferVar<double>       values,
               lcrt::Var<uint32_t>           total_segments) noexcept
            {
                auto D = DoubletDeviceView::from_handles(indices,
                                                         values,
                                                         total_segments,
                                                         2u);  // 2 doublets
                lps::LVec<lcrt::Var<std::int32_t>, 2> verts;
                verts(0) = std::int32_t{1};
                verts(1) = std::int32_t{2};
                lps::LVec<lcrt::Var<double>, 6> g;
                for(int i = 0; i < 6; ++i)
                    g(i) = T54LinearReporter::kGradientValues[static_cast<std::size_t>(i)];
                fem_linear_subsystem_detail::write_element_gradient(
                    D, std::int32_t{0}, verts, g);
            });
    return k;
}

auto& mock_hessian_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // slice rows
                                   lcrt::Buffer<std::int32_t>,  // slice cols
                                   lcrt::Buffer<double>,        // slice values
                                   int32_t,  // submatrix offset x
                                   int32_t,  // submatrix offset y
                                   int32_t,  // submatrix extent x
                                   int32_t,  // submatrix extent y
                                   uint32_t,  // total block rows
                                   uint32_t>(  // total block cols
            [](lcrt::BufferVar<std::int32_t> rows,
               lcrt::BufferVar<std::int32_t> cols,
               lcrt::BufferVar<double>       values,
               lcrt::Var<std::int32_t>       sm_offset_x,
               lcrt::Var<std::int32_t>       sm_offset_y,
               lcrt::Var<std::int32_t>       sm_extent_x,
               lcrt::Var<std::int32_t>       sm_extent_y,
               lcrt::Var<uint32_t>           total_block_rows,
               lcrt::Var<uint32_t>           total_block_cols) noexcept
            {
                auto H = TripletDeviceView::from_handles_full(rows,
                                                              cols,
                                                              values,
                                                              total_block_rows,
                                                              total_block_cols,
                                                              std::int32_t{0},
                                                              3u,  // triplet count
                                                              3u,
                                                              sm_offset_x,
                                                              sm_offset_y,
                                                              sm_extent_x,
                                                              sm_extent_y);
                lps::LVec<lcrt::Var<std::int32_t>, 2> verts;
                verts(0) = std::int32_t{1};
                verts(1) = std::int32_t{2};
                auto V = T54LinearReporter::hessian_values();
                auto H66 = lps::make_mat<lcrt::Var<double>, 6, 6>(
                    [&](int r, int c) noexcept
                    { return lcrt::Var<double>{V(r, c)}; });
                fem_linear_subsystem_detail::write_element_hessian(
                    H, std::int32_t{0}, verts, H66);
            });
    return k;
}

void T54LinearReporter::write_mock_gradient(lla::LDoubletVectorView<Float, 3> gradients)
{
    const auto offset = static_cast<std::size_t>(gradients.doublet_index_offset());
    const auto count  = static_cast<std::size_t>(gradients.doublet_count());
    if(count < kGradientCount || m_method == nullptr)
        return;
    auto indices = gradients.indices().subview(offset, count);
    auto values  = gradients.values().subview(offset * 3u, count * 3u);
    auto n       = static_cast<uint32_t>(m_method->xs().size());
    auto& k      = mock_gradient_kernel();
    gg::frame_stream().record(k(indices, values, n).dispatch(1));
}

void T54LinearReporter::write_mock_hessian(lla::LTripletMatrixView<Float, 3, 3> hessians)
{
    const auto offset = static_cast<std::size_t>(hessians.triplet_index_offset());
    const auto count  = static_cast<std::size_t>(hessians.triplet_count());
    if(count < kHessianCount)
        return;
    auto rows = hessians.row_indices().subview(offset, count);
    auto cols = hessians.col_indices().subview(offset, count);
    auto vals = hessians.values().subview(offset * 9u, count * 9u);
    auto& k   = mock_hessian_kernel();
    gg::frame_stream().record(
        k(rows,
          cols,
          vals,
          static_cast<std::int32_t>(hessians.submatrix_offset().x),
          static_cast<std::int32_t>(hessians.submatrix_offset().y),
          static_cast<std::int32_t>(hessians.extent().x),
          static_cast<std::int32_t>(hessians.extent().y),
          static_cast<uint32_t>(hessians.total_rows()),
          static_cast<uint32_t>(hessians.total_cols()))
            .dispatch(1));
}
}  // namespace t54

namespace t54
{
// ===========================================================================
// 4) the probe: grabs the new systems' public interfaces once they are built
// ===========================================================================
class T54Probe final : public gg::SimSystem
{
  public:
    using SimSystem::SimSystem;

    inline static gg::FiniteElementMethod*   fem      = nullptr;
    inline static gg::GlobalLinearSystem*    gls      = nullptr;
    inline static gg::FEMLinearSubsystem*    femsub   = nullptr;
    inline static gg::FiniteElementElastics* elastics = nullptr;
    inline static gg::FiniteElementAnimator* animator = nullptr;
    /// the scene's geometry slots, captured in `do_build` (the public `World` API
    /// does not expose the scene, and the dof attributes are written later)
    inline static std::vector<S<geometry::GeometrySlot>> geo_slots;

    static void reset()
    {
        geo_slots.clear();
        fem      = nullptr;
        gls      = nullptr;
        femsub   = nullptr;
        elastics = nullptr;
        animator = nullptr;
    }

  private:
    void do_build() override
    {
        fem      = &require<gg::FiniteElementMethod>();
        gls      = &require<gg::GlobalLinearSystem>();
        femsub   = &require<gg::FEMLinearSubsystem>();
        elastics = &require<gg::FiniteElementElastics>();
        animator = find<gg::FiniteElementAnimator>();
        for(auto& slot : world().scene().geometries())
            geo_slots.push_back(slot);
    }
};
}  // namespace t54

REGISTER_SIM_SYSTEM(t54::T54Probe);

namespace t54
{
// ===========================================================================
// 5) the scene: a tet mesh + ARAP, with the mock reporter's marker attribute
// ===========================================================================
struct SceneDesc
{
    std::vector<Vector3>  positions;
    std::vector<Vector4i> tets;
    std::vector<SizeT>    body_of_vertex;  // which geometry each vertex belongs to
    std::vector<IndexT>   is_fixed;
    std::string           integrator_type = "bdf1";
    Float                 dt              = 0.01;
    Float                 density         = 1000.0;
    bool                  marker          = true;  // the mock reporter's marker
    // NOTE: StableNeoHookean (uid 10), *not* ARAP (uid 9): the shared test binary
    // registers T5.1's mock dim-3 constitution with ARAP's uid, and the method
    // hands a (dim, uid) geometry group to the *first* system registered with that
    // uid -- measured: in a full-suite run ARAP owned no geometry and the elastics
    // reporter reported 1 Hessian block instead of 20 (T5.1 rule 12 / T5.2 rule 12:
    // SNH's uid is unique and carries the strong end-to-end checks).
    ElasticModuli moduli = ElasticModuli::youngs_poisson(120.0e3, 0.49);
};

/// Two unit tets as *two separate geometries* (the "several bodies" case); each
/// tet has volume 1/6, so the vertex count is 8 and the layout is unambiguous.
[[nodiscard]] SceneDesc two_body_desc()
{
    SceneDesc d;
    d.positions = {{0.0, 0.0, 0.0},
                   {1.0, 0.0, 0.0},
                   {0.0, 1.0, 0.0},
                   {0.0, 0.0, 1.0},  // body 0
                   {3.0, 0.0, 0.0},
                   {4.0, 0.0, 0.0},
                   {3.0, 1.0, 0.0},
                   {3.0, 0.0, 1.0}};  // body 1
    d.body_of_vertex = {0, 0, 0, 0, 1, 1, 1, 1};
    d.is_fixed.assign(d.positions.size(), 0);
    d.is_fixed[0] = 1;  // one fixed vertex per body in the layout test
    d.is_fixed[4] = 1;
    return d;
}

/// Two unit tets in *one* geometry (the reporters case).
[[nodiscard]] SceneDesc one_body_desc()
{
    SceneDesc d;
    d.positions = {{0.0, 0.0, 0.0},
                   {1.0, 0.0, 0.0},
                   {0.0, 1.0, 0.0},
                   {0.0, 0.0, 1.0},
                   {3.0, 0.0, 0.0},
                   {4.0, 0.0, 0.0},
                   {3.0, 1.0, 0.0},
                   {3.0, 0.0, 1.0}};
    d.tets          = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    d.body_of_vertex.assign(d.positions.size(), 0);
    d.is_fixed.assign(d.positions.size(), 0);
    d.is_fixed[0] = 1;
    return d;
}

Scene build_scene(const SceneDesc& desc)
{
    auto config = Scene::default_config();
    // the dytopo effect manager is only created when contact is enabled, and
    // contact is not ported; the FEM method does not need it
    config["contact"]["enable"]      = false;
    config["sanity_check"]["enable"] = false;
    config["integrator"]["type"]     = desc.integrator_type;
    config["dt"]                     = desc.dt;
    config["gravity"]                = Vector3{0.0, -9.8, 0.0};

    Scene scene{config};
    auto  object = scene.objects().create("t54");

    StableNeoHookean snh;

    if(desc.body_of_vertex.empty())
    {
        // one geometry holding every tet
        auto mesh = tetmesh(span<const Vector3>{desc.positions},
                            span<const Vector4i>{desc.tets});
        label_surface(mesh);
        snh.apply_to(mesh, desc.moduli, desc.density);
        auto is_fixed = mesh.vertices().find<IndexT>(builtin::is_fixed);
        if(!is_fixed)
            is_fixed = mesh.vertices().create<IndexT>(builtin::is_fixed, 0);
        std::ranges::copy(desc.is_fixed, view(*is_fixed).begin());
        if(desc.marker)
            mesh.vertices().create<IndexT>(T54LinearReporter::kMarker, 1);
        object->geometries().create(mesh);
    }
    else
    {
        // one geometry per body: each body gets *its own* vertices (the global
        // vertex ids of its tets, remapped to local ones), which is what makes the
        // two bodies two `geo_infos` with 4 vertices each
        std::map<SizeT, std::vector<SizeT>> vertices_of_body;
        std::map<SizeT, std::vector<Vector4i>> tets_of_body;
        for(auto& t : desc.tets)
        {
            const auto body = static_cast<SizeT>(desc.body_of_vertex[static_cast<std::size_t>(t(0))]);
            auto&      verts = vertices_of_body[body];
            for(int i = 0; i < 4; ++i)
            {
                const auto g = static_cast<SizeT>(t(i));
                if(std::find(verts.begin(), verts.end(), g) == verts.end())
                    verts.push_back(g);
            }
            tets_of_body[body].push_back(t);
        }
        for(auto& [body, tets] : tets_of_body)
        {
            const auto& verts = vertices_of_body[body];
            std::vector<Vector3> local_pos;
            std::vector<IndexT>  local_fixed;
            local_pos.reserve(verts.size());
            local_fixed.reserve(verts.size());
            for(auto g : verts)
            {
                local_pos.push_back(desc.positions[g]);
                local_fixed.push_back(desc.is_fixed[g]);
            }
            std::vector<Vector4i> local_tets;
            local_tets.reserve(tets.size());
            for(auto& t : tets)
            {
                Vector4i lt;
                for(int i = 0; i < 4; ++i)
                {
                    auto it = std::find(verts.begin(), verts.end(), static_cast<SizeT>(t(i)));
                    lt(i)   = static_cast<IndexT>(std::distance(verts.begin(), it));
                }
                local_tets.push_back(lt);
            }
            auto mesh = tetmesh(span<const Vector3>(local_pos),
                                span<const Vector4i>(local_tets));
            label_surface(mesh);
            snh.apply_to(mesh, desc.moduli, desc.density);
            auto is_fixed = mesh.vertices().find<IndexT>(builtin::is_fixed);
            if(!is_fixed)
                is_fixed = mesh.vertices().create<IndexT>(builtin::is_fixed, 0);
            std::ranges::copy(local_fixed, view(*is_fixed).begin());
            if(desc.marker)
                mesh.vertices().create<IndexT>(T54LinearReporter::kMarker, 1);
            object->geometries().create(mesh);
        }
    }
    return scene;
}

void step_frame(World& world)
{
    world.advance();
    world.sync();
    gg::frame_stream().synchronize();
}

/// The ARAP element data of a scene (rest metric + rest volume), derived here
/// from the scene's own positions/connectivity.
[[nodiscard]] std::vector<double> rest_volumes(const SceneDesc&      desc,
                                               const std::vector<Vector4i>& tets)
{
    std::vector<double> V;
    V.reserve(tets.size());
    for(auto& t : tets)
    {
        auto Ds = fem::Ds<double>(vec3_at(desc.positions, t(0)),
                                  vec3_at(desc.positions, t(1)),
                                  vec3_at(desc.positions, t(2)),
                                  vec3_at(desc.positions, t(3)));
        V.push_back(lps::determinant(Ds) / 6.0);
    }
    return V;
}

[[nodiscard]] std::vector<lps::LMat<double, 3, 3>> rest_Dm_invs(
    const SceneDesc& desc, const std::vector<Vector4i>& tets)
{
    std::vector<lps::LMat<double, 3, 3>> D;
    D.reserve(tets.size());
    for(auto& t : tets)
    {
        D.push_back(fem::Dm_inv<double>(vec3_at(desc.positions, t(0)),
                                        vec3_at(desc.positions, t(1)),
                                        vec3_at(desc.positions, t(2)),
                                        vec3_at(desc.positions, t(3))));
    }
    return D;
}

// ===========================================================================
// 6) case 4's shared round data + the host side of the round
// ===========================================================================
constexpr int kRoundVertices = 4;
constexpr int kRoundDoublets = 4;
constexpr int kRoundTriplets = 10;

struct RoundData
{
    std::vector<std::int32_t> is_fixed = {1, 0, 0, 1};
    std::vector<double>       masses = {2.0, 3.0, 4.5, 5.5};
    // the shared doublet buffer: one doublet per vertex
    std::vector<std::int32_t> d_indices = {0, 1, 2, 3};
    std::vector<double>       d_values  = {1.0,  2.0,  3.0,   //
                                           4.0,  5.0,  6.0,   //
                                           7.0,  8.0,  9.0,   //
                                           10.0, 11.0, 12.0};
    // the upper-packed triplet matrix of a tet (10 blocks)
    std::vector<std::int32_t> t_rows = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
    std::vector<std::int32_t> t_cols = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
    std::vector<double>       t_values;
    // the dytopo effect's own buffers (global vertex ids)
    std::vector<std::int32_t> dy_indices = {1, 2};
    std::vector<double>       dy_values = {100.0, 101.0, 102.0, 200.0, 201.0, 202.0};
    std::vector<std::int32_t> dh_rows = {0};
    std::vector<std::int32_t> dh_cols = {3};
    std::vector<double>       dh_values;
    std::vector<double>       solution;
    std::vector<double>       dst;
    std::vector<double>       dxs;
    std::vector<double>       diag_norm;
    std::vector<double>       mass_norm;

    RoundData()
    {
        t_values.resize(kRoundTriplets * 9);
        for(int i = 0; i < kRoundTriplets; ++i)
            for(int c = 0; c < 3; ++c)
                for(int r = 0; r < 3; ++r)
                    t_values[static_cast<std::size_t>(i * 9 + c * 3 + r)] =
                        1.0 + 100.0 * i + 10.0 * r + c;
        dh_values.resize(9);
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                dh_values[static_cast<std::size_t>(c * 3 + r)] = 0.5 + 10.0 * r + c;
        solution.resize(kRoundVertices * 3);
        for(int i = 0; i < kRoundVertices * 3; ++i)
            solution[static_cast<std::size_t>(i)] = 0.25 * (i + 1) - 1.0;
        dst.resize(kRoundVertices * 3, 0.0);
        dxs.resize(kRoundVertices * 3, 0.0);
        diag_norm.resize(kRoundVertices, 0.0);
        mass_norm.resize(kRoundVertices, 0.0);
    }
};

/// The dump order, shared by the host and the device side (a mismatch in the
/// scaffolding shows up as a mismatch of the two arrays, which is what the case
/// asserts).
void host_round(RoundData& d, std::vector<double>& out)
{
    constexpr std::int32_t NV = kRoundVertices;
    constexpr std::int32_t NT = kRoundTriplets;

    auto dst  = HostDense::from_handles(d.dst.data(), NV * 3);
    auto sol  = HostDense::from_handles(d.solution.data(), NV * 3);
    auto dxs  = HostDense::from_handles(d.dxs.data(), NV * 3);
    auto mass = HostDense::from_handles(d.masses.data(), NV);
    auto dnorm = HostDense::from_handles(d.diag_norm.data(), NV);
    auto mnorm = HostDense::from_handles(d.mass_norm.data(), NV);

    auto db = HostDoublet::from_handles(d.d_indices.data(), d.d_values.data(), NV, kRoundDoublets);
    auto dy = HostDoublet::from_handles(d.dy_indices.data(), d.dy_values.data(), NV, 2);
    auto A  = HostTriplet::from_handles(
        d.t_rows.data(), d.t_cols.data(), d.t_values.data(), NV, NV, NT);
    auto dh = HostTriplet::from_handles(
        d.dh_rows.data(), d.dh_cols.data(), d.dh_values.data(), NV, NV, 1);

    auto fixed_at = [&](std::int32_t i) noexcept
    { return d.is_fixed[static_cast<std::size_t>(i)]; };
    auto put = [&](double v) { out.push_back(v); };

    for(std::int32_t i = 0; i < NV; ++i)
        linf::clear_fixed_gradient_row(dst, i, fixed_at);
    for(std::int32_t i = 0; i < kRoundDoublets; ++i)
        linf::scatter_gradient_row(dst, db, i, fixed_at);
    for(std::int32_t i = 0; i < 2; ++i)
        linf::scatter_dytopo_gradient_row(dst, dy, i, std::int32_t{0}, fixed_at);
    for(std::int32_t i = 0; i < NV; ++i)
    {
        linf::retrieve_solution_row(
            sol,
            i,
            [&](auto&& v)
            {
                for(int c = 0; c < 3; ++c)
                    dxs.template segment<3>(i * 3)(c) = v(c);
            });
    }
    for(std::int32_t i = 0; i < NT; ++i)
    {
        auto r = linf::diag_block_norm_row(A, i, std::int32_t{0}, NV, fixed_at);
        if(!r.out_of_window && r.is_diagonal)
            dnorm.write(r.index, r.value);
        put(static_cast<double>(r.index));
        put(r.value);
        put(r.out_of_window ? 1.0 : 0.0);
        put(r.is_diagonal ? 1.0 : 0.0);
    }
    for(std::int32_t i = 0; i < NV; ++i)
        linf::mass_block_norm_row(mass, mnorm, i, fixed_at);
    for(std::int32_t i = 0; i < NT; ++i)
        linf::fix_hessian_row(A, i, fixed_at);
    linf::copy_dytopo_hessian_row(A, dh, std::int32_t{0}, std::int32_t{0});

    // the dump (fixed order)
    for(std::int32_t i = 0; i < kRoundDoublets; ++i)
    {
        auto dbl = linf::read_element(db, i);
        put(static_cast<double>(dbl.index));
        for(int c = 0; c < 3; ++c)
            put(dbl.value(c));
    }
    for(std::int32_t i = 0; i < NT; ++i)
    {
        auto t = linf::read_element(A, i);
        put(static_cast<double>(t.row_index));
        put(static_cast<double>(t.col_index));
        for(int c = 0; c < 3; ++c)
            for(int r = 0; r < 3; ++r)
                put(t.value(r, c));
    }
    for(std::int32_t i = 0; i < NV * 3; ++i)
        put(dst.read(i));
    for(std::int32_t i = 0; i < NV * 3; ++i)
        put(dxs.read(i));
    for(std::int32_t i = 0; i < NV; ++i)
        put(dnorm.read(i));
    for(std::int32_t i = 0; i < NV; ++i)
        put(mnorm.read(i));
}

/// The device side of the same round: one thread, the same calls, the same order.
auto& round_kernel()
{
    static auto& k =
        lla::detail::cached_kernel<lcrt::Buffer<std::int32_t>,  // is_fixed
                                   lcrt::Buffer<double>,        // masses
                                   lcrt::Buffer<std::int32_t>,  // d_indices
                                   lcrt::Buffer<double>,        // d_values
                                   lcrt::Buffer<std::int32_t>,  // dy_indices
                                   lcrt::Buffer<double>,        // dy_values
                                   lcrt::Buffer<std::int32_t>,  // t_rows
                                   lcrt::Buffer<std::int32_t>,  // t_cols
                                   lcrt::Buffer<double>,        // t_values
                                   lcrt::Buffer<std::int32_t>,  // dh_rows
                                   lcrt::Buffer<std::int32_t>,  // dh_cols
                                   lcrt::Buffer<double>,        // dh_values
                                   lcrt::Buffer<double>,        // dst
                                   lcrt::Buffer<double>,        // solution
                                   lcrt::Buffer<double>,        // dxs
                                   lcrt::Buffer<double>,        // diag_norm
                                   lcrt::Buffer<double>,        // mass_norm
                                   lcrt::Buffer<double>>(       // out
            [](lcrt::BufferVar<std::int32_t> is_fixed,
               lcrt::BufferVar<double>       masses,
               lcrt::BufferVar<std::int32_t> d_indices,
               lcrt::BufferVar<double>       d_values,
               lcrt::BufferVar<std::int32_t> dy_indices,
               lcrt::BufferVar<double>       dy_values,
               lcrt::BufferVar<std::int32_t> t_rows,
               lcrt::BufferVar<std::int32_t> t_cols,
               lcrt::BufferVar<double>       t_values,
               lcrt::BufferVar<std::int32_t> dh_rows,
               lcrt::BufferVar<std::int32_t> dh_cols,
               lcrt::BufferVar<double>       dh_values,
               lcrt::BufferVar<double>       dst_buf,
               lcrt::BufferVar<double>       solution,
               lcrt::BufferVar<double>       dxs_buf,
               lcrt::BufferVar<double>       diag_norm_buf,
               lcrt::BufferVar<double>       mass_norm_buf,
               lcrt::BufferVar<double>       out) noexcept
            {
                auto fixed_at = [&](auto&& i) noexcept
                { return is_fixed->read(i); };

                auto dst = DenseDeviceView::from_handles(dst_buf, std::int32_t{kRoundVertices * 3});
                auto sol = CDenseDeviceView::from_handles(solution,
                                                          std::int32_t{kRoundVertices * 3});
                auto dxs = DenseDeviceView::from_handles(dxs_buf, std::int32_t{kRoundVertices * 3});
                auto mass = CDenseDeviceView::from_handles(masses, std::int32_t{kRoundVertices});
                auto dnorm = DenseDeviceView::from_handles(diag_norm_buf,
                                                           std::int32_t{kRoundVertices});
                auto mnorm = DenseDeviceView::from_handles(mass_norm_buf,
                                                           std::int32_t{kRoundVertices});
                auto db = CDoubletDeviceView::from_handles(d_indices,
                                                           d_values,
                                                           std::int32_t{kRoundVertices},
                                                           std::int32_t{kRoundDoublets});
                auto dy = CDoubletDeviceView::from_handles(dy_indices,
                                                           dy_values,
                                                           std::int32_t{kRoundVertices},
                                                           std::int32_t{2});
                auto A = TripletDeviceView::from_handles(t_rows,
                                                         t_cols,
                                                         t_values,
                                                         std::int32_t{kRoundVertices},
                                                         std::int32_t{kRoundVertices},
                                                         std::int32_t{kRoundTriplets});
                auto dh = TripletDeviceView::from_handles(dh_rows,
                                                          dh_cols,
                                                          dh_values,
                                                          std::int32_t{kRoundVertices},
                                                          std::int32_t{kRoundVertices},
                                                          std::int32_t{1});

                uint32_t put = 0u;
                auto     push = [&](auto&& v) noexcept { out->write(put++, v); };

                for(std::int32_t i = 0; i < kRoundVertices; ++i)
                    fem_linear_subsystem_detail::clear_fixed_gradient_row(dst, i, fixed_at);
                for(std::int32_t i = 0; i < kRoundDoublets; ++i)
                    fem_linear_subsystem_detail::scatter_gradient_row(dst, db, i, fixed_at);
                for(std::int32_t i = 0; i < 2; ++i)
                    fem_linear_subsystem_detail::scatter_dytopo_gradient_row(
                        dst, dy, i, std::int32_t{0}, fixed_at);
                for(std::int32_t i = 0; i < kRoundVertices; ++i)
                {
                    fem_linear_subsystem_detail::retrieve_solution_row(
                        sol,
                        i,
                        [&](auto&& v) noexcept
                        {
                            for(int c = 0; c < 3; ++c)
                                dxs.segment<3>(i * 3)(c) = v(c);
                        });
                }
                for(std::int32_t i = 0; i < kRoundTriplets; ++i)
                {
                    auto r = fem_linear_subsystem_detail::diag_block_norm_row(
                        A, i, std::int32_t{0}, std::int32_t{kRoundVertices}, fixed_at);
                    $if(scalar_detail::pred_and(!r.out_of_window, r.is_diagonal))
                    {
                        dnorm.write(r.index, r.value);
                    };
                    push(lcrt::cast<double>(r.index));
                    push(r.value);
                    push(lcrt::select(0.0, 1.0, r.out_of_window));
                    push(lcrt::select(0.0, 1.0, r.is_diagonal));
                }
                for(std::int32_t i = 0; i < kRoundVertices; ++i)
                    fem_linear_subsystem_detail::mass_block_norm_row(mass, mnorm, i, fixed_at);
                for(std::int32_t i = 0; i < kRoundTriplets; ++i)
                    fem_linear_subsystem_detail::fix_hessian_row(A, i, fixed_at);
                fem_linear_subsystem_detail::copy_dytopo_hessian_row(
                    A, dh, std::int32_t{0}, std::int32_t{0});

                for(std::int32_t i = 0; i < kRoundDoublets; ++i)
                {
                    auto dbl = fem_linear_subsystem_detail::read_element(db, i);
                    push(lcrt::cast<double>(dbl.index));
                    for(int c = 0; c < 3; ++c)
                        push(dbl.value(c));
                }
                for(std::int32_t i = 0; i < kRoundTriplets; ++i)
                {
                    auto t = fem_linear_subsystem_detail::read_element(A, i);
                    push(lcrt::cast<double>(t.row_index));
                    push(lcrt::cast<double>(t.col_index));
                    for(int c = 0; c < 3; ++c)
                        for(int r = 0; r < 3; ++r)
                            push(t.value(r, c));
                }
                for(std::int32_t i = 0; i < kRoundVertices * 3; ++i)
                    push(dst.read(i));
                for(std::int32_t i = 0; i < kRoundVertices * 3; ++i)
                    push(dxs.read(i));
                for(std::int32_t i = 0; i < kRoundVertices; ++i)
                    push(dnorm.read(i));
                for(std::int32_t i = 0; i < kRoundVertices; ++i)
                    push(mnorm.read(i));
            });
    return k;
}
}  // namespace t54

// ===========================================================================
//  CASE 1 — the five-tet assembly, MatrixConverter, SpMV and the COO content
// ===========================================================================
TEST_CASE("fem_linear_assembly_five_tet", "[luisa][fem_linear]")
{
    using namespace t54;
    test_runtime();

    auto        mesh   = five_tet();
    const SizeT n_elem = mesh.tets.size();
    const SizeT n_vert = mesh.X.size();
    const SizeT n_dof  = n_vert * 3;
    const SizeT tri_per_elem = 10;  // 4 * 5 / 2 upper-triangle blocks
    const SizeT dou_per_elem = 4;
    const SizeT n_tri = n_elem * tri_per_elem;
    const SizeT n_dou = n_elem * dou_per_elem;

    // ---- host element data (the ported T5.2 ARAP routines) -------------------
    std::vector<lps::LMat<double, 12, 12>> H(n_elem);
    std::vector<lps::LVec<double, 12>>     G(n_elem);
    std::vector<double> h_H(n_elem * 144, 0.0), h_G(n_elem * 12, 0.0);
    double              min_volume = 1e9;
    for(SizeT e = 0; e < n_elem; ++e)
    {
        const double vol = element_volume(mesh, e);
        min_volume       = std::min(min_volume, std::abs(vol));
        REQUIRE(std::abs(vol) > 1e-3);  // every element must be non-degenerate
        H[e] = element_hessian(mesh, e);
        G[e] = element_gradient(mesh, e);
        for(int c = 0; c < 12; ++c)
            for(int r = 0; r < 12; ++r)
                h_H[e * 144 + static_cast<SizeT>(c * 12 + r)] = H[e](r, c);
        for(int i = 0; i < 12; ++i)
            h_G[e * 12 + static_cast<SizeT>(i)] = G[e](i);
    }
    std::printf("[t54] five-tet: %zu elements, min |V| = %.6e\n", n_elem, min_volume);
    CHECK(min_volume > 1e-3);

    // ---- the fan-out slices (the ported slice lookup) -----------------------
    OffsetCountCollection<IndexT> t_counts, d_counts;
    t_counts.resize(n_elem);
    d_counts.resize(n_elem);
    for(auto& c : t_counts.counts())
        c = static_cast<IndexT>(tri_per_elem);
    for(auto& c : d_counts.counts())
        c = static_cast<IndexT>(dou_per_elem);
    t_counts.scan();
    d_counts.scan();
    CHECK(t_counts.total_count() == static_cast<IndexT>(n_tri));
    CHECK(d_counts.total_count() == static_cast<IndexT>(n_dou));
    std::vector<IndexT> t_off(n_elem), d_off(n_elem);
    for(SizeT e = 0; e < n_elem; ++e)
    {
        auto [to, tc] = linf::slice_of(t_counts, e);
        auto [d_o, dc] = linf::slice_of(d_counts, e);
        CHECK(to == static_cast<IndexT>(e * tri_per_elem));
        CHECK(tc == static_cast<IndexT>(tri_per_elem));
        CHECK(d_o == static_cast<IndexT>(e * dou_per_elem));
        CHECK(dc == static_cast<IndexT>(dou_per_elem));
        t_off[e] = to;
        d_off[e] = d_o;
    }

    // ---- the host instantiation of the same assembly (bitwise reference) ----
    std::vector<std::int32_t> h_rows(n_tri, -1), h_cols(n_tri, -1);
    std::vector<double>       h_vals(n_tri * 9, 0.0);
    std::vector<std::int32_t> h_didx(n_dou, -1);
    std::vector<double>       h_dvals(n_dou * 3, 0.0);
    {
        auto A = HostTriplet::from_handles(
            h_rows.data(), h_cols.data(), h_vals.data(), n_vert, n_vert, n_tri);
        auto D = HostDoublet::from_handles(h_didx.data(), h_dvals.data(), n_vert, n_dou);
        for(SizeT e = 0; e < n_elem; ++e)
        {
            lps::LVec<IndexT, 4> tet;
            for(int i = 0; i < 4; ++i)
                tet(i) = mesh.tets[e](i);
            linf::write_element_gradient(D, d_off[e], tet, G[e]);
            linf::write_element_hessian(A, t_off[e], tet, H[e]);
        }
    }

    // ---- the device assembly ------------------------------------------------
    lla::LTripletMatrix<double, 3, 3> A;
    A.reshape(static_cast<int>(n_vert), static_cast<int>(n_vert));
    A.resize_triplets(n_tri);
    lla::LDoubletVector<double, 3> D;
    D.reshape(static_cast<int>(n_vert));
    D.resize_doublets(n_dou);

    lud::LDeviceBuffer<LVector4i>    d_tets;
    lud::LDeviceBuffer<double>       d_H, d_G;
    lud::LDeviceBuffer<std::int32_t> d_toff, d_doff;
    {
        std::vector<LVector4i> tets;
        tets.reserve(n_elem);
        for(auto& t : mesh.tets)
            tets.push_back(to_luisa(t));
        d_tets.resize(tets.size());
        d_tets.view().copy_from(span<const LVector4i>(tets));
        d_H.resize(h_H.size());
        d_H.view().copy_from(span<const double>(h_H));
        d_G.resize(h_G.size());
        d_G.view().copy_from(span<const double>(h_G));
        d_toff.resize(t_off.size());
        d_toff.view().copy_from(span<const std::int32_t>(t_off));
        d_doff.resize(d_off.size());
        d_doff.view().copy_from(span<const std::int32_t>(d_off));
    }

    auto& k = five_tet_assembly_kernel();
    gg::frame_stream().record(k(d_tets.view(),
                                d_H.view(),
                                d_G.view(),
                                d_toff.view(),
                                d_doff.view(),
                                A.row_indices(),
                                A.col_indices(),
                                A.values(),
                                D.indices(),
                                D.values(),
                                static_cast<uint32_t>(n_vert),
                                static_cast<uint32_t>(n_tri),
                                static_cast<uint32_t>(n_dou),
                                static_cast<uint32_t>(n_elem))
                               .dispatch(static_cast<uint32_t>(n_elem)));
    gg::frame_stream().synchronize();

    // device == host for the assembled (row, col, block) triples
    {
        std::vector<std::int32_t> rows(n_tri), cols(n_tri);
        std::vector<double>       vals(n_tri * 9);
        A.copy_to_row_indices(span<std::int32_t>(rows));
        A.copy_to_col_indices(span<std::int32_t>(cols));
        A.copy_to_values(span<double>(vals));
        double worst = 0.0;
        for(SizeT i = 0; i < n_tri; ++i)
        {
            CHECK(rows[i] == h_rows[i]);
            CHECK(cols[i] == h_cols[i]);
            for(int j = 0; j < 9; ++j)
                worst = std::max(worst,
                                 rel_err(vals[i * 9 + static_cast<SizeT>(j)],
                                         h_vals[i * 9 + static_cast<SizeT>(j)]));
        }
        std::printf("[t54] five-tet device-vs-host triplet blocks: worst rel %.3e\n", worst);
        CHECK(worst == 0.0);  // identical arithmetic on both flavours
        // and the gradient doublets
        std::vector<lla::VectorDoublet<double, 3>> dou(n_dou);
        D.copy_to_doublets(span<lla::VectorDoublet<double, 3>>(dou));
        double dworst = 0.0;
        for(SizeT i = 0; i < n_dou; ++i)
        {
            CHECK(dou[i].index == h_didx[i]);
            for(int c = 0; c < 3; ++c)
                dworst = std::max(dworst,
                                  rel_err(dou[i].value(c), h_dvals[i * 3 + static_cast<SizeT>(c)]));
        }
        std::printf("[t54] five-tet device-vs-host doublets: worst rel %.3e\n", dworst);
        CHECK(dworst == 0.0);
    }

    // ---- MatrixConverter -> Spmv -------------------------------------------
    lla::MatrixConverter<double, 3> converter;
    converter.ge2sym(A);  // the reference's step: already upper-packed -> a no-op
    CHECK(A.triplet_count() == n_tri);
    lla::LBCOOMatrix<double, 3, 3> B;
    converter.convert(A, B);

    std::vector<double> x(n_dof);
    for(SizeT i = 0; i < n_dof; ++i)
        x[i] = 0.3 * static_cast<double>(i) - 1.7;

    lla::LDenseVector<double> x_dev, y_dev;
    x_dev.resize(n_dof);
    x_dev.copy_from(span<const double>(x));
    y_dev.resize(n_dof);
    y_dev.fill(0.0);
    lla::Spmv spmv;
    spmv.rbk_sym_spmv(1.0, B.cview(), x_dev.cview(), 0.0, y_dev.view());
    lud::main_stream().synchronize();

    std::vector<double> y(n_dof);
    y_dev.copy_to(span<double>(y));

    // ---- the independent dense oracle (Eigen, a different traversal order) --
    Eigen::MatrixXd A_dense = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_dof),
                                                    static_cast<Eigen::Index>(n_dof));
    for(SizeT e = 0; e < n_elem; ++e)
    {
        for(int ii = 0; ii < 4; ++ii)
            for(int jj = 0; jj < 4; ++jj)
            {
                const auto gi = static_cast<Eigen::Index>(mesh.tets[e](ii)) * 3;
                const auto gj = static_cast<Eigen::Index>(mesh.tets[e](jj)) * 3;
                for(int r = 0; r < 3; ++r)
                    for(int c = 0; c < 3; ++c)
                        A_dense(gi + r, gj + c) += H[e](3 * ii + r, 3 * jj + c);
            }
    }
    Eigen::VectorXd x_eig(static_cast<Eigen::Index>(n_dof));
    for(SizeT i = 0; i < n_dof; ++i)
        x_eig[static_cast<Eigen::Index>(i)] = x[i];
    Eigen::VectorXd y_eig = A_dense * x_eig;

    double worst_spmv = 0.0;
    for(SizeT i = 0; i < n_dof; ++i)
        worst_spmv = std::max(worst_spmv, rel_err(y[i], y_eig[static_cast<Eigen::Index>(i)]));
    std::printf("[t54] five-tet dense-vs-SpMV: worst rel/abs %.6e (|y|max %.6e)\n",
                worst_spmv,
                y_eig.cwiseAbs().maxCoeff());
    CHECK(worst_spmv <= 1e-5);

    // ---- the COO content ---------------------------------------------------
    std::vector<lla::MatrixTriplet<double, 3, 3>> coo(B.triplet_count());
    B.copy_to_triplets(span<lla::MatrixTriplet<double, 3, 3>>(coo));
    REQUIRE(coo.size() > 0);

    // (a) sorted, no duplicates
    bool sorted = true;
    for(SizeT i = 1; i < coo.size(); ++i)
    {
        if(coo[i].row_index < coo[i - 1].row_index
           || (coo[i].row_index == coo[i - 1].row_index
               && coo[i].col_index <= coo[i - 1].col_index))
            sorted = false;
    }
    CHECK(sorted);

    // (b) every stored (row, col) is upper-triangular and != (-1, -1)
    bool all_upper = true;
    for(auto& t : coo)
        all_upper &= (t.row_index <= t.col_index);
    CHECK(all_upper);

    // (c) the block content equals the hand-derived sum over the elements
    std::map<std::pair<int, int>, Eigen::Matrix3d> expected;
    for(SizeT e = 0; e < n_elem; ++e)
    {
        for(int ii = 0; ii < 4; ++ii)
            for(int jj = ii; jj < 4; ++jj)
            {
                int  li = mesh.tets[e](ii);
                int  lj = mesh.tets[e](jj);
                bool keep = li < lj;
                int  gr   = keep ? li : lj;
                int  gc   = keep ? lj : li;
                Eigen::Matrix3d blk = Eigen::Matrix3d::Zero();
                for(int r = 0; r < 3; ++r)
                    for(int c = 0; c < 3; ++c)
                        blk(r, c) = keep ? H[e](3 * ii + r, 3 * jj + c)
                                         : H[e](3 * jj + r, 3 * ii + c);
                auto [it, inserted] = expected.emplace(std::make_pair(gr, gc), blk);
                if(!inserted)
                    it->second += blk;
            }
    }
    CHECK(expected.size() == coo.size());
    double worst_coo = 0.0;
    int    printed   = 0;
    for(auto& t : coo)
    {
        auto it = expected.find({t.row_index, t.col_index});
        REQUIRE(it != expected.end());
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
            {
                const auto e = rel_err(t.value(r, c), it->second(r, c));
                if(e > 1e-12 && printed < 6)
                {
                    std::printf("[t54][debug] (%d,%d) [%d,%d] BCOO %.17g expected %.17g\n",
                                t.row_index, t.col_index, r, c, t.value(r, c),
                                it->second(r, c));
                    ++printed;
                }
                worst_coo = std::max(worst_coo, e);
            }
    }
    std::printf("[t54] five-tet COO content: %zu blocks, worst rel %.3e\n",
                coo.size(),
                worst_coo);
    CHECK(worst_coo <= 1e-12);

    // (d) no structural zero *introduced* by the assembly: a stored block may sum
    //     to exactly zero (the projected ARAP Hessian of this state does produce
    //     one), but the set of zero blocks must be exactly the oracle's -- i.e. the
    //     assembly neither adds a zero entry nor drops a non-zero one.  Measured
    //     and printed, not assumed.
    SizeT zero_blocks = 0;
    for(auto& t : coo)
    {
        double block_norm = 0.0;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                block_norm += std::abs(t.value(r, c));
        if(block_norm == 0.0)
            ++zero_blocks;
    }
    SizeT expected_zero_blocks = 0;
    for(auto& [key, blk] : expected)
    {
        double block_norm = 0.0;
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < 3; ++c)
                block_norm += std::abs(blk(r, c));
        if(block_norm == 0.0)
            ++expected_zero_blocks;
    }
    std::printf("[t54] five-tet COO zero blocks: %zu (hand-derived %zu, of %zu stored)\n",
                zero_blocks,
                expected_zero_blocks,
                coo.size());
    CHECK(zero_blocks == expected_zero_blocks);
}

// ===========================================================================
//  CASE 2 — the dof layout and the extent budget
// ===========================================================================
TEST_CASE("fem_linear_layout", "[luisa][fem_linear]")
{
    using namespace t54;
    test_runtime();

    // ---- (A) the factored uid-ordering + dof layout --------------------------
    {
        const std::vector<U64>   uids{7ull, 1ull, 100ull, 3ull};
        const std::vector<SizeT> counts{30u, 60u, 9u, 12u};
        auto                     layout = lgls::plan_dof_layout(span<const U64>(uids),
                                               span<const SizeT>(counts));
        // hand-derived: uid 1 (60) -> uid 3 (12) -> uid 7 (30) -> uid 100 (9)
        const std::vector<U64>   e_uids{1ull, 3ull, 7ull, 100ull};
        const std::vector<SizeT> e_counts{60u, 12u, 30u, 9u};
        const std::vector<SizeT> e_offsets{0u, 60u, 72u, 102u};
        CHECK(layout.uids == e_uids);
        CHECK(layout.counts == e_counts);
        CHECK(layout.offsets == e_offsets);
        CHECK(layout.total_count() == 111u);
        std::printf("[t54] layout: uids %llu,%llu,%llu,%llu offsets %zu,%zu,%zu,%zu total %zu\n",
                    static_cast<unsigned long long>(layout.uids[0]),
                    static_cast<unsigned long long>(layout.uids[1]),
                    static_cast<unsigned long long>(layout.uids[2]),
                    static_cast<unsigned long long>(layout.uids[3]),
                    layout.offsets[0],
                    layout.offsets[1],
                    layout.offsets[2],
                    layout.offsets[3],
                    layout.total_count());
    }

    // ---- (B) the factored extent budget -------------------------------------
    {
        // a mixture: the kinetic block, three reporter classes and the dytopo effect
        const std::vector<SizeT> grad_counts{8u, 4u, 2u};   // elastics, animator, mock
        const std::vector<SizeT> hess_counts{20u, 0u, 3u};
        auto                     full = linf::plan_extent(8u,
                                      false,           // gradient_only
                                      true,            // has_complement
                                      span<const SizeT>(grad_counts),
                                      span<const SizeT>(hess_counts),
                                      0u,              // dytopo gradient
                                      0u);             // dytopo hessian
        CHECK(full.kinetic_gradient_count == 8u);
        CHECK(full.kinetic_hessian_count == 8u);
        CHECK(full.reporter_gradient_count == 14u);
        CHECK(full.reporter_hessian_count == 23u);
        CHECK(full.gradient_offset == 22u);
        CHECK(full.hessian_offset == 31u);
        CHECK(full.dof_count == 24u);
        CHECK(full.check_gradient_only(false));

        auto only = linf::plan_extent(8u,
                                      true,  // gradient_only
                                      true,
                                      span<const SizeT>(grad_counts),
                                      span<const SizeT>(hess_counts),
                                      0u,
                                      0u);
        CHECK(only.kinetic_hessian_count == 0u);
        CHECK(only.reporter_hessian_count == 0u);
        CHECK(only.hessian_offset == 0u);
        CHECK(only.gradient_offset == 22u);
        CHECK(only.dof_count == 24u);
        CHECK(only.check_gradient_only(true));

        // the invalid combination is a *predicate* here (the reference's
        // UIPC_ASSERT aborts, so a test may not call it): a non-empty Hessian
        // budget with gradient_only is exactly what it rejects
        auto bad = linf::plan_extent(8u, true, true, span<const SizeT>(grad_counts),
                                     span<const SizeT>(hess_counts), 0u, 4u);
        CHECK(!bad.check_gradient_only(true));
        std::printf("[t54] extent budget: full hessian_offset %zu / gradient_offset %zu; "
                    "gradient_only hessian_offset %zu\n",
                    full.hessian_offset,
                    full.gradient_offset,
                    only.hessian_offset);
    }

    // ---- (C) the real chain: two bodies, the real layout ---------------------
    {
        T54Probe::reset();
        T54LinearReporter::reset();
        auto   desc = two_body_desc();
        desc.tets   = {{0, 1, 2, 3}, {4, 5, 6, 7}};
        auto   scene = build_scene(desc);
        Engine engine{"luisa", workspace_path("uipc_luisa_t54a_layout")};
        World  world{engine};
        world.init(scene);
        REQUIRE(world.is_valid());
        REQUIRE(T54Probe::gls != nullptr);
        REQUIRE(T54Probe::femsub != nullptr);
        REQUIRE(T54Probe::femsub->is_valid());

        // the subsystem's identity (issue #271's uid rule)
        CHECK(T54Probe::femsub->uid() == 1ull);

        // the port's `init()` is public on purpose (see the header's deviation
        // note): it is the dof-layout entry `SimEngine::do_init` calls (T8.4)
        T54Probe::gls->init();
        CHECK(T54Probe::gls->dof_count() == 24u);
        CHECK(T54Probe::femsub->dof_offset() == 0);
        CHECK(T54Probe::femsub->dof_count() == 24);

        // the per-geometry dof offset/count attributes `receive_init_dof_info`
        // writes (bodies in method order)
        std::vector<std::pair<IndexT, IndexT>> dofs;
        REQUIRE(T54Probe::geo_slots.size() == 2u);
        for(auto& slot : T54Probe::geo_slots)
        {
            auto& geo = slot->geometry();
            auto  off = geo.meta().find<IndexT>(builtin::dof_offset);
            auto  cnt = geo.meta().find<IndexT>(builtin::dof_count);
            REQUIRE(off != nullptr);
            REQUIRE(cnt != nullptr);
            dofs.push_back({view(*off)[0], view(*cnt)[0]});
        }
        REQUIRE(dofs.size() == 2u);
        CHECK(dofs[0].first == 0);
        CHECK(dofs[0].second == 12);
        CHECK(dofs[1].first == 12);
        CHECK(dofs[1].second == 12);

        // the mock reporter is enabled here (its marker attribute is present):
        // its init() ran exactly once through this scene's `gls->init()`
        CHECK(T54LinearReporter::enabled);
        CHECK(T54LinearReporter::init_count == 1);

        // the *full* (gradient_only == false) extent path: a test-local
        // `GlobalLinearSystem::Impl` (a public nested class) drives the very same
        // subsystem, so `report_extent` runs with the Hessian counted
        gg::GlobalLinearSystem::Impl impl;
        impl.diag_subsystems.register_sim_system(*T54Probe::femsub);
        impl.init();
        CHECK(impl.diag_dof_offsets_counts.total_count() == 24u);
        CHECK(impl.subsystem_infos.size() == 1u);
        CHECK(impl.subsystem_infos[0].is_diag);
        CHECK(impl.subsystem_infos[0].index == 0u);
        CHECK(impl._update_subsystem_extent());
        // hand-derived: kinetic 8 + elastics 2*10 + animator 0 + mock 3 = 31
        // (and the mock's hessian slice is exactly its 3 reported blocks)
        std::printf("[t54] layout diag: mock(enabled %d, reported g=%zu h=%zu, extent_calls %d), "
                    "real blocks %d\n",
                    T54LinearReporter::enabled ? 1 : 0,
                    T54LinearReporter::last_reported_gradient,
                    T54LinearReporter::last_reported_hessian,
                    T54LinearReporter::report_extent_count,
                    static_cast<int>(impl.subsystem_triplet_offsets_counts.counts()[0]));
        CHECK(impl.subsystem_triplet_offsets_counts.counts()[0] == 31);
        CHECK(impl.diag_dof_offsets_counts.counts()[0] == 24);
        CHECK(T54LinearReporter::report_extent_count >= 1);
        CHECK(T54LinearReporter::last_gradient_only == false);
        std::printf("[t54] layout: real hessian blocks %d (hand-derived 31), dofs %zu\n",
                    static_cast<int>(impl.subsystem_triplet_offsets_counts.counts()[0]),
                    static_cast<size_t>(impl.diag_dof_offsets_counts.total_count()));
    }

    // ---- (D) `ReportExtentInfo::check`'s legal path (the invalid one aborts) --
    {
        gg::FEMLinearSubsystem::ReportExtentInfo info;
        info.gradient_count(4);
        info.hessian_count(10);
        CHECK(info.gradient_only() == false);
        info.check("t54_mock");  // must not throw
        CHECK(true);
    }
}

// ===========================================================================
//  CASE 3 — a scene through the public API: the new systems stay inert, and
//  the gradient-only assembly matches the frontend's laws
// ===========================================================================
TEST_CASE("fem_linear_reporters", "[luisa][fem_linear]")
{
    using namespace t54;
    test_runtime();

    auto desc = one_body_desc();
    desc.is_fixed[0] = 1;  // one fixed vertex (its assembled gradient must be 0)
    const double dt  = desc.dt;

    T54Probe::reset();
    T54LinearReporter::reset();
    auto   scene = build_scene(desc);
    Engine engine{"luisa", workspace_path("uipc_luisa_t54a_reporters")};
    World  world{engine};
    world.init(scene);
    REQUIRE(world.is_valid());

    // --- the new systems all build and are valid ----------------------------
    REQUIRE(T54Probe::gls != nullptr);
    CHECK(T54Probe::gls->is_valid());
    REQUIRE(T54Probe::femsub != nullptr);
    CHECK(T54Probe::femsub->is_valid());
    // `FEMLinearSubsystem` *is* a `DiagLinearSubsystem` (the reference's base)
    CHECK(dynamic_cast<gg::DiagLinearSubsystem*>(T54Probe::femsub) != nullptr);
    REQUIRE(T54Probe::elastics != nullptr);
    CHECK(T54Probe::elastics->is_valid());
    REQUIRE(T54Probe::animator != nullptr);
    // The animator requires both a FiniteElement *and* a Constraint constitution
    // type (cookbook §7.18 rule 3): this scene has no constraint, so the system is
    // registered but invalidated.  Assert exactly that.
    CHECK(!T54Probe::animator->is_valid());

    // nothing has solved anything: the linear system is inert until T8.3 wires it
    CHECK(T54Probe::gls->last_solve_iterations() == 0u);
    CHECK(T54Probe::femsub->uid() == 1ull);

    // --- one advance must not disturb the pipeline --------------------------
    step_frame(world);
    CHECK(world.is_valid());
    CHECK(T54Probe::gls->last_solve_iterations() == 0u);

    // --- the layout (one body) + the reporter chain -------------------------
    T54Probe::gls->init();
    CHECK(T54Probe::gls->dof_count() == 24u);
    CHECK(T54Probe::femsub->dof_offset() == 0);
    CHECK(T54Probe::femsub->dof_count() == 24);
    CHECK(T54LinearReporter::init_count == 1);
    CHECK(T54LinearReporter::enabled);                   // its marker was found
    CHECK(T54LinearReporter::report_extent_count == 0);  // nothing reported yet

    // --- deform the state through the public accessor feature, so the
    //     *constitution*'s gradient is non-zero at the point the assembly
    //     evaluates it (at the rest state SNH's dE/dx is exactly 0, which would
    //     make the constitution term of the oracle below vacuous) --------------
    {
        auto feature = world.features().find<FiniteElementStateAccessorFeature>();
        REQUIRE(feature != nullptr);
        auto state_geo = feature->create_geometry();
        auto pos = state_geo.vertices().create<Vector3>(builtin::position, Vector3::Zero());
        state_geo.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());
        feature->copy_to(state_geo);
        auto pos_view = view(*pos);
        REQUIRE(pos_view.size() == 8u);
        for(SizeT i = 0; i < 8u; ++i)
        {
            const auto p = desc.positions[i];
            pos_view[i]  = Vector3{p.x() + 0.03 * (static_cast<double>(i) + 1.0),
                                   p.y() - 0.02 * static_cast<double>(i),
                                   p.z() + 0.01 * (static_cast<double>(i) + 2.0)};
        }
        feature->copy_from(state_geo);
        gg::frame_stream().synchronize();
    }

    // --- the gradient-only assembly through the public `compute_gradient` ----
    lla::LDenseVector<double> grad;
    grad.resize(24);
    grad.fill(0.0);
    gg::GlobalLinearSystem::ComputeGradientInfo info;
    info.flags(gg::GlobalLinearSystem::ComponentFlags::All);
    info.buffer_view(grad.view());
    T54Probe::gls->compute_gradient(info);
    gg::frame_stream().synchronize();

    std::vector<double> g(24, 0.0);
    grad.copy_to(span<double>(g));

    CHECK(T54LinearReporter::report_extent_count >= 1);
    CHECK(T54LinearReporter::last_gradient_only == true);
    CHECK(T54LinearReporter::assemble_count >= 1);
    // the mock's own slice of the shared doublet buffer: exactly its 2 reported
    // doublets, and an *empty* Hessian view in gradient-only mode
    CHECK(T54LinearReporter::last_gradient_size == T54LinearReporter::kGradientCount);
    CHECK(T54LinearReporter::last_hessian_size == 0u);

    // --- the frontend's own laws -------------------------------------------
    auto& method = *T54Probe::fem;
    auto  xs     = read_vec3(method.xs());
    auto  xt     = read_vec3(method.x_tildes());
    auto  masses = read_device(method.masses());
    auto  isf    = read_device(method.is_fixed());
    REQUIRE(xs.size() == 8u);
    REQUIRE(xt.size() == 8u);
    REQUIRE(masses.size() == 8u);
    REQUIRE(isf.size() == 8u);

    const auto volumes = rest_volumes(desc, desc.tets);
    const auto Dm_invs = rest_Dm_invs(desc, desc.tets);
    // the frontend's SNH reparameterisation (`StableNeoHookean::apply_to`):
    //   snh_mu = 4 mu / 3,  snh_lambda = lambda + 5 mu / 6
    // (a hand-derived oracle has to read the *stored* moduli, not the arguments of
    //  `apply_to` -- measured: the mismatch without this is 1.8e-2 relative)
    const auto snh_mu     = 4.0 * desc.moduli.mu() / 3.0;
    const auto snh_lambda = desc.moduli.lambda() + 5.0 * desc.moduli.mu() / 6.0;

    std::vector<double> kinetic_term(24, 0.0);
    std::vector<double> snh_term(24, 0.0);
    std::vector<double> expected(24, 0.0);
    // kinetic: `G = m (x - x~)` for a non-fixed vertex (T5.3 rule 1)
    for(SizeT v = 0; v < 8u; ++v)
    {
        if(isf[v] != 0)
            continue;
        for(int c = 0; c < 3; ++c)
        {
            kinetic_term[3 * v + static_cast<SizeT>(c)] =
                masses[v] * (xs[v](c) - xt[v](c));
            expected[3 * v + static_cast<SizeT>(c)] +=
                kinetic_term[3 * v + static_cast<SizeT>(c)];
        }
    }
    // the StableNeoHookean constitutions' element gradients, scattered at the
    // element's global vertex indices (the same moduli/dt the scene's `apply_to`
    // used).  At the rest state F = I SNH's gradient is *not* zero, so this term is
    // load-bearing in the comparison below.
    for(SizeT e = 0; e < desc.tets.size(); ++e)
    {
        const auto& t = desc.tets[e];
        auto        G = snh_3d_detail::tet_gradient<double>(snh_mu,
                                                   snh_lambda,
                                                   dt,
                                                   volumes[e],
                                                   vec3_at(xs, t(0)),
                                                   vec3_at(xs, t(1)),
                                                   vec3_at(xs, t(2)),
                                                   vec3_at(xs, t(3)),
                                                   Dm_invs[e]);
        for(int i = 0; i < 4; ++i)
        {
            const auto v = static_cast<SizeT>(t(i));
            if(isf[v] != 0)
                continue;
            for(int c = 0; c < 3; ++c)
            {
                snh_term[3 * v + static_cast<SizeT>(c)] += G(3 * i + c);
                expected[3 * v + static_cast<SizeT>(c)] += G(3 * i + c);
            }
        }
    }
    // the mock reporter's writes (vertices 1 and 2, non-fixed)
    for(int c = 0; c < 3; ++c)
    {
        expected[3 + static_cast<SizeT>(c)] += T54LinearReporter::kGradientValues[static_cast<SizeT>(c)];
        expected[6 + static_cast<SizeT>(c)] += T54LinearReporter::kGradientValues[static_cast<SizeT>(3 + c)];
    }

    // the fixed vertex is *exactly* zero (the reference's k1 kernel)
    for(int c = 0; c < 3; ++c)
        CHECK(g[static_cast<SizeT>(c)] == 0.0);

    double worst = 0.0, gmax = 0.0;
    for(SizeT i = 0; i < 24u; ++i)
    {
        worst = std::max(worst, rel_err(g[i], expected[i]));
        gmax  = std::max(gmax, std::abs(expected[i]));
    }
    std::printf("[t54] reporters: assembled gradient worst rel %.6e (max |g| %.6e, mock %d/%d)\n",
                worst,
                gmax,
                static_cast<int>(T54LinearReporter::last_gradient_size),
                static_cast<int>(T54LinearReporter::last_hessian_size));
    CHECK(gmax > 1.0);  // non-vacuous: the expected gradient is not trivial
    CHECK(worst <= 1e-9);
}

// ===========================================================================
//  CASE 4 — every factored assembly routine: host `double` vs device DSL
// ===========================================================================
TEST_CASE("fem_linear_device_vs_host", "[luisa][fem_linear]")
{
    using namespace t54;
    test_runtime();

    RoundData           data;
    std::vector<double> host_out;
    host_round(data, host_out);
    REQUIRE(host_out.size() > 100u);

    // --- the round's *semantics*, hand-derived on the host state (the device
    //     side is then required to be bitwise equal) -------------------------
    // (a) clear_fixed_gradient_row + the scatter gates: a fixed vertex's row is 0,
    //     a non-fixed one carries its doublets (shared + dytopo)
    CHECK(data.dst[0] == 0.0);
    CHECK(data.dst[1] == 0.0);
    CHECK(data.dst[2] == 0.0);
    CHECK(data.dst[3] == 4.0 + 100.0);
    CHECK(data.dst[4] == 5.0 + 101.0);
    CHECK(data.dst[5] == 6.0 + 102.0);
    CHECK(data.dst[6] == 7.0 + 200.0);
    CHECK(data.dst[7] == 8.0 + 201.0);
    CHECK(data.dst[8] == 9.0 + 202.0);
    CHECK(data.dst[9] == 0.0);
    CHECK(data.dst[10] == 0.0);
    CHECK(data.dst[11] == 0.0);
    // (b) retrieve_solution_row: dxs = -solution
    for(int i = 0; i < kRoundVertices * 3; ++i)
        CHECK(data.dxs[static_cast<std::size_t>(i)]
              == -data.solution[static_cast<std::size_t>(i)]);
    // (c) mass_block_norm_row: 0 for a fixed vertex, the mass otherwise
    CHECK(data.mass_norm[0] == 0.0);
    CHECK(data.mass_norm[1] == data.masses[1]);
    CHECK(data.mass_norm[2] == data.masses[2]);
    CHECK(data.mass_norm[3] == 0.0);
    // (d) diag_block_norm_row: the diagonal blocks (1,1) and (2,2) hold
    //     1 + 100 i + 10 r + c, so their max |diagonal| is 421 and 721; the two
    //     fixed vertices' diagonals are 0
    CHECK(data.diag_norm[0] == 0.0);
    CHECK(data.diag_norm[1] == 423.0);
    CHECK(data.diag_norm[2] == 723.0);
    CHECK(data.diag_norm[3] == 0.0);
    // (e) fix_hessian_row: a fixed diagonal block becomes Identity and a fixed
    //     off-diagonal one Zero -- and an untouched block keeps its value
    // the flat block layout is `t_values[i * 9 + c * 3 + r]` (column-major)
    CHECK(data.t_values[9 + 0] == 0.0);   // triplet 1 = (0,1): Zero (row 0 is fixed)
    CHECK(data.t_values[9 + 4] == 0.0);
    CHECK(data.t_values[9 + 8] == 0.0);
    CHECK(data.t_values[18 + 0] == 0.0);  // triplet 2 = (0,2): Zero
    CHECK(data.t_values[9 * 9] == 1.0);  // triplet 9 = (3,3): Identity
    CHECK(data.t_values[9 * 9 + 4] == 1.0);
    CHECK(data.t_values[9 * 9 + 8] == 1.0);
    CHECK(data.t_values[4 * 9] == 401.0);  // (1,1): untouched (no fixed vertex)
    CHECK(data.t_values[7 * 9] == 701.0);  // (2,2): untouched
    // (f) copy_dytopo_hessian_row(dst = A, src = dh, I = 0, vertex_offset = 0): the
    //     destination *position* 0 now carries the source's (row, col) and block,
    //     ignoring the fixed mask (the reference's documented behaviour)
    CHECK(data.t_rows[0] == 0);
    CHECK(data.t_cols[0] == 3);
    for(int c = 0; c < 3; ++c)
        for(int r = 0; r < 3; ++r)
            CHECK(data.t_values[c * 3 + r] == data.dh_values[static_cast<std::size_t>(c * 3 + r)]);

    // the device side: the same data, the same calls, one kernel dispatch
    RoundData           dev;
    lud::LDeviceBuffer<std::int32_t> d_is_fixed, d_didx, d_dyidx, d_trows, d_tcols, d_dhrows,
        d_dhcols;
    lud::LDeviceBuffer<double> d_masses, d_dvals, d_dyvals, d_tvals, d_dhvals, d_dst, d_sol,
        d_dxs, d_dnorm, d_mnorm, d_out;
    {
        d_is_fixed.resize(dev.is_fixed.size());
        d_is_fixed.view().copy_from(span<const std::int32_t>(dev.is_fixed));
        d_masses.resize(dev.masses.size());
        d_masses.view().copy_from(span<const double>(dev.masses));
        d_didx.resize(dev.d_indices.size());
        d_didx.view().copy_from(span<const std::int32_t>(dev.d_indices));
        d_dvals.resize(dev.d_values.size());
        d_dvals.view().copy_from(span<const double>(dev.d_values));
        d_dyidx.resize(dev.dy_indices.size());
        d_dyidx.view().copy_from(span<const std::int32_t>(dev.dy_indices));
        d_dyvals.resize(dev.dy_values.size());
        d_dyvals.view().copy_from(span<const double>(dev.dy_values));
        d_trows.resize(dev.t_rows.size());
        d_trows.view().copy_from(span<const std::int32_t>(dev.t_rows));
        d_tcols.resize(dev.t_cols.size());
        d_tcols.view().copy_from(span<const std::int32_t>(dev.t_cols));
        d_tvals.resize(dev.t_values.size());
        d_tvals.view().copy_from(span<const double>(dev.t_values));
        d_dhrows.resize(dev.dh_rows.size());
        d_dhrows.view().copy_from(span<const std::int32_t>(dev.dh_rows));
        d_dhcols.resize(dev.dh_cols.size());
        d_dhcols.view().copy_from(span<const std::int32_t>(dev.dh_cols));
        d_dhvals.resize(dev.dh_values.size());
        d_dhvals.view().copy_from(span<const double>(dev.dh_values));
        d_dst.resize(dev.dst.size());
        d_dst.view().copy_from(span<const double>(dev.dst));
        d_sol.resize(dev.solution.size());
        d_sol.view().copy_from(span<const double>(dev.solution));
        d_dxs.resize(dev.dxs.size());
        d_dxs.view().copy_from(span<const double>(dev.dxs));
        d_dnorm.resize(dev.diag_norm.size());
        d_dnorm.view().copy_from(span<const double>(dev.diag_norm));
        d_mnorm.resize(dev.mass_norm.size());
        d_mnorm.view().copy_from(span<const double>(dev.mass_norm));
        d_out.resize(host_out.size());
    }

    // NOTE: the host round mutated `data`; the device side runs on the *pristine*
    // `dev` copy, so both sides see the same inputs (the host's own dump in
    // `host_out` is the reference for the comparison).
    auto& k = round_kernel();
    gg::frame_stream()
        .record(k(d_is_fixed.view(),
                  d_masses.view(),
                  d_didx.view(),
                  d_dvals.view(),
                  d_dyidx.view(),
                  d_dyvals.view(),
                  d_trows.view(),
                  d_tcols.view(),
                  d_tvals.view(),
                  d_dhrows.view(),
                  d_dhcols.view(),
                  d_dhvals.view(),
                  d_dst.view(),
                  d_sol.view(),
                  d_dxs.view(),
                  d_dnorm.view(),
                  d_mnorm.view(),
                  d_out.view())
                     .dispatch(1));
    gg::frame_stream().synchronize();

    std::vector<double> dev_out(host_out.size(), 0.0);
    d_out.view().copy_to(span<double>(dev_out));
    lud::main_stream().synchronize();

    REQUIRE(dev_out.size() == host_out.size());
    SizeT mismatches = 0;
    SizeT non_zero   = 0;
    for(SizeT i = 0; i < host_out.size(); ++i)
    {
        if(dev_out[i] != host_out[i])
            ++mismatches;
        if(host_out[i] != 0.0)
            ++non_zero;
    }
    std::printf("[t54] device-vs-host round: %zu entries, %zu non-zero, %zu mismatches "
                "(bitwise)\n",
                host_out.size(),
                non_zero,
                mismatches);
    CHECK(non_zero > 80u);  // non-vacuous: the round actually produced values
    CHECK(mismatches == 0u);
}
