/********************************************************************
 * @file distance_system/distance_diagnoser.cpp
 * @brief S9' (T8.5-2) — the distance diagnoser (port of
 * src/backends/cuda/distance_system/distance_diagnoser.cu, 1035 reference lines).
 *
 * A diagnosis-only query service: `do_build()` inserts one
 * `core::DistanceDiagnoserFeature` and nothing else (no action, no `require<>`), so the
 * system cannot perturb a solve.  The four queries compute the closest-point
 * configuration of every pair and write the reference's attribute set on the result
 * geometry `R`.  See the header for the host-evaluation decision.
 *
 * WHAT IS AND IS NOT RE-DERIVED
 * -----------------------------
 * The math is *calls only*:
 *  * `ldist::*_distance_flag` / `*_distance2` / `*_distance2_gradient` /
 *    `*_distance2_hessian` — the T2.3 port of `utils/distance/distance_flagged.h`
 *    (bit-exact against the reference `.inl`, 10 000 configs);
 *  * `ldist::edge_edge_mollifier*` — the T2.3 port of `utils/distance/edge_edge_mollifier.h`;
 *  * `sym::codim_ipc_contact::KappaBarrier` / `dKappaBarrierdD` / `ddKappaBarrierddD` — the
 *    T8.1b port of the generated barrier law (`sym/codim_ipc_contact.inl`);
 *  * `PT_/EE_/PE_/PP_d_hat` + `*_thickness` + `D_range` + `is_active_D` — the T5.2/T7.2b
 *    ports of `utils/primitive_d_hat.h` / `utils/codim_thickness.h`.
 * The only arithmetic written out here is the reference's own composition of those
 * results, transcribed term for term and in the reference's association order
 * (`GradB = dBdD * GradD`, `HessB = ddBddD * GradD * GradD^T + dBdD * HessD`, and the
 * mollified `E = e_k * B` triple), because the association order is part of the
 * bit-exactness contract.  The port could have called
 * `sym::codim_ipc_simplex_contact::PT_barrier_gradient_hessian(kappa, ...)` instead — it
 * computes the same three lines — but that helper re-evaluates `D`, `GradD` and `HessD`
 * internally (the diagnoser has to report those anyway) and, for EE, does not return
 * `e_k`/`Gradek`/`Hessek`, which the EE query must also produce.
 *
 * CORRECTION TO THE PLAN (research §1 correction A)
 * ------------------------------------------------
 * The plan (21-distance-diagnoser §1.3, §2) claims six config keys (`dist2/{grad,hess}`,
 * `e_k/{grad,hess}`, `barrier/{grad,hess}`).  **There are none**: the reference has zero
 * `config.find<>` calls (the only `find<>`s are `sc.vertices().find` and
 * `R.instances().find`, .cu:294/308/772) and those six names are *attribute* names that
 * are always written (.cu:733-740, 855-869, 949-956, 1028-1033).  The port therefore
 * creates the whole attribute set unconditionally; the only "default" behaviour is the
 * reference's per-attribute create-time fill (all zero except `e_k` == 1, .cu:861) and
 * the per-entry values the kernel leaves at their local zero when a pair is inactive
 * (`is_active_D` false ⇒ barrier and its derivatives stay exactly 0, while `dist2`, its
 * gradient/Hessian, the flag, the coord and `e_k` are written for every pair).
 *
 * HOST CONTROL FLOW
 * -----------------
 * The reference branches on device values inside its kernels; here the same values are
 * host values, so `if`s are allowed again (RULE M2 constrains device data only) — the
 * flag/coord helpers and the per-pair bodies are therefore literal transcriptions of the
 * reference, with Eigen -> `lps::LVec`/`lps::LMat`, `T(0)`/`T(1)` -> `zero_of`/`one_of`,
 * and `cuda_tool::DeviceBuffer<T>` -> a host `uipc::vector<T>`.  The reference's
 * `UIPC_GENERIC` helpers become plain host templates (`T = double` is the only
 * instantiation) — the flag element type of the ported layer is `int` for a host `T`,
 * which is exactly the reference's `Vector4i`/`Vector3i`.
 *********************************************************************/
#include <distance_system/distance_diagnoser.h>
#include <luisa_device/math/distance/flagged.h>
#include <luisa_device/math/distance/mollifier.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <utils/codim_thickness.h>
#include <utils/primitive_d_hat.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/log.h>
#include <uipc/common/span.h>
#include <uipc/common/vector.h>
#include <algorithm>
#include <cstring>
#include <string_view>

namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(DistanceDiagnoser);

// =====================================================================
// Overrider
// =====================================================================
DistanceDiagnoserFeatureOverrider::DistanceDiagnoserFeatureOverrider(
    DistanceDiagnoser* diagnoser)
{
    UIPC_ASSERT(diagnoser, "DistanceDiagnoser must not be null");
    m_diagnoser = *diagnoser;
}

void DistanceDiagnoserFeatureOverrider::do_compute_point_triangle_distance(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points,
    const geometry::SimplicialComplex& triangles)
{
    m_diagnoser->compute_point_triangle_distance(R, points, triangles);
}

void DistanceDiagnoserFeatureOverrider::do_compute_edge_edge_distance(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& edges_a,
    const geometry::SimplicialComplex& edges_b,
    const geometry::SimplicialComplex& rest_edges_a,
    const geometry::SimplicialComplex& rest_edges_b)
{
    m_diagnoser->compute_edge_edge_distance(
        R, edges_a, edges_b, rest_edges_a, rest_edges_b);
}

void DistanceDiagnoserFeatureOverrider::do_compute_point_edge_distance(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points,
    const geometry::SimplicialComplex& edges)
{
    m_diagnoser->compute_point_edge_distance(R, points, edges);
}

void DistanceDiagnoserFeatureOverrider::do_compute_point_point_distance(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points_a,
    const geometry::SimplicialComplex& points_b)
{
    m_diagnoser->compute_point_point_distance(R, points_a, points_b);
}

// =====================================================================
// SimSystem
// =====================================================================
void DistanceDiagnoser::do_build()
{
    // The reference's six-line do_build (.cu:61-66), verbatim: one overrider, one
    // feature, no action and no `require<>`, so the system is inert in the solve chain.
    auto overrider = std::make_shared<DistanceDiagnoserFeatureOverrider>(this);
    features().insert(
        std::make_shared<core::DistanceDiagnoserFeature>(overrider));
}

namespace
{
namespace ldist = luisa_device::math::distance;

// The reference's `namespace detail` helpers, TU-local here: `detail` is already a
// namespace of this backend (`affine_body`, `collision_detection` and `finite_element` use
// `uipc::backend::luisa::detail`), so a second one of the same name would make every
// unqualified `detail::` an ambiguity.  The function names are the reference's.

/// `Eigen::Vector<T,N>` <- the numerics layer's `LVec<T,N>` (identical storage, explicit
/// element-wise conversion — the reference's `Eigen::Vector<T,N>` becomes `LVec` inside
/// the math calls, and the attributes keep the Eigen type downstream code expects).
template <typename T, int N>
[[nodiscard]] Vector<T, N> to_eigen_vec(const lps::LVec<T, N>& v) noexcept
{
    Vector<T, N> out;
    for(int i = 0; i < N; ++i)
        out(i) = v(i);
    return out;
}

/// `Eigen::Matrix<T,N,N>` <- `LMat<T,N,N>` (both column-major).
template <typename T, int N>
[[nodiscard]] Matrix<T, N, N> to_eigen_mat(const lps::LMat<T, N, N>& m) noexcept
{
    Matrix<T, N, N> out;
    for(int c = 0; c < N; ++c)
        for(int r = 0; r < N; ++r)
            out(r, c) = m(r, c);
    return out;
}

/// The reference's `upload_vertex_attribute` (.cu:289-299) without the device copy: the
/// attribute is a hard precondition of every query, so a missing one is an assert, never
/// a silent zero fill (a zeroed `d_hat` would make every pair inactive and every
/// barrier attribute look "correctly" empty).
template <typename T>
[[nodiscard]] span<const T> vertex_attribute(
    const geometry::SimplicialComplex& sc,
    std::string_view                   name)
{
    auto attr = sc.vertices().find<T>(std::string{name});
    UIPC_ASSERT(attr,
                "Vertex attribute '{}' not found on SimplicialComplex",
                name);
    return attr->view();
}

/// The reference's `copy_back` (.cu:301-313): find-or-create the attribute on the result
/// instances, then bulk-copy the whole value buffer into it.
///
/// The template shape is the reference's: `T` is deduced from the *buffer* and the
/// default value is a free type, because the call sites pass Eigen nullary expressions
/// (`Vector12::Zero()`); `T(default_val)` is the reference's create-time fill (all zero
/// except `e_k`'s 1).  The bulk copy is a byte copy — on the device side the reference
/// does exactly `buf.copy_to(view(*attr).data())` and both sides here are the same
/// trivially-copyable element type.  The size assertion replaces the reference's
/// unchecked `copy_to` (it can only fire if `R.instances().resize(num_pairs)` and the
/// buffer length disagree, i.e. never — the reference relies on that too).
template <typename T, typename D>
void copy_back(geometry::Geometry&     R,
               std::string_view       name,
               const D&               default_val,
               const uipc::vector<T>& buf)
{
    std::string n{name};
    auto        attr = R.instances().find<T>(n);
    if(!attr)
        attr = R.instances().create<T>(n, T(default_val));
    auto dst = view(*attr);
    UIPC_ASSERT(dst.size() == buf.size(),
                "DistanceDiagnoser: '{}' has {} entries, expected {}",
                name,
                dst.size(),
                buf.size());
    std::memcpy(dst.data(), buf.data(), buf.size() * sizeof(T));
}

// -----------------------------------------------------------------------
//  Closest-point coordinates (the reference's "which feature is closest"
//  bookkeeping).  Branchless candidate selection is a *device* concern; every
//  value here is a host value, so these are the reference's own `if` chains.
// -----------------------------------------------------------------------
template <typename T>
void point_triangle_closest_point_coord(const lps::LVec<int, 4>& flag,
                                        const lps::LVec<T, 3>&   p,
                                        const lps::LVec<T, 3>&   t0,
                                        const lps::LVec<T, 3>&   t1,
                                        const lps::LVec<T, 3>&   t2,
                                        lps::LVec<T, 4>&         coord)
{
    coord.setZero();
    IndexT dim = ldist::detail::active_count(flag);

    if(dim == 2)
    {
        if(flag[1])
            coord[1] = T(1);
        else if(flag[2])
            coord[2] = T(1);
        else if(flag[3])
            coord[3] = T(1);
    }
    else if(dim == 3)
    {
        lps::LVec<T, 3> e0, e1;
        IndexT          idx0 = -1;
        IndexT          idx1 = -1;
        if(flag[1] && flag[2])
        {
            e0   = t0;
            e1   = t1;
            idx0 = 1;
            idx1 = 2;
        }
        else if(flag[2] && flag[3])
        {
            e0   = t1;
            e1   = t2;
            idx0 = 2;
            idx1 = 3;
        }
        else if(flag[3] && flag[1])
        {
            e0   = t2;
            e1   = t0;
            idx0 = 3;
            idx1 = 1;
        }

        lps::LVec<T, 3> edge    = e1 - e0;
        T               len2    = edge.squaredNorm();
        T               t_param = T(0);
        if(len2 > T(0))
            t_param = (p - e0).dot(edge) / len2;
        t_param = lps::max(T(0), lps::min(T(1), t_param));

        coord[idx0] = T(1) - t_param;
        coord[idx1] = t_param;
    }
    else if(dim == 4)
    {
        lps::LVec<T, 3> v0 = t1 - t0;
        lps::LVec<T, 3> v1 = t2 - t0;
        lps::LVec<T, 3> v2 = p - t0;

        T d00 = v0.dot(v0);
        T d01 = v0.dot(v1);
        T d11 = v1.dot(v1);
        T d20 = v2.dot(v0);
        T d21 = v2.dot(v1);

        T denom = d00 * d11 - d01 * d01;
        T v     = (d11 * d20 - d01 * d21) / denom;
        T w     = (d00 * d21 - d01 * d20) / denom;
        T u     = T(1) - v - w;

        coord[1] = u;
        coord[2] = v;
        coord[3] = w;
    }
}

template <typename T>
void edge_edge_closest_point_coord(const lps::LVec<int, 4>& flag,
                                   const lps::LVec<T, 3>&   ea0,
                                   const lps::LVec<T, 3>&   ea1,
                                   const lps::LVec<T, 3>&   eb0,
                                   const lps::LVec<T, 3>&   eb1,
                                   lps::LVec<T, 4>&         coord)
{
    coord.setZero();
    IndexT dim = ldist::detail::active_count(flag);

    if(dim == 2)
    {
        if(flag[0])
            coord[0] = T(1);
        else
            coord[1] = T(1);

        if(flag[2])
            coord[2] = T(1);
        else
            coord[3] = T(1);
    }
    else if(dim == 3)
    {
        lps::LVec<T, 3> point, e0, e1;
        IndexT          pt_idx = -1;
        IndexT          e0_idx = -1;
        IndexT          e1_idx = -1;

        if(!flag[0])
        {
            point  = ea1;
            pt_idx = 1;
            e0     = eb0;
            e0_idx = 2;
            e1     = eb1;
            e1_idx = 3;
        }
        else if(!flag[1])
        {
            point  = ea0;
            pt_idx = 0;
            e0     = eb0;
            e0_idx = 2;
            e1     = eb1;
            e1_idx = 3;
        }
        else if(!flag[2])
        {
            point  = eb1;
            pt_idx = 3;
            e0     = ea0;
            e0_idx = 0;
            e1     = ea1;
            e1_idx = 1;
        }
        else
        {
            point  = eb0;
            pt_idx = 2;
            e0     = ea0;
            e0_idx = 0;
            e1     = ea1;
            e1_idx = 1;
        }

        coord[pt_idx] = T(1);

        lps::LVec<T, 3> edge    = e1 - e0;
        T               len2    = edge.squaredNorm();
        T               t_param = T(0);
        if(len2 > T(0))
            t_param = (point - e0).dot(edge) / len2;
        t_param = lps::max(T(0), lps::min(T(1), t_param));

        coord[e0_idx] = T(1) - t_param;
        coord[e1_idx] = t_param;
    }
    else if(dim == 4)
    {
        lps::LVec<T, 3> u = ea1 - ea0;
        lps::LVec<T, 3> v = eb1 - eb0;
        lps::LVec<T, 3> w = ea0 - eb0;

        T a = u.dot(u);
        T b = u.dot(v);
        T c = v.dot(v);
        T d = u.dot(w);
        T e = v.dot(w);
        T D = a * c - b * b;

        T s = T(0), t = T(0);
        if(D > T(1e-30))
        {
            s = (b * e - c * d) / D;
            t = (a * e - b * d) / D;
        }
        s = lps::max(T(0), lps::min(T(1), s));
        t = lps::max(T(0), lps::min(T(1), t));

        coord[0] = T(1) - s;
        coord[1] = s;
        coord[2] = T(1) - t;
        coord[3] = t;
    }
}

template <typename T>
void point_edge_closest_point_coord(const lps::LVec<int, 3>& flag,
                                    const lps::LVec<T, 3>&   p,
                                    const lps::LVec<T, 3>&   e0,
                                    const lps::LVec<T, 3>&   e1,
                                    lps::LVec<T, 3>&         coord)
{
    coord.setZero();
    IndexT dim = ldist::detail::active_count(flag);

    if(dim == 2)
    {
        if(flag[1])
            coord[1] = T(1);
        else
            coord[2] = T(1);
    }
    else if(dim == 3)
    {
        lps::LVec<T, 3> edge    = e1 - e0;
        T               len2    = edge.squaredNorm();
        T               t_param = T(0);
        if(len2 > T(0))
            t_param = (p - e0).dot(edge) / len2;
        t_param = lps::max(T(0), lps::min(T(1), t_param));

        coord[1] = T(1) - t_param;
        coord[2] = t_param;
    }
}
}  // namespace

// =====================================================================
// Point-Triangle Distance (+ Barrier)
// =====================================================================
void DistanceDiagnoser::compute_point_triangle_distance(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points,
    const geometry::SimplicialComplex& triangles)
{
    auto h_point_pos = points.positions().view();
    auto h_tri_pos   = triangles.positions().view();
    auto h_tri_topo  = triangles.triangles().topo().view();

    SizeT num_points    = h_point_pos.size();
    SizeT num_triangles = h_tri_topo.size();
    SizeT num_pairs     = num_points * num_triangles;

    if(num_pairs == 0)
    {
        R.instances().resize(0);
        return;
    }

    auto h_dhat_p      = vertex_attribute<Float>(points, "d_hat");
    auto h_dhat_t      = vertex_attribute<Float>(triangles, "d_hat");
    auto h_thickness_p = vertex_attribute<Float>(points, "thickness");
    auto h_thickness_t = vertex_attribute<Float>(triangles,
                                                         "thickness");

    // the reference's eight `DeviceBuffer<T>` outputs, as host buffers
    uipc::vector<Float>       dist2(num_pairs);
    uipc::vector<Vector12>    dist2_grad(num_pairs);
    uipc::vector<Matrix12x12> dist2_hess(num_pairs);
    uipc::vector<Vector4i>    flags(num_pairs);
    uipc::vector<Vector4>     coords(num_pairs);
    uipc::vector<Float>       barrier(num_pairs);
    uipc::vector<Vector12>    barrier_grad(num_pairs);
    uipc::vector<Matrix12x12> barrier_hess(num_pairs);

    SizeT M = num_triangles;
    // the reference's thread `i = blockIdx.x * blockDim.x + threadIdx.x`, in order
    for(SizeT i = 0; i < num_pairs; ++i)
    {
        IndexT pi = static_cast<IndexT>(i / M);
        IndexT ti = static_cast<IndexT>(i % M);

        const lps::LVec<Float, 3> P  = to_luisa(h_point_pos[pi]);
        Vector3i                  TI = h_tri_topo[ti];
        const lps::LVec<Float, 3> T0 = to_luisa(h_tri_pos[TI[0]]);
        const lps::LVec<Float, 3> T1 = to_luisa(h_tri_pos[TI[1]]);
        const lps::LVec<Float, 3> T2 = to_luisa(h_tri_pos[TI[2]]);

        auto flag = ldist::point_triangle_distance_flag(P, T0, T1, T2);

        Float D;
        ldist::point_triangle_distance2(flag, P, T0, T1, T2, D);

        lps::LVec<Float, 12> GradD;
        ldist::point_triangle_distance2_gradient(flag, P, T0, T1, T2, GradD);

        lps::LMat<Float, 12, 12> HessD;
        ldist::point_triangle_distance2_hessian(flag, P, T0, T1, T2, HessD);

        lps::LVec<Float, 4> coord;
        point_triangle_closest_point_coord(flag, P, T0, T1, T2, coord);

        Float d_hat_pair = PT_d_hat(h_dhat_p[pi],
                                    h_dhat_t[TI[0]],
                                    h_dhat_t[TI[1]],
                                    h_dhat_t[TI[2]]);
        Float xi         = PT_thickness(h_thickness_p[pi],
                                h_thickness_t[TI[0]],
                                h_thickness_t[TI[1]],
                                h_thickness_t[TI[2]]);
        auto dr = D_range(xi, d_hat_pair);

        Float                    B = Float(0);
        lps::LVec<Float, 12>     GradB;
        lps::LMat<Float, 12, 12> HessB;
        GradB.setZero();
        HessB.setZero();

        if(is_active_D(dr, D))
        {
            Float kappa = Float(1);
            sym::codim_ipc_contact::KappaBarrier(B, kappa, D, d_hat_pair, xi);
            Float dBdD;
            sym::codim_ipc_contact::dKappaBarrierdD(
                dBdD, kappa, D, d_hat_pair, xi);
            Float ddBddD;
            sym::codim_ipc_contact::ddKappaBarrierddD(
                ddBddD, kappa, D, d_hat_pair, xi);
            GradB = dBdD * GradD;
            HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
        }

        dist2[i]        = D;
        dist2_grad[i]   = to_eigen_vec(GradD);
        dist2_hess[i]   = to_eigen_mat(HessD);
        flags[i]        = to_eigen_vec(flag);
        coords[i]       = to_eigen_vec(coord);
        barrier[i]      = B;
        barrier_grad[i] = to_eigen_vec(GradB);
        barrier_hess[i] = to_eigen_mat(HessB);
    }

    R.instances().resize(num_pairs);

    copy_back(R, "dist2", Float(0), dist2);
    copy_back(R, "dist2/grad", Vector12::Zero(), dist2_grad);
    copy_back(R, "dist2/hess", Matrix12x12::Zero(), dist2_hess);
    copy_back(R, "flag", Vector4i::Zero(), flags);
    copy_back(R, "coord", Vector4::Zero(), coords);
    copy_back(R, "barrier", Float(0), barrier);
    copy_back(R, "barrier/grad", Vector12::Zero(), barrier_grad);
    copy_back(R, "barrier/hess", Matrix12x12::Zero(), barrier_hess);
}

// =====================================================================
// Edge-Edge Distance (with Mollifier + Barrier)
// =====================================================================
void DistanceDiagnoser::compute_edge_edge_distance(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& edges_a,
    const geometry::SimplicialComplex& edges_b,
    const geometry::SimplicialComplex& rest_edges_a,
    const geometry::SimplicialComplex& rest_edges_b)
{
    auto h_pos_a      = edges_a.positions().view();
    auto h_pos_b      = edges_b.positions().view();
    auto h_topo_a     = edges_a.edges().topo().view();
    auto h_topo_b     = edges_b.edges().topo().view();
    auto h_rest_pos_a = rest_edges_a.positions().view();
    auto h_rest_pos_b = rest_edges_b.positions().view();

    SizeT num_ea    = h_topo_a.size();
    SizeT num_eb    = h_topo_b.size();
    SizeT num_pairs = num_ea * num_eb;

    if(num_pairs == 0)
    {
        R.instances().resize(0);
        return;
    }

    // read *before* the resize, (re)created *after* it (.cu:772-779, 868-869): `create`
    // sizes itself off the collection's current size, so the order is part of the contract
    auto  mc_attr             = R.instances().find<Float>("mollifier_coeff");
    Float mollifier_coeff_val = Float(1e-3);
    if(mc_attr)
    {
        auto mc_view = mc_attr->view();
        if(mc_view.size() > 0)
            mollifier_coeff_val = mc_view[0];
    }

    auto h_dhat_a      = vertex_attribute<Float>(edges_a, "d_hat");
    auto h_dhat_b      = vertex_attribute<Float>(edges_b, "d_hat");
    auto h_thickness_a = vertex_attribute<Float>(edges_a, "thickness");
    auto h_thickness_b = vertex_attribute<Float>(edges_b, "thickness");

    uipc::vector<Float>       dist2(num_pairs);
    uipc::vector<Vector12>    dist2_grad(num_pairs);
    uipc::vector<Matrix12x12> dist2_hess(num_pairs);
    uipc::vector<Vector4i>    flags(num_pairs);
    uipc::vector<Vector4>     coords(num_pairs);
    uipc::vector<Float>       eps_xs(num_pairs);
    uipc::vector<Float>       e_ks(num_pairs);
    uipc::vector<Vector12>    ek_grad(num_pairs);
    uipc::vector<Matrix12x12> ek_hess(num_pairs);
    uipc::vector<Float>       barrier(num_pairs);
    uipc::vector<Vector12>    barrier_grad(num_pairs);
    uipc::vector<Matrix12x12> barrier_hess(num_pairs);

    SizeT M     = num_eb;
    Float coeff = mollifier_coeff_val;

    for(SizeT i = 0; i < num_pairs; ++i)
    {
        IndexT ai = static_cast<IndexT>(i / M);
        IndexT bi = static_cast<IndexT>(i % M);

        Vector2i EA = h_topo_a[ai];
        Vector2i EB = h_topo_b[bi];

        const lps::LVec<Float, 3> ea0 = to_luisa(h_pos_a[EA[0]]);
        const lps::LVec<Float, 3> ea1 = to_luisa(h_pos_a[EA[1]]);
        const lps::LVec<Float, 3> eb0 = to_luisa(h_pos_b[EB[0]]);
        const lps::LVec<Float, 3> eb1 = to_luisa(h_pos_b[EB[1]]);

        const lps::LVec<Float, 3> ea0_rest = to_luisa(h_rest_pos_a[EA[0]]);
        const lps::LVec<Float, 3> ea1_rest = to_luisa(h_rest_pos_a[EA[1]]);
        const lps::LVec<Float, 3> eb0_rest = to_luisa(h_rest_pos_b[EB[0]]);
        const lps::LVec<Float, 3> eb1_rest = to_luisa(h_rest_pos_b[EB[1]]);

        auto flag = ldist::edge_edge_distance_flag(ea0, ea1, eb0, eb1);

        Float D;
        ldist::edge_edge_distance2(flag, ea0, ea1, eb0, eb1, D);

        lps::LVec<Float, 12> GradD;
        ldist::edge_edge_distance2_gradient(flag, ea0, ea1, eb0, eb1, GradD);

        lps::LMat<Float, 12, 12> HessD;
        ldist::edge_edge_distance2_hessian(flag, ea0, ea1, eb0, eb1, HessD);

        Float eps_x;
        ldist::edge_edge_mollifier_threshold(
            ea0_rest, ea1_rest, eb0_rest, eb1_rest, coeff, eps_x);

        Float ek;
        ldist::edge_edge_mollifier(ea0, ea1, eb0, eb1, eps_x, ek);

        lps::LVec<Float, 12> Gradek;
        ldist::edge_edge_mollifier_gradient(ea0, ea1, eb0, eb1, eps_x, Gradek);

        lps::LMat<Float, 12, 12> Hessek;
        ldist::edge_edge_mollifier_hessian(ea0, ea1, eb0, eb1, eps_x, Hessek);

        IndexT dim = ldist::detail::active_count(flag);
        if(dim < 4)
        {
            Gradek.setZero();
            Hessek.setZero();
        }

        lps::LVec<Float, 4> coord;
        edge_edge_closest_point_coord(flag, ea0, ea1, eb0, eb1, coord);

        // Barrier: E = e_k * B(D)
        Float d_hat_pair = EE_d_hat(h_dhat_a[EA[0]],
                                    h_dhat_a[EA[1]],
                                    h_dhat_b[EB[0]],
                                    h_dhat_b[EB[1]]);
        Float xi         = EE_thickness(h_thickness_a[EA[0]],
                                h_thickness_a[EA[1]],
                                h_thickness_b[EB[0]],
                                h_thickness_b[EB[1]]);
        auto  dr         = D_range(xi, d_hat_pair);

        Float                    E = Float(0);
        lps::LVec<Float, 12>     GradE;
        lps::LMat<Float, 12, 12> HessE;
        GradE.setZero();
        HessE.setZero();

        if(is_active_D(dr, D))
        {
            Float kappa = Float(1);
            Float B;
            sym::codim_ipc_contact::KappaBarrier(B, kappa, D, d_hat_pair, xi);
            Float dBdD;
            sym::codim_ipc_contact::dKappaBarrierdD(
                dBdD, kappa, D, d_hat_pair, xi);
            Float ddBddD;
            sym::codim_ipc_contact::ddKappaBarrierddD(
                ddBddD, kappa, D, d_hat_pair, xi);

            // the reference's three additive terms, with the two repeated factors
            // hoisted so that every line fits the 80-column limit; the factors and the
            // left-to-right association of the sum are the reference's, so the value is
            // bit-identical (.cu:499-501)
            lps::LMat<Float, 12, 12> hess_D =
                ddBddD * GradD * GradD.transpose() + dBdD * HessD;
            lps::LMat<Float, 12, 12> cross_ek =
                GradD * Gradek.transpose() + Gradek * GradD.transpose();

            E     = ek * B;
            GradE = ek * dBdD * GradD + B * Gradek;
            HessE = ek * hess_D + dBdD * cross_ek + B * Hessek;
        }

        dist2[i]        = D;
        dist2_grad[i]   = to_eigen_vec(GradD);
        dist2_hess[i]   = to_eigen_mat(HessD);
        flags[i]        = to_eigen_vec(flag);
        coords[i]       = to_eigen_vec(coord);
        eps_xs[i]       = eps_x;
        e_ks[i]         = ek;
        ek_grad[i]      = to_eigen_vec(Gradek);
        ek_hess[i]      = to_eigen_mat(Hessek);
        barrier[i]      = E;
        barrier_grad[i] = to_eigen_vec(GradE);
        barrier_hess[i] = to_eigen_mat(HessE);
    }

    R.instances().resize(num_pairs);

    copy_back(R, "dist2", Float(0), dist2);
    copy_back(R, "dist2/grad", Vector12::Zero(), dist2_grad);
    copy_back(R, "dist2/hess", Matrix12x12::Zero(), dist2_hess);
    copy_back(R, "flag", Vector4i::Zero(), flags);
    copy_back(R, "coord", Vector4::Zero(), coords);
    copy_back(R, "eps_x", Float(0), eps_xs);
    copy_back(R, "e_k", Float(1), e_ks);
    copy_back(R, "e_k/grad", Vector12::Zero(), ek_grad);
    copy_back(R, "e_k/hess", Matrix12x12::Zero(), ek_hess);
    copy_back(R, "barrier", Float(0), barrier);
    copy_back(R, "barrier/grad", Vector12::Zero(), barrier_grad);
    copy_back(R, "barrier/hess", Matrix12x12::Zero(), barrier_hess);

    if(!mc_attr)
        R.instances().create<Float>("mollifier_coeff", mollifier_coeff_val);
}

// =====================================================================
// Point-Edge Distance (+ Barrier)
// =====================================================================
void DistanceDiagnoser::compute_point_edge_distance(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points,
    const geometry::SimplicialComplex& edges)
{
    auto h_point_pos = points.positions().view();
    auto h_edge_pos  = edges.positions().view();
    auto h_edge_topo = edges.edges().topo().view();

    SizeT num_points = h_point_pos.size();
    SizeT num_edges  = h_edge_topo.size();
    SizeT num_pairs  = num_points * num_edges;

    if(num_pairs == 0)
    {
        R.instances().resize(0);
        return;
    }

    auto h_dhat_p      = vertex_attribute<Float>(points, "d_hat");
    auto h_dhat_e      = vertex_attribute<Float>(edges, "d_hat");
    auto h_thickness_p = vertex_attribute<Float>(points, "thickness");
    auto h_thickness_e = vertex_attribute<Float>(edges, "thickness");

    uipc::vector<Float>     dist2(num_pairs);
    uipc::vector<Vector9>   dist2_grad(num_pairs);
    uipc::vector<Matrix9x9> dist2_hess(num_pairs);
    uipc::vector<Vector3i>  flags(num_pairs);
    uipc::vector<Vector3>   coords(num_pairs);
    uipc::vector<Float>     barrier(num_pairs);
    uipc::vector<Vector9>   barrier_grad(num_pairs);
    uipc::vector<Matrix9x9> barrier_hess(num_pairs);

    SizeT M = num_edges;
    for(SizeT i = 0; i < num_pairs; ++i)
    {
        IndexT pi = static_cast<IndexT>(i / M);
        IndexT ei = static_cast<IndexT>(i % M);

        const lps::LVec<Float, 3> P  = to_luisa(h_point_pos[pi]);
        Vector2i                  EI = h_edge_topo[ei];
        const lps::LVec<Float, 3> E0 = to_luisa(h_edge_pos[EI[0]]);
        const lps::LVec<Float, 3> E1 = to_luisa(h_edge_pos[EI[1]]);

        auto flag = ldist::point_edge_distance_flag(P, E0, E1);

        Float D;
        ldist::point_edge_distance2(flag, P, E0, E1, D);

        lps::LVec<Float, 9> GradD;
        ldist::point_edge_distance2_gradient(flag, P, E0, E1, GradD);

        lps::LMat<Float, 9, 9> HessD;
        ldist::point_edge_distance2_hessian(flag, P, E0, E1, HessD);

        lps::LVec<Float, 3> coord;
        point_edge_closest_point_coord(flag, P, E0, E1, coord);

        Float d_hat_pair = PE_d_hat(h_dhat_p[pi],
                                    h_dhat_e[EI[0]],
                                    h_dhat_e[EI[1]]);
        Float xi         = PE_thickness(h_thickness_p[pi],
                                h_thickness_e[EI[0]],
                                h_thickness_e[EI[1]]);
        auto dr = D_range(xi, d_hat_pair);

        Float                   B = Float(0);
        lps::LVec<Float, 9>     GradB;
        lps::LMat<Float, 9, 9>  HessB;
        GradB.setZero();
        HessB.setZero();

        if(is_active_D(dr, D))
        {
            Float kappa = Float(1);
            sym::codim_ipc_contact::KappaBarrier(B, kappa, D, d_hat_pair, xi);
            Float dBdD;
            sym::codim_ipc_contact::dKappaBarrierdD(
                dBdD, kappa, D, d_hat_pair, xi);
            Float ddBddD;
            sym::codim_ipc_contact::ddKappaBarrierddD(
                ddBddD, kappa, D, d_hat_pair, xi);
            GradB = dBdD * GradD;
            HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
        }

        dist2[i]        = D;
        dist2_grad[i]   = to_eigen_vec(GradD);
        dist2_hess[i]   = to_eigen_mat(HessD);
        flags[i]        = to_eigen_vec(flag);
        coords[i]       = to_eigen_vec(coord);
        barrier[i]      = B;
        barrier_grad[i] = to_eigen_vec(GradB);
        barrier_hess[i] = to_eigen_mat(HessB);
    }

    R.instances().resize(num_pairs);

    copy_back(R, "dist2", Float(0), dist2);
    copy_back(R, "dist2/grad", Vector9::Zero(), dist2_grad);
    copy_back(R, "dist2/hess", Matrix9x9::Zero(), dist2_hess);
    copy_back(R, "flag", Vector3i::Zero(), flags);
    copy_back(R, "coord", Vector3::Zero(), coords);
    copy_back(R, "barrier", Float(0), barrier);
    copy_back(R, "barrier/grad", Vector9::Zero(), barrier_grad);
    copy_back(R, "barrier/hess", Matrix9x9::Zero(), barrier_hess);
}

// =====================================================================
// Point-Point Distance (+ Barrier)
// =====================================================================
void DistanceDiagnoser::compute_point_point_distance(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points_a,
    const geometry::SimplicialComplex& points_b)
{
    auto h_pos_a = points_a.positions().view();
    auto h_pos_b = points_b.positions().view();

    SizeT num_a     = h_pos_a.size();
    SizeT num_b     = h_pos_b.size();
    SizeT num_pairs = num_a * num_b;

    if(num_pairs == 0)
    {
        R.instances().resize(0);
        return;
    }

    auto h_dhat_a      = vertex_attribute<Float>(points_a, "d_hat");
    auto h_dhat_b      = vertex_attribute<Float>(points_b, "d_hat");
    auto h_thickness_a = vertex_attribute<Float>(points_a, "thickness");
    auto h_thickness_b = vertex_attribute<Float>(points_b, "thickness");

    uipc::vector<Float>     dist2(num_pairs);
    uipc::vector<Vector6>   dist2_grad(num_pairs);
    uipc::vector<Matrix6x6> dist2_hess(num_pairs);
    uipc::vector<Float>     barrier(num_pairs);
    uipc::vector<Vector6>   barrier_grad(num_pairs);
    uipc::vector<Matrix6x6> barrier_hess(num_pairs);

    SizeT M = num_b;
    for(SizeT i = 0; i < num_pairs; ++i)
    {
        IndexT ai = static_cast<IndexT>(i / M);
        IndexT bi = static_cast<IndexT>(i % M);

        const lps::LVec<Float, 3> A = to_luisa(h_pos_a[ai]);
        const lps::LVec<Float, 3> B = to_luisa(h_pos_b[bi]);

        Float D;
        ldist::point_point_distance2(A, B, D);

        lps::LVec<Float, 6> GradD;
        ldist::point_point_distance2_gradient(A, B, GradD);

        lps::LMat<Float, 6, 6> HessD;
        ldist::point_point_distance2_hessian(A, B, HessD);

        Float d_hat_pair = PP_d_hat(h_dhat_a[ai], h_dhat_b[bi]);
        Float xi         = PP_thickness(h_thickness_a[ai], h_thickness_b[bi]);
        auto  dr         = D_range(xi, d_hat_pair);

        Float                   Bv = Float(0);
        lps::LVec<Float, 6>     GradB;
        lps::LMat<Float, 6, 6>  HessB;
        GradB.setZero();
        HessB.setZero();

        if(is_active_D(dr, D))
        {
            Float kappa = Float(1);
            sym::codim_ipc_contact::KappaBarrier(Bv, kappa, D, d_hat_pair, xi);
            Float dBdD;
            sym::codim_ipc_contact::dKappaBarrierdD(
                dBdD, kappa, D, d_hat_pair, xi);
            Float ddBddD;
            sym::codim_ipc_contact::ddKappaBarrierddD(
                ddBddD, kappa, D, d_hat_pair, xi);
            GradB = dBdD * GradD;
            HessB = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
        }

        dist2[i]        = D;
        dist2_grad[i]   = to_eigen_vec(GradD);
        dist2_hess[i]   = to_eigen_mat(HessD);
        barrier[i]      = Bv;
        barrier_grad[i] = to_eigen_vec(GradB);
        barrier_hess[i] = to_eigen_mat(HessB);
    }

    R.instances().resize(num_pairs);

    copy_back(R, "dist2", Float(0), dist2);
    copy_back(R, "dist2/grad", Vector6::Zero(), dist2_grad);
    copy_back(R, "dist2/hess", Matrix6x6::Zero(), dist2_hess);
    copy_back(R, "barrier", Float(0), barrier);
    copy_back(R, "barrier/grad", Vector6::Zero(), barrier_grad);
    copy_back(R, "barrier/hess", Matrix6x6::Zero(), barrier_hess);
}
}  // namespace uipc::backend::luisa
