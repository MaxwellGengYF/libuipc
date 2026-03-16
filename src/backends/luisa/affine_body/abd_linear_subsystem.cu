#include <affine_body/abd_linear_subsystem.h>
#include <sim_engine.h>
#include <kernel_cout.h>
#include <luisa/dsl/dsl.h>
#include <uipc/builtin/attribute_name.h>
#include <affine_body/inter_affine_body_constitution_manager.h>
#include <affine_body/abd_linear_subsystem_reporter.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/affine_body_constitution.h>
#include <utils/report_extent_check.h>

namespace uipc::backend::luisa
{
UIPC_GENERIC void zero_out_lower(Matrix12x12& H)
{
    // clear lower triangle (3x3 block based)
    for(IndexT jj = 0; jj < 4; ++jj)
    {
        for(IndexT ii = jj + 1; ii < 4; ++ii)
        {
            // H.block<3, 3>(ii * 3, jj * 3).setZero();
            for(int r = 0; r < 3; ++r)
                for(int c = 0; c < 3; ++c)
                    H(ii * 3 + r, jj * 3 + c) = 0.0f;
        }
    }
}
}  // namespace uipc::backend::luisa


namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(ABDLinearSubsystem);

// ref: https://github.com/spiriMirror/libuipc/issues/271
constexpr U64 ABDLinearSubsystemUID = 0ull;

void ABDLinearSubsystem::do_build(DiagLinearSubsystem::BuildInfo& info)
{
    m_impl.affine_body_dynamics        = require<AffineBodyDynamics>();
    m_impl.affine_body_vertex_reporter = require<AffineBodyVertexReporter>();
    auto attr = world().scene().config().find<Float>("dt");
    m_impl.dt = attr->view()[0];

    m_impl.dytopo_effect_receiver = find<ABDDyTopoEffectReceiver>();
}

void ABDLinearSubsystem::Impl::init()
{
    auto reporter_view = reporters.view();
    for(auto&& [i, r] : enumerate(reporter_view))
        r->m_index = i;
    for(auto& r : reporter_view)
        r->init();

    reporter_gradient_offsets_counts.resize(device(), reporter_view.size());
    reporter_hessian_offsets_counts.resize(device(), reporter_view.size());

    SizeT body_count = abd().body_count();
    body_id_to_shape_hessian = device().create_buffer<Matrix12x12>(body_count);
    body_id_to_shape_gradient = device().create_buffer<Vector12>(body_count);
    body_id_to_kinetic_hessian = device().create_buffer<Matrix12x12>(body_count);
    body_id_to_kinetic_gradient = device().create_buffer<Vector12>(body_count);
    diag_hessian = device().create_buffer<Matrix12x12>(body_count);
}

void ABDLinearSubsystem::Impl::report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    info.extent(abd().body_count() * 12);
}

void ABDLinearSubsystem::Impl::receive_init_dof_info(WorldVisitor& w,
                                                     GlobalLinearSystem::InitDofInfo& info)
{
    auto& geo_infos = abd().geo_infos;
    auto  geo_slots = w.scene().geometries();

    IndexT offset = info.dof_offset();

    // fill the dof_offset and dof_count for each geometry
    affine_body_dynamics->for_each(
        geo_slots,
        [&](const AffineBodyDynamics::ForEachInfo& foreach_info, geometry::SimplicialComplex& sc)
        {
            auto I          = foreach_info.global_index();
            auto dof_offset = sc.meta().find<IndexT>(builtin::dof_offset);
            UIPC_ASSERT(dof_offset, "dof_offset not found on ABD mesh why can it happen?");
            auto dof_count = sc.meta().find<IndexT>(builtin::dof_count);
            UIPC_ASSERT(dof_count, "dof_count not found on ABD mesh why can it happen?");

            IndexT this_dof_count = 12 * sc.instances().size();
            view(*dof_offset)[0]  = offset;
            view(*dof_count)[0]   = this_dof_count;

            offset += this_dof_count;
        });

    UIPC_ASSERT(offset == info.dof_offset() + info.dof_count(), "dof size mismatch");
}

void ABDLinearSubsystem::Impl::report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    // 1. Gradient Count
    constexpr SizeT G12_to_dof = 12;
    SizeT           body_count = abd().body_count();
    auto            dof_count  = body_count * G12_to_dof;

    auto has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    SizeT H12x12_count = 0;

    if(has_complement)
    {
        // 1) Body hessian: kinetic + shape
        if(!info.gradient_only())
            H12x12_count += abd().body_count();

        // 2) Reporters
        auto reporter_view = reporters.view();
        auto grad_counts   = reporter_gradient_offsets_counts.counts();
        auto hess_counts   = reporter_hessian_offsets_counts.counts();

        for(auto&& R : reporter_view)
        {
            ReportExtentInfo extent_info;
            extent_info.m_gradient_only = info.gradient_only();
            R->report_extent(extent_info);

            grad_counts[R->m_index] = extent_info.m_gradient_count;
            hess_counts[R->m_index] = extent_info.m_hessian_count;
        }

        reporter_gradient_offsets_counts.scan(stream());
        reporter_hessian_offsets_counts.scan(stream());

        if(!info.gradient_only())
            H12x12_count += reporter_hessian_offsets_counts.total_count();
    }


    if(dytopo_effect_receiver && !info.gradient_only())
    {
        H12x12_count += dytopo_effect_receiver->hessians().triplet_count;
    }


    auto H3x3_count = H12x12_count * (4 * 4);

    if(info.gradient_only())
    {
        UIPC_ASSERT(H3x3_count == 0,
                    "Hessian block count should be zero (got {}) when gradient_only is true",
                    H3x3_count);
    }

    info.extent(H3x3_count, dof_count);
}

void ABDLinearSubsystem::Impl::assemble(GlobalLinearSystem::DiagInfo& info)
{
    using namespace luisa::compute;

    // 0) Prepare buffers for reporters
    {
        auto N = abd().body_count();

        // Resize reporter gradient/hessian buffers
        reporter_gradient_count = reporter_gradient_offsets_counts.total_count();
        reporter_gradient_indices = device().create_buffer<uint>(reporter_gradient_count);
        reporter_gradient_values = device().create_buffer<Vector12>(reporter_gradient_count);

        reporter_hessian_count = reporter_hessian_offsets_counts.total_count();
        reporter_hessian_row_indices = device().create_buffer<uint>(reporter_hessian_count);
        reporter_hessian_col_indices = device().create_buffer<uint>(reporter_hessian_count);
        reporter_hessian_values = device().create_buffer<Matrix12x12>(reporter_hessian_count);
    }

    bool has_complement =
        has_flags(info.component_flags(), GlobalLinearSystem::ComponentFlags::Complement);

    IndexT hess_offset = 0;

    // 1) Static Topo Effect: Kinetic + Shape + Other Reporters
    if(has_complement)
    {
        _assemble_kinetic_shape(hess_offset, info);
        _assemble_reporters(hess_offset, info);
    }
    else  // contact only
    {
        // Fill gradient buffer with zeros
        auto grad_view = info.gradients();
        Kernel1D zero_kernel = [&](BufferVar<Float> grads) noexcept
        {
            auto idx = dispatch_id().x;
            $if(idx < grads.size())
            {
                grads.write(idx, 0.0f);
            };
        };
        auto zero_shader = device().compile(zero_kernel);
        stream() << zero_shader(grad_view).dispatch(grad_view.size());
    }

    // 2) Dynamic Topology Effect
    _assemble_dytopo_effect(hess_offset, info);

    UIPC_ASSERT(hess_offset == info.hessians().count,
                "Hessian size mismatch: expected {}, got {}",
                info.hessians().count,
                hess_offset);
}

void ABDLinearSubsystem::Impl::_assemble_kinetic_shape(IndexT& hess_offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace luisa::compute;

    // Collect Kinetic
    ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
        info.gradient_only(), body_id_to_kinetic_gradient.view(), body_id_to_kinetic_hessian.view(), dt};
    abd().kinetic->compute_gradient_hessian(this_info);

    // Collect Shape
    for(auto&& [i, cst] : enumerate(abd().constitutions.view()))
    {
        ABDLinearSubsystem::ComputeGradientHessianInfo this_info{
            info.gradient_only(),
            abd().subview(body_id_to_shape_gradient, cst->m_index),
            abd().subview(body_id_to_shape_hessian, cst->m_index),
            dt};

        cst->compute_gradient_hessian(this_info);
    }

    // Kernel for assembling kinetic + shape gradients
    Kernel1D grad_kernel = [&](BufferVar<const IndexT> is_fixed,
                                BufferVar<const IndexT> is_external_kinetic,
                                BufferVar<const Vector12> shape_gradient,
                                BufferVar<const Vector12> kinetic_gradient,
                                BufferVar<Float> gradients) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < is_fixed.size())
        {
            Vector12 src;

            $if(is_fixed.read(i) != 0)
            {
                // if fixed, set to zero
                for(int k = 0; k < 12; ++k)
                    src[k] = 0.0f;
            }
            $else
            {
                src = shape_gradient.read(i);

                // if not external kinetic, add kinetic gradient
                $if(is_external_kinetic.read(i) == 0)
                {
                    Vector12 kin_grad = kinetic_gradient.read(i);
                    for(int k = 0; k < 12; ++k)
                        src[k] += kin_grad[k];
                };
            };

            // Write to gradients
            for(int k = 0; k < 12; ++k)
                gradients.write(i * 12u + cast<uint>(k), src[k]);
        };
    };

    auto grad_shader = device().compile(grad_kernel);
    stream() << grad_shader(abd().body_id_to_is_fixed.view(),
                            abd().body_id_to_external_kinetic.view(),
                            body_id_to_shape_gradient.view(),
                            body_id_to_kinetic_gradient.view(),
                            info.gradients()).dispatch(abd().body_count());

    if(info.gradient_only())
        return;

    auto body_count = abd().body_count();
    auto H3x3_count = body_count * (4 * 4);
    
    // Get subview for hessian triplets
    auto hessians_view = info.hessians();
    auto row_indices = hessians_view.row_indices.subview(hess_offset, H3x3_count);
    auto col_indices = hessians_view.col_indices.subview(hess_offset, H3x3_count);
    auto values = hessians_view.values.subview(hess_offset, H3x3_count);

    // Kernel for assembling kinetic + shape hessians
    Kernel1D hess_kernel = [&](BufferVar<uint> dst_row,
                                BufferVar<uint> dst_col,
                                BufferVar<float3x3> dst_val,
                                BufferVar<const IndexT> is_fixed,
                                BufferVar<const IndexT> is_external_kinetic,
                                BufferVar<const Matrix12x12> shape_hessian,
                                BufferVar<const Matrix12x12> kinetic_hessian,
                                BufferVar<Matrix12x12> diag_hessian_buf) noexcept
    {
        auto I = dispatch_id().x;
        $if(I < is_fixed.size())
        {
            Matrix12x12 H12x12;

            $if(is_fixed.read(I) != 0)
            {
                // Fill kinetic hessian to identity to avoid singularity
                for(int r = 0; r < 12; ++r)
                    for(int c = 0; c < 12; ++c)
                        H12x12(r, c) = (r == c) ? 1.0f : 0.0f;
            }
            $else
            {
                // if not fixed, fill shape hessian
                H12x12 = shape_hessian.read(I);

                // if not external kinetic, add kinetic hessian
                $if(is_external_kinetic.read(I) == 0)
                {
                    Matrix12x12 kin_hess = kinetic_hessian.read(I);
                    for(int r = 0; r < 12; ++r)
                        for(int c = 0; c < 12; ++c)
                            H12x12(r, c) += kin_hess(r, c);
                };
            };

            // record diagonal hessian for diag-inv preconditioner
            diag_hessian_buf.write(I, H12x12);

            // set the lower triangle blocks to zero for robustness
            zero_out_lower(H12x12);

            // Write to triplet matrix - 16 3x3 blocks per body
            IndexT base_idx = I * 16;  // 4*4 blocks per body
            for(int ii = 0; ii < 4; ++ii)
            {
                for(int jj = 0; jj < 4; ++jj)
                {
                    auto entry_idx = base_idx + ii * 4 + jj;
                    dst_row.write(entry_idx, I * 4 + ii);
                    dst_col.write(entry_idx, I * 4 + jj);
                    
                    // Extract 3x3 block
                    Matrix3x3 block;
                    for(int c = 0; c < 3; ++c)
                        for(int r = 0; r < 3; ++r)
                            block[c][r] = H12x12(ii * 3 + r, jj * 3 + c);
                    dst_val.write(entry_idx, block);
                }
            }
        };
    };

    auto hess_shader = device().compile(hess_kernel);
    stream() << hess_shader(row_indices,
                            col_indices,
                            values,
                            abd().body_id_to_is_fixed.view(),
                            abd().body_id_to_external_kinetic.view(),
                            body_id_to_shape_hessian.view(),
                            body_id_to_kinetic_hessian.view(),
                            diag_hessian.view()).dispatch(body_count);

    hess_offset += H3x3_count;
}

void ABDLinearSubsystem::Impl::_assemble_reporters(IndexT& offset,
                                                   GlobalLinearSystem::DiagInfo& info)
{
    using namespace luisa::compute;

    // Fill TripletMatrix and DoubletVector
    for(auto& R : reporters.view())
    {
        AssembleInfo assemble_info{this, R->m_index, info.gradient_only()};
        R->assemble(assemble_info);
    }

    if(reporter_gradient_count > 0)
    {
        Kernel1D grad_kernel = [&](BufferVar<Float> dst,
                                    BufferVar<const uint> src_indices,
                                    BufferVar<const Vector12> src_values,
                                    BufferVar<const IndexT> is_fixed) noexcept
        {
            auto I = dispatch_id().x;
            $if(I < src_indices.size())
            {
                auto body_i = src_indices.read(I);
                Vector12 G12 = src_values.read(I);

                $if(is_fixed.read(body_i) == 0)
                {
                    for(int k = 0; k < 12; ++k)
                    {
                        auto idx = body_i * 12 + k;
                        dst.atomic(idx).fetch_add(G12[k]);
                    }
                };
            };
        };

        auto grad_shader = device().compile(grad_kernel);
        stream() << grad_shader(info.gradients(),
                                reporter_gradient_indices.view(),
                                reporter_gradient_values.view(),
                                abd().body_id_to_is_fixed.view()).dispatch(reporter_gradient_count);
    }

    if(!info.gradient_only() && reporter_hessian_count > 0)
    {
        // get rest
        auto hessians_view = info.hessians();
        auto dst_row = hessians_view.row_indices.subview(offset);
        auto dst_col = hessians_view.col_indices.subview(offset);
        auto dst_val = hessians_view.values.subview(offset);

        Kernel1D hess_kernel = [&](BufferVar<uint> dst_row_buf,
                                    BufferVar<uint> dst_col_buf,
                                    BufferVar<float3x3> dst_val_buf,
                                    BufferVar<const uint> src_row_indices,
                                    BufferVar<const uint> src_col_indices,
                                    BufferVar<const Matrix12x12> src_values,
                                    BufferVar<Matrix12x12> diag_hessian_buf,
                                    BufferVar<const IndexT> is_fixed) noexcept
        {
            auto I = dispatch_id().x;
            $if(I < src_row_indices.size())
            {
                Matrix12x12 H12x12 = src_values.read(I);
                auto body_i = src_row_indices.read(I);
                auto body_j = src_col_indices.read(I);

                bool has_fixed = (is_fixed.read(body_i) != 0 || is_fixed.read(body_j) != 0);

                // Fill diagonal hessian for diag-inv preconditioner
                $if(body_i == body_j && !has_fixed)
                {
                    Matrix12x12 current = diag_hessian_buf.read(body_i);
                    for(int r = 0; r < 12; ++r)
                        for(int c = 0; c < 12; ++c)
                            current(r, c) += H12x12(r, c);
                    diag_hessian_buf.write(body_i, current);
                };

                $if(has_fixed)
                {
                    // Zero out hessian for fixed bodies
                    for(int r = 0; r < 12; ++r)
                        for(int c = 0; c < 12; ++c)
                            H12x12(r, c) = 0.0f;
                }
                $else
                {
                    $if(body_i == body_j)
                    {
                        // Since body_i == body_j, we only fill the upper triangle part
                        zero_out_lower(H12x12);
                    }
                    $elif(body_i > body_j)
                    {
                        // If all the reporters only report upper triangle part, this branch should not be hit
                        for(int r = 0; r < 12; ++r)
                            for(int c = 0; c < 12; ++c)
                                H12x12(r, c) = 0.0f;
                    };
                };

                // Write to triplet matrix - 16 3x3 blocks per entry
                IndexT base_idx = I * 16;  // 4*4 blocks per entry
                for(int ii = 0; ii < 4; ++ii)
                {
                    for(int jj = 0; jj < 4; ++jj)
                    {
                        auto entry_idx = base_idx + ii * 4 + jj;
                        dst_row_buf.write(entry_idx, body_i * 4 + ii);
                        dst_col_buf.write(entry_idx, body_j * 4 + jj);
                        
                        float3x3 block;
                        for(int c = 0; c < 3; ++c)
                            for(int r = 0; r < 3; ++r)
                                block[c][r] = H12x12(ii * 3 + r, jj * 3 + c);
                        dst_val_buf.write(entry_idx, block);
                    }
                }
            };
        };

        auto hess_shader = device().compile(hess_kernel);
        stream() << hess_shader(dst_row,
                                dst_col,
                                dst_val,
                                reporter_hessian_row_indices.view(),
                                reporter_hessian_col_indices.view(),
                                reporter_hessian_values.view(),
                                diag_hessian.view(),
                                abd().body_id_to_is_fixed.view()).dispatch(reporter_hessian_count);

        offset += reporter_hessian_count * (4 * 4);
    }
}

void ABDLinearSubsystem::Impl::_assemble_dytopo_effect(IndexT& offset,
                                                       GlobalLinearSystem::DiagInfo& info)
{
    using namespace luisa::compute;

    auto  vertex_offset = affine_body_vertex_reporter->vertex_offset();
    SizeT dytopo_effect_gradient_count = 0;
    if(dytopo_effect_receiver)
    {
        dytopo_effect_gradient_count =
            dytopo_effect_receiver->gradients().count;
    }

    if(dytopo_effect_gradient_count > 0)
    {
        Kernel1D grad_kernel = [&](BufferVar<const uint> dytopo_indices,
                                    BufferVar<const float3> dytopo_gradients,
                                    BufferVar<Float> gradients,
                                    BufferVar<const IndexT> v2b,
                                    BufferVar<const ABDJacobi> Js,
                                    BufferVar<const IndexT> is_fixed,
                                    IndexT v_offset) noexcept
        {
            auto I = dispatch_id().x;
            $if(I < dytopo_indices.size())
            {
                auto g_i = dytopo_indices.read(I);
                float3 G3 = dytopo_gradients.read(I);

                auto i = g_i - v_offset;
                auto body_i = v2b.read(i);
                ABDJacobi J_i = Js.read(i);

                $if(is_fixed.read(body_i) == 0)
                {
                    Vector12 G12 = J_i.T() * G3;
                    for(int k = 0; k < 12; ++k)
                    {
                        auto idx = body_i * 12 + k;
                        gradients.atomic(idx).fetch_add(G12[k]);
                    }
                };
            };
        };

        auto grad_shader = device().compile(grad_kernel);
        stream() << grad_shader(dytopo_effect_receiver->gradients().indices,
                                dytopo_effect_receiver->gradients().values,
                                info.gradients(),
                                abd().vertex_id_to_body_id.view(),
                                abd().vertex_id_to_J.view(),
                                abd().body_id_to_is_fixed.view(),
                                vertex_offset).dispatch(dytopo_effect_gradient_count);
    }

    if(info.gradient_only())
        return;

    SizeT dytopo_effect_hessian_count = 0;
    if(dytopo_effect_receiver)
        dytopo_effect_hessian_count = dytopo_effect_receiver->hessians().count;

    auto H3x3_count         = dytopo_effect_hessian_count * (4 * 4);
    
    auto hessians_view = info.hessians();
    auto dst_row = hessians_view.row_indices.subview(offset, H3x3_count);
    auto dst_col = hessians_view.col_indices.subview(offset, H3x3_count);
    auto dst_val = hessians_view.values.subview(offset, H3x3_count);

    if(dytopo_effect_hessian_count > 0)
    {
        // Half Contact Hessian
        // ref: https://github.com/spiriMirror/libuipc/issues/272
        Kernel1D hess_kernel = [&](BufferVar<uint> dst_row_buf,
                                    BufferVar<uint> dst_col_buf,
                                    BufferVar<Matrix3x3> dst_val_buf,
                                    BufferVar<const uint> dytopo_row_indices,
                                    BufferVar<const uint> dytopo_col_indices,
                                    BufferVar<const float3x3> dytopo_hessians,
                                    BufferVar<const IndexT> v2b,
                                    BufferVar<const ABDJacobi> Js,
                                    BufferVar<const IndexT> is_fixed,
                                    BufferVar<Matrix12x12> diag_hessian_buf,
                                    IndexT v_offset) noexcept
        {
            auto I = dispatch_id().x;
            $if(I < dytopo_row_indices.size())
            {
                auto g_i = dytopo_row_indices.read(I);
                auto g_j = dytopo_col_indices.read(I);
                float3x3 H3x3 = dytopo_hessians.read(I);

                auto i = g_i - v_offset;
                auto j = g_j - v_offset;

                auto body_i = v2b.read(i);
                auto body_j = v2b.read(j);

                ABDJacobi J_i = Js.read(i);
                ABDJacobi J_j = Js.read(j);

                Matrix12x12 H12x12;

                // We know half contact hessian i <= j
                // but we don't know body_i and body_j order
                // so test and swap if necessary
                IndexT L = body_i;
                IndexT R = body_j;
                $if(body_i > body_j)
                {
                    L = body_j;
                    R = body_i;
                };

                bool has_fixed = (is_fixed.read(body_i) != 0 || is_fixed.read(body_j) != 0);

                $if(has_fixed)
                {
                    for(int r = 0; r < 12; ++r)
                        for(int c = 0; c < 12; ++c)
                            H12x12(r, c) = 0.0f;
                }
                $else
                {
                    $if(body_i < body_j)
                    {
                        H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
                    }
                    $elif(body_i > body_j)
                    {
                        H12x12 = ABDJacobi::JT_H_J(J_j.T(), luisa::transpose(H3x3), J_i);
                    }
                    $else
                    {
                        // body_i == body_j
                        // Two vertices from the same body
                        $if(i != j)
                        {
                            Matrix12x12 H1 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
                            Matrix12x12 H2 = ABDJacobi::JT_H_J(J_j.T(), luisa::transpose(H3x3), J_i);
                            for(int r = 0; r < 12; ++r)
                                for(int c = 0; c < 12; ++c)
                                    H12x12(r, c) = H1(r, c) + H2(r, c);
                        }
                        $else
                        {
                            // i == j
                            H12x12 = ABDJacobi::JT_H_J(J_i.T(), H3x3, J_j);
                        };

                        // Fill diagonal hessian for diag-inv preconditioner
                        Matrix12x12 current = diag_hessian_buf.read(body_i);
                        for(int r = 0; r < 12; ++r)
                            for(int c = 0; c < 12; ++c)
                                current(r, c) += H12x12(r, c);
                        diag_hessian_buf.write(body_i, current);

                        // Since body_i == body_j, we only fill the upper triangle part
                        zero_out_lower(H12x12);
                    };
                };

                // Write to triplet matrix - 16 3x3 blocks per entry
                IndexT base_idx = I * 16;  // 4*4 blocks per entry
                for(int ii = 0; ii < 4; ++ii)
                {
                    for(int jj = 0; jj < 4; ++jj)
                    {
                        auto entry_idx = base_idx + ii * 4 + jj;
                        dst_row_buf.write(entry_idx, L * 4 + ii);
                        dst_col_buf.write(entry_idx, R * 4 + jj);
                        
                        float3x3 block;
                        for(int c = 0; c < 3; ++c)
                            for(int r = 0; r < 3; ++r)
                                block[c][r] = H12x12(ii * 3 + r, jj * 3 + c);
                        dst_val_buf.write(entry_idx, block);
                    }
                }
            };
        };

        auto hess_shader = device().compile(hess_kernel);
        stream() << hess_shader(dst_row,
                                dst_col,
                                dst_val,
                                dytopo_effect_receiver->hessians().row_indices,
                                dytopo_effect_receiver->hessians().col_indices,
                                dytopo_effect_receiver->hessians().values,
                                abd().vertex_id_to_body_id.view(),
                                abd().vertex_id_to_J.view(),
                                abd().body_id_to_is_fixed.view(),
                                diag_hessian.view(),
                                vertex_offset).dispatch(dytopo_effect_hessian_count);
    }

    offset += H3x3_count;
}

void ABDLinearSubsystem::Impl::accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    info.satisfied(true);
}

void ABDLinearSubsystem::Impl::retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    using namespace luisa::compute;

    auto dq = abd().body_id_to_dq.view();
    Kernel1D retrieve_kernel = [&](BufferVar<Vector12> dq_buf,
                                    BufferVar<const Float> x) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < dq_buf.size())
        {
            Vector12 val;
            for(int k = 0; k < 12; ++k)
                val[k] = -x.read(i * 12u + cast<uint>(k));
            dq_buf.write(i, val);
        };
    };

    auto retrieve_shader = device().compile(retrieve_kernel);
    stream() << retrieve_shader(dq, info.solution()).dispatch(abd().body_count());
}
}  // namespace uipc::backend::luisa

namespace uipc::backend::luisa
{
void ABDLinearSubsystem::do_init(InitInfo& info)
{
    m_impl.init();
}

void ABDLinearSubsystem::do_report_extent(GlobalLinearSystem::DiagExtentInfo& info)
{
    m_impl.report_extent(info);
}

void ABDLinearSubsystem::do_assemble(GlobalLinearSystem::DiagInfo& info)
{
    m_impl.assemble(info);
}

void ABDLinearSubsystem::do_accuracy_check(GlobalLinearSystem::AccuracyInfo& info)
{
    m_impl.accuracy_check(info);
}

void ABDLinearSubsystem::do_retrieve_solution(GlobalLinearSystem::SolutionInfo& info)
{
    m_impl.retrieve_solution(info);
}

U64 ABDLinearSubsystem::get_uid() const noexcept
{
    return ABDLinearSubsystemUID;
}

void ABDLinearSubsystem::add_reporter(ABDLinearSubsystemReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter cannot be null");
    check_state(SimEngineState::BuildSystems, "add_reporter");
    m_impl.reporters.register_sim_system(*reporter);
}

void ABDLinearSubsystem::do_report_init_extent(GlobalLinearSystem::InitDofExtentInfo& info)
{
    m_impl.report_init_extent(info);
}

void ABDLinearSubsystem::do_receive_init_dof_info(GlobalLinearSystem::InitDofInfo& info)
{
    m_impl.receive_init_dof_info(world(), info);
}

ABDLinearSubsystem::AssembleInfo::AssembleInfo(Impl* impl, IndexT index, bool gradient_only) noexcept
    : m_impl(impl)
    , m_index(index)
    , m_gradient_only(gradient_only)
{
}

ABDLinearSubsystem::DoubletVectorView<Float, 12> ABDLinearSubsystem::AssembleInfo::gradients() const
{
    auto [offset, count] = (*m_impl->reporter_gradient_offsets_counts)[m_index];
    
    DoubletVectorView<Float, 12> view;
    view.indices = m_impl->reporter_gradient_indices.view(offset, count);
    view.values = m_impl->reporter_gradient_values.view(offset, count);
    view.count = count;
    return view;
}

ABDLinearSubsystem::TripletMatrixView<Float, 12, 12> ABDLinearSubsystem::AssembleInfo::hessians() const
{
    auto [offset, count] = (*m_impl->reporter_hessian_offsets_counts)[m_index];
    
    TripletMatrixView<Float, 12, 12> view;
    view.row_indices = m_impl->reporter_hessian_row_indices.view(offset, count);
    view.col_indices = m_impl->reporter_hessian_col_indices.view(offset, count);
    view.values = m_impl->reporter_hessian_values.view(offset, count);
    view.count = count;
    return view;
}

bool ABDLinearSubsystem::AssembleInfo::gradient_only() const noexcept
{
    return m_gradient_only;
}

void ABDLinearSubsystem::ReportExtentInfo::gradient_count(SizeT size)
{
    m_gradient_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::hessian_count(SizeT size)
{
    m_hessian_count = size;
}

void ABDLinearSubsystem::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}

AffineBodyDynamics::Impl& ABDLinearSubsystem::Impl::abd() const noexcept
{
    return affine_body_dynamics->m_impl;
}
}  // namespace uipc::backend::luisa
