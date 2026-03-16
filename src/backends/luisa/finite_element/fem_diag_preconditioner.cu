#include <type_define.h>
#include <Eigen/Dense>
#include <linear_system/local_preconditioner.h>
#include <finite_element/finite_element_method.h>
#include <linear_system/global_linear_system.h>
#include <finite_element/fem_linear_subsystem.h>
#include <global_geometry/global_vertex_manager.h>
#include <kernel_cout.h>
#include <uipc/geometry/simplicial_complex.h>
#include <luisa/dsl/syntax.h>

namespace uipc::backend::luisa
{
class FEMDiagPreconditioner : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    FiniteElementMethod* finite_element_method = nullptr;
    GlobalLinearSystem*  global_linear_system  = nullptr;
    FEMLinearSubsystem*  fem_linear_subsystem  = nullptr;

    luisa::compute::Buffer<Matrix3x3> diag_inv;

    virtual void do_build(BuildInfo& info) override
    {
        finite_element_method       = &require<FiniteElementMethod>();
        global_linear_system        = &require<GlobalLinearSystem>();
        fem_linear_subsystem        = &require<FEMLinearSubsystem>();
        auto& global_vertex_manager = require<GlobalVertexManager>();

        // If ANY FEM geometry has mesh_part, defer to FEMMASPreconditioner,
        // which handles both partitioned (MAS) and unpartitioned (diag) vertices.
        auto geo_slots = world().scene().geometries();
        for(SizeT i = 0; i < geo_slots.size(); i++)
        {
            auto& geo = geo_slots[i]->geometry();
            auto* sc  = geo.as<geometry::SimplicialComplex>();
            if(sc && sc->dim() >= 1)
            {
                auto mesh_part = sc->vertices().find<IndexT>("mesh_part");
                if(mesh_part)
                {
                    throw SimSystemException(
                        "FEMDiagPreconditioner: mesh_part found, "
                        "deferring to FEMMASPreconditioner.");
                }
            }
        }

        // This FEMDiagPreconditioner depends on FEMLinearSubsystem
        info.connect(fem_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace luisa::compute;

        diag_inv = engine().luisa_device().create_buffer<Matrix3x3>(finite_element_method->xs().size());

        // 1) collect diagonal blocks
        Kernel1D collect_diag_kernel = [&](BufferVar<Matrix3x3> diag_inv, 
                                           BufferVar<Triplet<Float, 3, 3>> triplet,
                                           UInt fem_segment_offset,
                                           UInt fem_segment_count) noexcept
        {
            UInt I = dispatch_x();
            
            auto g_i = triplet.read(I).row;
            auto g_j = triplet.read(I).col;
            auto H3x3 = triplet.read(I).value;

            UInt i = g_i - fem_segment_offset;
            UInt j = g_j - fem_segment_offset;

            if_(i < fem_segment_count && j < fem_segment_count, [&]
            {
                if_(i == j, [&]
                {
                    diag_inv.write(i, inverse(H3x3));
                });
            });
        };

        auto shader = engine().luisa_device().compile(collect_diag_kernel);
        engine().compute_stream() << shader(diag_inv, info.A().triplet_buffer(), 
                                            info.dof_offset() / 3, 
                                            info.dof_count() / 3).dispatch(info.A().triplet_count())
                                  << synchronize();
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        using namespace luisa::compute;
        auto converged = info.converged();

        Kernel1D apply_precond_kernel = [&](BufferVar<Float> r,
                                            BufferVar<Float> z,
                                            BufferVar<Int> converged,
                                            BufferVar<Matrix3x3> diag_inv) noexcept
        {
            UInt i = dispatch_x();
            
            if_(converged.read(0) == 0, [&]
            {
                // z.segment<3>(i * 3) = diag_inv(i) * r.segment<3>(i * 3)
                Float3 r_vec = make_float3(r.read(i * 3), r.read(i * 3 + 1), r.read(i * 3 + 2));
                Matrix3x3 d_inv = diag_inv.read(i);
                Float3 z_vec = d_inv * r_vec;
                z.write(i * 3, z_vec.x);
                z.write(i * 3 + 1, z_vec.y);
                z.write(i * 3 + 2, z_vec.z);
            });
        };

        auto shader = engine().luisa_device().compile(apply_precond_kernel);
        engine().compute_stream() << shader(info.r(), info.z(), converged, diag_inv).dispatch(diag_inv.size())
                                  << synchronize();
    }
};

REGISTER_SIM_SYSTEM(FEMDiagPreconditioner);
}  // namespace uipc::backend::luisa
