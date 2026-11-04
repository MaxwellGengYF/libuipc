/********************************************************************
 * @file vertex_half_plane_normal_contact.cpp
 * @brief T8.1b — the vertex/half-plane normal contact base bodies + its exporter (port of
 * src/backends/cuda/contact_system/vertex_half_plane_normal_contact.cu, 277 reference lines).
 *
 * The four forwarding overrides are the *unsliced* variants of the simplex model's: one
 * primitive type means the manager's whole energy / gradient / Hessian view belongs to this
 * model, so `m_impl.energies/gradients/hessians` are stored as-is and `do_assemble` writes
 * one doublet and one triplet per pair.  `get_contact_energy` is the one place the
 * half-plane instance bias shows up in host code: the exported `topo`'s half-plane index is
 * biased by `HalfPlaneVertexReporter::vertex_offset()` (the reference's
 * `for(Vector2i& topo : topo_view) topo[1] += v_offset;`).
 *********************************************************************/
#include <contact_system/vertex_half_plane_normal_contact.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <luisa_device/math/matrix_ops.h>  // lps::make_spd_inplace (the reference's utils/make_spd.h)
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <global_geometry/global_vertex_manager.h>
#include <contact_system/global_contact_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <luisa_device/math/distance/distance.h>

namespace uipc::backend::luisa
{
void VertexHalfPlaneNormalContact::do_build(ContactReporter::BuildInfo& info)
{
    m_impl.global_trajectory_filter = require<GlobalTrajectoryFilter>();
    m_impl.global_contact_manager   = require<GlobalContactManager>();
    m_impl.global_vertex_manager    = require<GlobalVertexManager>();
    m_impl.vertex_reporter          = require<HalfPlaneVertexReporter>();
    m_impl.dt_attr = world().scene().config().find<Float>("dt");
    UIPC_ASSERT(m_impl.dt_attr, "Scene config must have a 'dt' attribute.");

    BuildInfo this_info;
    do_build(this_info);

    on_init_scene(
        [this]
        {
            m_impl.veretx_half_plane_trajectory_filter =
                m_impl.global_trajectory_filter->find<VertexHalfPlaneTrajectoryFilter>();
        });
}

void VertexHalfPlaneNormalContact::do_report_energy_extent(GlobalContactManager::EnergyExtentInfo& info)
{
    auto& filter = m_impl.veretx_half_plane_trajectory_filter;

    SizeT count     = filter->PHs().size();
    m_impl.PH_count = count;

    info.energy_count(count);
}

void VertexHalfPlaneNormalContact::do_compute_energy(GlobalContactManager::EnergyInfo& info)
{
    EnergyInfo this_info{&m_impl};
    this_info.m_energies = info.energies();
    m_impl.energies      = this_info.m_energies;

    do_compute_energy(this_info);
}

void VertexHalfPlaneNormalContact::do_report_gradient_hessian_extent(
    GlobalContactManager::GradientHessianExtentInfo& info)
{
    auto& filter = m_impl.veretx_half_plane_trajectory_filter;

    SizeT count = filter->PHs().size();

    info.gradient_count(count);

    if(info.gradient_only())
        return;

    info.hessian_count(count);
}


void VertexHalfPlaneNormalContact::do_assemble(GlobalContactManager::GradientHessianInfo& info)
{
    ContactInfo this_info{&m_impl};
    this_info.m_gradient_only = info.gradient_only();

    this_info.m_gradients = info.gradients();
    m_impl.gradients      = this_info.m_gradients;
    this_info.m_hessians  = info.hessians();
    m_impl.hessians       = this_info.m_hessians;

    // let subclass to fill in the data
    do_assemble(this_info);
}

lud::LCBufferView<ContactCoeff> VertexHalfPlaneNormalContact::BaseInfo::contact_tabular() const
{
    return m_impl->global_contact_manager->contact_tabular();
}

IndexT VertexHalfPlaneNormalContact::BaseInfo::contact_tabular_size() const
{
    return m_impl->global_contact_manager->contact_tabular_size();
}

lud::LCBufferView<LVector2i> VertexHalfPlaneNormalContact::BaseInfo::PHs() const
{
    return m_impl->veretx_half_plane_trajectory_filter->PHs();
}

lud::LCBufferView<LVector3> VertexHalfPlaneNormalContact::BaseInfo::positions() const
{
    return m_impl->global_vertex_manager->positions();
}

lud::LCBufferView<LVector3> VertexHalfPlaneNormalContact::BaseInfo::prev_positions() const
{
    return m_impl->global_vertex_manager->prev_positions();
}

lud::LCBufferView<LVector3> VertexHalfPlaneNormalContact::BaseInfo::rest_positions() const
{
    return m_impl->global_vertex_manager->rest_positions();
}

lud::LCBufferView<Float> VertexHalfPlaneNormalContact::BaseInfo::thicknesses() const
{
    return m_impl->global_vertex_manager->thicknesses();
}

lud::LCBufferView<IndexT> VertexHalfPlaneNormalContact::BaseInfo::contact_element_ids() const
{
    return m_impl->global_vertex_manager->contact_element_ids();
}


lud::LCBufferView<IndexT> VertexHalfPlaneNormalContact::BaseInfo::subscene_element_ids() const
{
    return m_impl->global_vertex_manager->subscene_element_ids();
}

Float VertexHalfPlaneNormalContact::BaseInfo::d_hat() const
{
    return m_impl->global_contact_manager->d_hat();
}

lud::LCBufferView<Float> VertexHalfPlaneNormalContact::BaseInfo::d_hats() const
{
    return m_impl->global_vertex_manager->d_hats();
}

Float VertexHalfPlaneNormalContact::BaseInfo::dt() const
{
    return m_impl->dt_attr->view()[0];
}

Float VertexHalfPlaneNormalContact::BaseInfo::eps_velocity() const
{
    return m_impl->global_contact_manager->eps_velocity();
}

IndexT VertexHalfPlaneNormalContact::BaseInfo::half_plane_vertex_offset() const
{
    return m_impl->vertex_reporter->vertex_offset();
}

lud::LBufferView<Float> VertexHalfPlaneNormalContact::EnergyInfo::energies() const noexcept
{
    return m_energies;
}


lud::LCBufferView<LVector2i> VertexHalfPlaneNormalContact::PHs() const noexcept
{
    return m_impl.veretx_half_plane_trajectory_filter->PHs();
}

lud::LCBufferView<Float> VertexHalfPlaneNormalContact::energies() const noexcept
{
    return m_impl.energies;
}

lla::LCDoubletVectorView<Float, 3> VertexHalfPlaneNormalContact::gradients() const noexcept
{
    return m_impl.gradients;
}

lla::LCTripletMatrixView<Float, 3, 3> VertexHalfPlaneNormalContact::hessians() const noexcept
{
    return m_impl.hessians;
}
}  // namespace uipc::backend::luisa

#include <contact_system/contact_exporter.h>

namespace uipc::backend::luisa
{
class VertexHalfPlaneNormalContactExporter : public ContactExporter
{
  public:
    using ContactExporter::ContactExporter;

    SimSystemSlot<VertexHalfPlaneNormalContact> vertex_half_plane_normal_contact;
    SimSystemSlot<HalfPlaneVertexReporter>      half_plane_vertex_reporter;

    void do_build(BuildInfo& info) override
    {
        vertex_half_plane_normal_contact =
            require<VertexHalfPlaneNormalContact>(QueryOptions{.exact = false});
        half_plane_vertex_reporter = require<HalfPlaneVertexReporter>();
    }

    std::string_view get_prim_type() const noexcept override { return "PH+N"; }

    void get_contact_energy(std::string_view prim_type, geometry::Geometry& energy_geo) override
    {
        auto PHs      = vertex_half_plane_normal_contact->PHs();
        auto energies = vertex_half_plane_normal_contact->energies();

        UIPC_ASSERT(PHs.size() == energies.size(), "PHs and energies must have the same size.");

        energy_geo.instances().resize(PHs.size());
        auto topo = energy_geo.instances().find<Vector2i>("topo");
        if(!topo)
        {
            topo = energy_geo.instances().create<Vector2i>("topo", Vector2i::Zero());
        }

        auto topo_view = view(*topo);
        {
            // the reference's `PHs.copy_to(topo_view.data())` with the port's staged copy
            if(!PHs.empty())
            {
                std::vector<LVector2i> staging(PHs.size());
                PHs.copy_to(span<LVector2i>{staging});
                lud::main_stream().synchronize();
                for(SizeT i = 0; i < static_cast<SizeT>(topo_view.size()); ++i)
                {
                    topo_view[i] = to_eigen(staging[i]);
                }
            }
        }
        auto v_offset = half_plane_vertex_reporter->vertex_offset();

        for(Vector2i& topo : topo_view)
            topo[1] += v_offset;


        auto energy = energy_geo.instances().find<Float>("energy");
        if(!energy)
        {
            energy = energy_geo.instances().create<Float>("energy", 0.0f);
        }

        auto energy_view = view(*energy);
        energies.copy_to(span<Float>{energy_view.data(), energy_view.size()});
        lud::main_stream().synchronize();
    }

    void get_contact_gradient(std::string_view prim_type, geometry::Geometry& vert_grad) override
    {
        auto PH_grads = vertex_half_plane_normal_contact->gradients();
        vert_grad.instances().resize(PH_grads.doublet_count());
        auto i = vert_grad.instances().find<IndexT>("i");
        if(!i)
        {
            i = vert_grad.instances().create<IndexT>("i", -1);
        }
        auto i_view = view(*i);
        if(!PH_grads.indices().empty())
        {
            std::vector<std::int32_t> staging(PH_grads.indices().size());
            PH_grads.indices().copy_to(span<std::int32_t>{staging});
            lud::main_stream().synchronize();
            for(SizeT k = 0; k < static_cast<SizeT>(i_view.size()); ++k)
            {
                i_view[k] = static_cast<IndexT>(staging[k]);
            }
        }

        auto grad = vert_grad.instances().find<Vector3>("grad");
        if(!grad)
        {
            grad = vert_grad.instances().create<Vector3>("grad", Vector3::Zero());
        }
        auto grad_view = view(*grad);
        if(!PH_grads.values().empty())
        {
            std::vector<Float> staging(PH_grads.values().size());
            PH_grads.values().copy_to(span<Float>{staging});
            lud::main_stream().synchronize();
            for(SizeT k = 0; k < static_cast<SizeT>(grad_view.size()); ++k)
            {
                grad_view[k] = Vector3{staging[3 * k + 0], staging[3 * k + 1], staging[3 * k + 2]};
            }
        }
    }

    void get_contact_hessian(std::string_view prim_type, geometry::Geometry& vert_hess) override
    {
        auto PH_hess = vertex_half_plane_normal_contact->hessians();
        vert_hess.instances().resize(PH_hess.triplet_count());
        auto i = vert_hess.instances().find<IndexT>("i");
        if(!i)
        {
            i = vert_hess.instances().create<IndexT>("i", -1);
        }
        auto i_view = view(*i);
        if(!PH_hess.row_indices().empty())
        {
            std::vector<std::int32_t> staging(PH_hess.row_indices().size());
            PH_hess.row_indices().copy_to(span<std::int32_t>{staging});
            lud::main_stream().synchronize();
            for(SizeT k = 0; k < static_cast<SizeT>(i_view.size()); ++k)
            {
                i_view[k] = static_cast<IndexT>(staging[k]);
            }
        }

        auto j = vert_hess.instances().find<IndexT>("j");
        if(!j)
        {
            j = vert_hess.instances().create<IndexT>("j", -1);
        }
        auto j_view = view(*j);
        if(!PH_hess.col_indices().empty())
        {
            std::vector<std::int32_t> staging(PH_hess.col_indices().size());
            PH_hess.col_indices().copy_to(span<std::int32_t>{staging});
            lud::main_stream().synchronize();
            for(SizeT k = 0; k < static_cast<SizeT>(j_view.size()); ++k)
            {
                j_view[k] = static_cast<IndexT>(staging[k]);
            }
        }

        auto hess = vert_hess.instances().find<Matrix3x3>("hess");
        if(!hess)
        {
            hess = vert_hess.instances().create<Matrix3x3>("hess", Matrix3x3::Zero());
        }
        auto hess_view = view(*hess);
        if(!PH_hess.values().empty())
        {
            std::vector<Float> staging(PH_hess.values().size());
            PH_hess.values().copy_to(span<Float>{staging});
            lud::main_stream().synchronize();
            for(SizeT k = 0; k < static_cast<SizeT>(hess_view.size()); ++k)
            {
                for(int c = 0; c < 3; ++c)
                {
                    for(int r = 0; r < 3; ++r)
                    {
                        hess_view[k](r, c) =
                            staging[k * 9 + static_cast<SizeT>(c) * 3 + static_cast<SizeT>(r)];
                    }
                }
            }
        }
    }
};

REGISTER_SIM_SYSTEM(VertexHalfPlaneNormalContactExporter);
}  // namespace uipc::backend::luisa
