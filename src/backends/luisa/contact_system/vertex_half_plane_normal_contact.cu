#include <contact_system/vertex_half_plane_normal_contact.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <utils/make_spd.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>

namespace uipc::backend::luisa
{
void VertexHalfPlaneNormalContact::do_build(ContactReporter::BuildInfo& info)
{
    m_impl.global_trajectory_filter = require<GlobalTrajectoryFilter>();
    m_impl.global_contact_manager   = require<GlobalContactManager>();
    m_impl.global_vertex_manager    = require<GlobalVertexManager>();
    m_impl.vertex_reporter          = require<HalfPlaneVertexReporter>();
    auto dt_attr = world().scene().config().find<Float>("dt");
    m_impl.dt    = dt_attr->view()[0];

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

BufferView<ContactCoeff> VertexHalfPlaneNormalContact::BaseInfo::contact_tabular() const
{
    return m_impl->global_contact_manager->contact_tabular();
}

BufferView<Vector2i> VertexHalfPlaneNormalContact::BaseInfo::PHs() const
{
    return m_impl->veretx_half_plane_trajectory_filter->PHs();
}

BufferView<Vector3> VertexHalfPlaneNormalContact::BaseInfo::positions() const
{
    return m_impl->global_vertex_manager->positions();
}

BufferView<Vector3> VertexHalfPlaneNormalContact::BaseInfo::prev_positions() const
{
    return m_impl->global_vertex_manager->prev_positions();
}

BufferView<Vector3> VertexHalfPlaneNormalContact::BaseInfo::rest_positions() const
{
    return m_impl->global_vertex_manager->rest_positions();
}

BufferView<Float> VertexHalfPlaneNormalContact::BaseInfo::thicknesses() const
{
    return m_impl->global_vertex_manager->thicknesses();
}

BufferView<IndexT> VertexHalfPlaneNormalContact::BaseInfo::contact_element_ids() const
{
    return m_impl->global_vertex_manager->contact_element_ids();
}


BufferView<IndexT> VertexHalfPlaneNormalContact::BaseInfo::subscene_element_ids() const
{
    return m_impl->global_vertex_manager->subscene_element_ids();
}

Float VertexHalfPlaneNormalContact::BaseInfo::d_hat() const
{
    return m_impl->global_contact_manager->d_hat();
}

BufferView<Float> VertexHalfPlaneNormalContact::BaseInfo::d_hats() const
{
    return m_impl->global_vertex_manager->d_hats();
}

Float VertexHalfPlaneNormalContact::BaseInfo::dt() const
{
    return m_impl->dt;
}

Float VertexHalfPlaneNormalContact::BaseInfo::eps_velocity() const
{
    return m_impl->global_contact_manager->eps_velocity();
}

IndexT VertexHalfPlaneNormalContact::BaseInfo::half_plane_vertex_offset() const
{
    return m_impl->vertex_reporter->vertex_offset();
}

BufferView<Float> VertexHalfPlaneNormalContact::EnergyInfo::energies() const noexcept
{
    return m_energies;
}


BufferView<Vector2i> VertexHalfPlaneNormalContact::PHs() const noexcept
{
    return m_impl.veretx_half_plane_trajectory_filter->PHs();
}

BufferView<Float> VertexHalfPlaneNormalContact::energies() const noexcept
{
    return m_impl.energies;
}

BufferView<Float> VertexHalfPlaneNormalContact::gradients() const noexcept
{
    return m_impl.gradients;
}

BufferView<Float> VertexHalfPlaneNormalContact::hessians() const noexcept
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
    SimSystemSlot<HalfPlaneVertexReporter> half_plane_vertex_reporter;

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
        // Copy from device buffer to host vector
        vector<Vector2i> h_ph_data(PHs.size());
        auto& engine = vertex_half_plane_normal_contact->world().sim_engine();
        auto& stream = static_cast<SimEngine&>(engine).compute_stream();
        stream << PHs.copy_to(h_ph_data.data())
               << synchronize();
        std::memcpy(topo_view.data(), h_ph_data.data(), h_ph_data.size() * sizeof(Vector2i));

        auto v_offset = half_plane_vertex_reporter->vertex_offset();
        for(Vector2i& topo : topo_view)
            topo[1] += v_offset;

        auto energy = energy_geo.instances().find<Float>("energy");
        if(!energy)
        {
            energy = energy_geo.instances().create<Float>("energy", 0.0f);
        }

        auto energy_view = view(*energy);
        vector<Float> h_energy_data(PHs.size());
        stream << energies.copy_to(h_energy_data.data())
               << synchronize();
        std::memcpy(energy_view.data(), h_energy_data.data(), h_energy_data.size() * sizeof(Float));
    }

    void get_contact_gradient(std::string_view prim_type, geometry::Geometry& vert_grad) override
    {
        auto PH_grads = vertex_half_plane_normal_contact->gradients();
        // Gradient is stored as (index, x, y, z) tuples - 4 floats per entry
        SizeT entry_count = PH_grads.size() / 4;
        vert_grad.instances().resize(entry_count);
        
        auto i = vert_grad.instances().find<IndexT>("i");
        if(!i)
        {
            i = vert_grad.instances().create<IndexT>("i", -1);
        }
        auto i_view = view(*i);

        auto grad = vert_grad.instances().find<Vector3>("grad");
        if(!grad)
        {
            grad = vert_grad.instances().create<Vector3>("grad", Vector3::Zero());
        }
        auto grad_view = view(*grad);

        // Copy gradient data from device
        vector<Float> h_grad_data(PH_grads.size());
        auto& engine = vertex_half_plane_normal_contact->world().sim_engine();
        auto& stream = static_cast<SimEngine&>(engine).compute_stream();
        stream << PH_grads.copy_to(h_grad_data.data())
               << synchronize();
        
        // Parse (index, x, y, z) tuples
        for(SizeT idx = 0; idx < entry_count; ++idx)
        {
            i_view[idx] = static_cast<IndexT>(h_grad_data[idx * 4 + 0]);
            grad_view[idx][0] = h_grad_data[idx * 4 + 1];
            grad_view[idx][1] = h_grad_data[idx * 4 + 2];
            grad_view[idx][2] = h_grad_data[idx * 4 + 3];
        }
    }

    void get_contact_hessian(std::string_view prim_type, geometry::Geometry& vert_hess) override
    {
        auto PH_hess = vertex_half_plane_normal_contact->hessians();
        // Hessian is stored as (row, col, 3x3 matrix) - 11 floats per entry (2 indices + 9 matrix elements)
        SizeT entry_count = PH_hess.size() / 11;
        vert_hess.instances().resize(entry_count);
        
        auto i = vert_hess.instances().find<IndexT>("i");
        if(!i)
        {
            i = vert_hess.instances().create<IndexT>("i", -1);
        }
        auto i_view = view(*i);

        auto j = vert_hess.instances().find<IndexT>("j");
        if(!j)
        {
            j = vert_hess.instances().create<IndexT>("j", -1);
        }
        auto j_view = view(*j);

        auto hess = vert_hess.instances().find<Matrix3x3>("hess");
        if(!hess)
        {
            hess = vert_hess.instances().create<Matrix3x3>("hess", Matrix3x3::Zero());
        }
        auto hess_view = view(*hess);

        // Copy hessian data from device
        vector<Float> h_hess_data(PH_hess.size());
        auto& engine = vertex_half_plane_normal_contact->world().sim_engine();
        auto& stream = static_cast<SimEngine&>(engine).compute_stream();
        stream << PH_hess.copy_to(h_hess_data.data())
               << synchronize();
        
        // Parse (row, col, 3x3 matrix) tuples
        for(SizeT idx = 0; idx < entry_count; ++idx)
        {
            i_view[idx] = static_cast<IndexT>(h_hess_data[idx * 11 + 0]);
            j_view[idx] = static_cast<IndexT>(h_hess_data[idx * 11 + 1]);
            // Matrix is stored in row-major order
            for(int row = 0; row < 3; ++row)
            {
                for(int col = 0; col < 3; ++col)
                {
                    hess_view[idx](row, col) = h_hess_data[idx * 11 + 2 + row * 3 + col];
                }
            }
        }
    }
};

REGISTER_SIM_SYSTEM(VertexHalfPlaneNormalContactExporter);
}  // namespace uipc::backend::luisa
