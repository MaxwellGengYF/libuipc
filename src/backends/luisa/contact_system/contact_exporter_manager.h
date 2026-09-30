#pragma once
/********************************************************************
 * @file contact_exporter_manager.h
 * @brief Registry / dispatcher of the per-primitive-type contact exporters (port of
 * src/backends/cuda/contact_system/contact_exporter_manager.h, 42 lines).
 *
 * It also owns the scene's `core::ContactSystemFeature` (created in `do_build`, see the
 * .cpp) so the frontend can read the aggregated contact energy / gradient / Hessian
 * back per primitive type.
 *
 * Deviations (reported): the reference header also includes
 * `contact_system/simplex_frictional_contact.h`, `simplex_normal_contact.h`,
 * `vertex_half_plane_normal_contact.h` and `vertex_half_plane_frictional_contact.h`.
 * Those four types are the IPC contact *models* — T8.1b, not ported yet — and the class
 * body never names them (they are dead includes in the reference), so the port drops
 * them (the T5.3 "drop the include of an un-ported sibling" precedent).  The four
 * trajectory-filter / half-plane-vertex-reporter includes are ported and kept verbatim.
 *********************************************************************/
#include <sim_system.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>

namespace uipc::backend::luisa
{
class ContactExporter;

class ContactExporterManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

  private:
    friend class ContactSystemFeatureOverrider;

    void do_build() override;
    void init();

    void get_contact_energy(std::string_view prim_type, geometry::Geometry& prim_energy);
    void get_contact_gradient(std::string_view prim_type, geometry::Geometry& prim_grad);
    void get_contact_hessian(std::string_view prim_type, geometry::Geometry& prim_hess);

    vector<std::string> get_contact_primitive_types() const;

    unordered_map<std::string, ContactExporter*> m_exporter_map;
    SimSystemSlotCollection<ContactExporter>     m_contact_exporters;
    vector<std::string>                          m_contact_prim_types;

    friend class ContactExporter;
    void add_exporter(ContactExporter* exporter);

    ContactExporter* find_exporter(std::string_view prim_type) const;
    void _create_prim_type_on_geo(std::string_view prim_type, geometry::Geometry& geo);
};
}  // namespace uipc::backend::luisa
