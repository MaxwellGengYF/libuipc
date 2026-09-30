#pragma once
/********************************************************************
 * @file contact_system_feature.h
 * @brief The backend half of the frontend's contact-system feature (port of
 * src/backends/cuda/contact_system/contact_system_feature.h, 22 lines).
 *
 * `ContactExporterManager` creates one of these in `do_build()` and inserts the
 * `core::ContactSystemFeature` it wraps into the scene's feature collection, so the
 * frontend can read the per-primitive-type contact energy / gradient / Hessian back
 * (`frontend.contact_energy(prim_type, geo)` etc.).  Namespace change only.
 *********************************************************************/
#include <sim_system.h>
#include <uipc/core/contact_system_feature.h>

namespace uipc::backend::luisa
{
class ContactExporterManager;

class ContactSystemFeatureOverrider final : public core::ContactSystemFeatureOverrider
{
  public:
    ContactSystemFeatureOverrider(ContactExporterManager* contact_system);

  private:
    vector<std::string> get_contact_primitive_types() const override;

    void get_contact_gradient(std::string_view prim_type, geometry::Geometry& vert_grad) override;
    void get_contact_hessian(std::string_view prim_type, geometry::Geometry& vert_hess) override;
    void get_contact_energy(std::string_view prim_type, geometry::Geometry& prims) override;

    SimSystemSlot<ContactExporterManager> m_manager;
};
}  // namespace uipc::backend::luisa
