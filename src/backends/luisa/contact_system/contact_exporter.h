#pragma once
/********************************************************************
 * @file contact_exporter.h
 * @brief The per-primitive-type contact readback base (port of
 * src/backends/cuda/contact_system/contact_exporter.h, 39 lines).
 *
 * A concrete exporter (one per contact primitive type, e.g. the IPC simplex/half-plane
 * exporters of T8.1b) knows how to write the *aggregated* contact energy / gradient /
 * Hessian of one primitive type into a `geometry::Geometry`; `ContactExporterManager`
 * dispatches to it and the `ContactSystemFeature` surfaces it to the frontend.  Namespace
 * change only (the class body is host code).
 *********************************************************************/
#include <sim_system.h>
#include <uipc/geometry/geometry.h>

namespace uipc::backend::luisa
{
class ContactExporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    std::string_view prim_type() const noexcept;

    class BuildInfo
    {
      public:
    };

  protected:
    virtual void             do_build(BuildInfo& info)      = 0;
    virtual std::string_view get_prim_type() const noexcept = 0;

    virtual void get_contact_energy(std::string_view    prim_type,
                                    geometry::Geometry& energy_geo) = 0;

    virtual void get_contact_gradient(std::string_view    prim_type,
                                      geometry::Geometry& vert_grad) = 0;

    virtual void get_contact_hessian(std::string_view    prim_type,
                                     geometry::Geometry& vert_hess) = 0;

  private:
    virtual void do_build() override final;

    friend class ContactExporterManager;
    void contact_energy(std::string_view prim_type, geometry::Geometry& energy_geo);
    void contact_gradient(std::string_view prim_type, geometry::Geometry& vert_grad);
    void contact_hessian(std::string_view prim_type, geometry::Geometry& vert_hess);
};
}  // namespace uipc::backend::luisa
