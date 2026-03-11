#pragma once
#include <global_geometry/vertex_reporter.h>
#include <affine_body/affine_body_dynamics.h>

namespace uipc::backend::luisa
{
class AffineBodyBodyReporter;

/**
 * @brief Vertex reporter for affine body dynamics in LuisaCompute backend
 * 
 * This class reports vertex information from affine body dynamics to the global
 * vertex management system. It handles vertex count reporting, attribute 
 * initialization/updates, and displacement reporting for simulation.
 */
class AffineBodyVertexReporter final : public VertexReporter
{
  public:
    using VertexReporter::VertexReporter;

    /**
     * @brief Implementation class containing internal state and helper methods
     */
    class Impl
    {
      public:
        /**
         * @brief Report the number of vertices to the global vertex manager
         * @param info Vertex count information to fill
         */
        void report_count(VertexCountInfo& info);

        /**
         * @brief Initialize vertex attributes
         * @param info Vertex attribute information to initialize
         */
        void init_attributes(VertexAttributeInfo& info);

        /**
         * @brief Update vertex attributes
         * @param info Vertex attribute information to update
         */
        void update_attributes(VertexAttributeInfo& info);

        /**
         * @brief Report vertex displacements
         * @param info Vertex displacement information to report
         */
        void report_displacements(VertexDisplacementInfo& info);

        /// Pointer to the affine body dynamics system
        AffineBodyDynamics* affine_body_dynamics = nullptr;
        
        /// Pointer to the body reporter (optional)
        AffineBodyBodyReporter* body_reporter = nullptr;

        /**
         * @brief Get reference to ABD implementation
         * @return Reference to AffineBodyDynamics implementation
         */
        AffineBodyDynamics::Impl& abd() { return affine_body_dynamics->m_impl; }

        /// Flag indicating whether attribute update is required before next simulation step
        bool require_update_attributes = false;
    };

    /**
     * @brief Request to update vertex attributes before next simulation step
     * 
     * This method can be called to mark that vertex attributes need to be
     * updated before the next simulation step begins.
     */
    void request_attribute_update() noexcept;

  protected:
    /**
     * @brief Build the reporter system
     * @param info Build information
     */
    virtual void do_build(BuildInfo& info) override;

    /**
     * @brief Report vertex count to global vertex manager
     * @param info Vertex count information
     */
    virtual void do_report_count(VertexCountInfo& info) override;

    /**
     * @brief Report vertex attributes to global vertex manager
     * @param info Vertex attribute information
     */
    virtual void do_report_attributes(VertexAttributeInfo& info) override;

    /**
     * @brief Report vertex displacements to global vertex manager
     * @param info Vertex displacement information
     */
    virtual void do_report_displacements(VertexDisplacementInfo& info) override;

    /**
     * @brief Get unique identifier for this reporter
     * @return Unique 64-bit identifier
     */
    virtual U64 get_uid() const noexcept override;

  private:
    Impl m_impl;
};

}  // namespace uipc::backend::luisa
