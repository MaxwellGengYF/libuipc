/********************************************************************
 * @file parity_probe.cpp
 * @brief T9.3a — probe extraction (see parity_probe.h for the contract).
 *********************************************************************/
#include <parity_probe.h>

#include <uipc/backend/visitors/world_visitor.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/core/affine_body_state_accessor_feature.h>
#include <uipc/core/engine.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/core/world.h>
#include <uipc/geometry/simplicial_complex.h>

#include <cmath>
#include <stdexcept>

namespace uipc::parity
{
using namespace uipc;

namespace
{
    void push_vec3(std::vector<double>& out, const Vector3& v)
    {
        out.push_back(v(0));
        out.push_back(v(1));
        out.push_back(v(2));
    }

    void push_mat4(std::vector<double>& out, const Matrix4x4& m)
    {
        // Column-major, the storage order of `Matrix4x4` (Eigen default), so the reader
        // does not have to know which convention the writer used.
        for(int c = 0; c < 4; ++c)
            for(int r = 0; r < 4; ++r)
                out.push_back(m(r, c));
    }

    /// `builtin::position` of one scene geometry.
    bool collect_vertices(const geometry::SimplicialComplex& sc,
                          int                                geometry_index,
                          std::vector<ArrayBlock>&           blocks)
    {
        auto position = sc.vertices().find<Vector3>(builtin::position);
        if(!position)
            return false;
        auto                view = position->view();
        std::vector<double> data;
        data.reserve(view.size() * 3);
        for(const Vector3& v : view)
            push_vec3(data, v);
        blocks.push_back(ArrayBlock{"vertices", geometry_index, 3, std::move(data)});
        return true;
    }

    /// The two affine-body probes. Returns false when the backend publishes no
    /// affine-body state (a finite-element-only scene, for instance).
    bool collect_affine_body_state(core::World& world, std::vector<ArrayBlock>& blocks)
    {
        auto feature = world.features().find<core::AffineBodyStateAccessorFeature>();
        if(!feature)
            return false;
        const SizeT body_count = feature->body_count();
        if(body_count == 0)
            return false;

        auto state = feature->create_geometry(0, body_count);
        // create_geometry deliberately creates neither attribute
        // (src/core/core/affine_body_state_accessor_feature.cpp:6-25); the overrider's
        // do_copy_to only fills the ones that exist, so both are created here explicitly.
        state.instances().create<Matrix4x4>(builtin::transform);
        state.instances().create<Matrix4x4>(builtin::velocity);
        feature->copy_to(state);

        auto transform = state.instances().find<Matrix4x4>(builtin::transform);
        auto velocity  = state.instances().find<Matrix4x4>(builtin::velocity);
        if(!transform || !velocity)
            throw std::runtime_error{"AffineBodyStateAccessorFeature::copy_to did not fill transform/velocity"};

        std::vector<double> transforms;
        transforms.reserve(body_count * 16);
        for(const Matrix4x4& m : transform->view())
            push_mat4(transforms, m);
        blocks.push_back(ArrayBlock{"transforms", -1, 16, std::move(transforms)});

        std::vector<double> velocities;
        velocities.reserve(body_count * 16);
        for(const Matrix4x4& m : velocity->view())
            push_mat4(velocities, m);
        blocks.push_back(ArrayBlock{"velocities", -1, 16, std::move(velocities)});
        return true;
    }

    /// The two finite-element probes (same route, the FEM sibling of the above).
    bool collect_finite_element_state(core::World& world, std::vector<ArrayBlock>& blocks)
    {
        auto feature = world.features().find<core::FiniteElementStateAccessorFeature>();
        if(!feature)
            return false;
        const SizeT vertex_count = feature->vertex_count();
        if(vertex_count == 0)
            return false;

        auto state = feature->create_geometry(0, vertex_count);
        state.vertices().create<Vector3>(builtin::position);
        state.vertices().create<Vector3>(builtin::velocity);
        feature->copy_to(state);

        auto position = state.vertices().find<Vector3>(builtin::position);
        auto velocity = state.vertices().find<Vector3>(builtin::velocity);
        if(!position || !velocity)
            throw std::runtime_error{"FiniteElementStateAccessorFeature::copy_to did not fill position/velocity"};

        std::vector<double> positions;
        positions.reserve(vertex_count * 3);
        for(const Vector3& v : position->view())
            push_vec3(positions, v);
        blocks.push_back(ArrayBlock{"fe_positions", -1, 3, std::move(positions)});

        std::vector<double> velocities;
        velocities.reserve(vertex_count * 3);
        for(const Vector3& v : velocity->view())
            push_vec3(velocities, v);
        blocks.push_back(ArrayBlock{"fe_velocities", -1, 3, std::move(velocities)});
        return true;
    }
}  // namespace

std::vector<ArrayBlock> collect_probes(core::World& world)
{
    std::vector<ArrayBlock> blocks;

    auto visitor    = backend::WorldVisitor{world};
    auto geometries = visitor.scene().geometries();
    for(SizeT i = 0; i < geometries.size(); ++i)
    {
        auto* sc = geometries[i]->geometry().as<geometry::SimplicialComplex>();
        if(sc == nullptr)
            continue;
        collect_vertices(*sc, static_cast<int>(i), blocks);
    }

    collect_affine_body_state(world, blocks);
    collect_finite_element_state(world, blocks);
    return blocks;
}

Json frame_scalars(const core::Engine& engine)
{
    Json scalars = engine.frame_stats();
    if(!scalars.is_object())
        scalars = Json::object();
    return scalars;
}

bool all_finite(const std::vector<ArrayBlock>& blocks) noexcept
{
    for(const auto& block : blocks)
    {
        for(double value : block.data)
        {
            if(!std::isfinite(value))
                return false;
        }
    }
    return true;
}
}  // namespace uipc::parity
