#include <luisa/luisa-compute.h>
#include <uipc/backend/visitors/scene_visitor.h>
#include <uipc/builtin/constitution_uid_register.h>
#include <uipc/builtin/attribute_name.h>
#include <uipc/common/map.h>
#include <uipc/common/set.h>
#include <uipc/common/zip.h>
#include <uipc/builtin/constitution_type.h>

#include <affine_body/affine_body_constitution.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/affine_body_kinetic.h>
#include <affine_body/affine_body_manager.h>
#include <affine_body/inter_affine_body_constitution.h>
#include <affine_body/utils.h>

#include <affine_body/constitutions/joint_limit_penalty.h>
#include <affine_body/constitutions/sym/affine_body_revolute_joint_limit.inl>

namespace uipc::backend::luisa
{
class AffineBodyRevoluteJointLimit final : public InterAffineBodyConstitution
{
  public:
    using InterAffineBodyConstitution::InterAffineBodyConstitution;

    static constexpr std::string_view ConstitutionUID = "670";
    static constexpr std::string_view JointUID        = "18";

    static constexpr U64 ConstitutionUIDValue = 670;
    static constexpr U64 JointUIDValue        = 18;

    virtual void do_build(BuildInfo& info) override
    {
        InterAffineBodyConstitution::do_build(info);

        auto& joint_info = require_joint();
        UIPC_ASSERT(joint_info.joint_uid == JointUIDValue,
                    "JointUID mismatch, expected: {}, got: {}",
                    JointUIDValue,
                    joint_info.joint_uid);
    }

    class Impl
    {
      public:
        void do_build(AffineBodyRevoluteJointLimit* self)
        {
            auto& joint_info = self->require_joint();

            // Get joint attributes
            auto  joint_scene  = self->scene().joints();
            auto& joint_attributes = joint_info.joint_attributes;

            auto lower_attr = joint_scene.find<Float>(builtin::lower_limit);
            auto upper_attr = joint_scene.find<Float>(builtin::upper_limit);
            auto strength_attr = joint_scene.find<Float>(builtin::strength);

            if(!lower_attr || !upper_attr || !strength_attr)
            {
                return;
            }

            auto lower_view  = lower_attr->view();
            auto upper_view  = upper_attr->view();
            auto strength_view = strength_attr->view();

            auto joint_count = joint_info.body_ids.size();

            // Collect valid joints
            vector<Float> lowers;
            vector<Float> uppers;
            vector<Float> strengths;
            vector<Vector2i> body_ids;
            vector<Vector6> l_basis;
            vector<Vector6> r_basis;
            vector<Vector24> ref_qs;

            auto& body_manager = self->affine_body_dynamics().affine_body_manager();

            for(Size i = 0; i < joint_count; ++i)
            {
                auto j = joint_info.joint_indices[i];

                Float lower = lower_view[j];
                Float upper = upper_view[j];
                Float strength = strength_view[j];

                if(lower < upper && strength > 0.0)
                {
                    lowers.push_back(lower);
                    uppers.push_back(upper);
                    strengths.push_back(strength);
                    body_ids.push_back(joint_info.body_ids[i]);
                    l_basis.push_back(joint_info.l_basis[i]);
                    r_basis.push_back(joint_info.r_basis[i]);
                    ref_qs.push_back(joint_info.ref_qs[i]);
                }
            }

            if(body_ids.empty())
            {
                return;
            }

            // Create device buffers
            Size joint_limit_count = body_ids.size();

            m_body_ids.resize(joint_limit_count);
            m_l_basis.resize(joint_limit_count);
            m_r_basis.resize(joint_limit_count);
            m_ref_qs.resize(joint_limit_count);
            m_lowers.resize(joint_limit_count);
            m_uppers.resize(joint_limit_count);
            m_strengths.resize(joint_limit_count);

            m_body_ids.view().copy_from(body_ids.data());
            m_l_basis.view().copy_from(l_basis.data());
            m_r_basis.view().copy_from(r_basis.data());
            m_ref_qs.view().copy_from(ref_qs.data());
            m_lowers.view().copy_from(lowers.data());
            m_uppers.view().copy_from(uppers.data());
            m_strengths.view().copy_from(strengths.data());
        }

        void do_step(AffineBodyRevoluteJointLimit* self)
        {
            if(m_body_ids.empty())
                return;

            auto& manager = self->inter_affine_body_constitution_manager();

            // Get buffers from manager
            auto qs = manager.qs();
            auto q_prevs = manager.q_prevs();
            auto G12s = manager.G12s();
            auto H12x12s = manager.H12x12s();

            // Compute energy
            do_compute_energy(qs, q_prevs);

            // Compute gradient and Hessian
            do_compute_gradient_hessian(qs, q_prevs, G12s, H12x12s, false);
        }

        void do_compute_energy(BufferView<Vector12> qs, BufferView<Vector12> q_prevs)
        {
            using namespace luisa;
            using namespace luisa::compute;

            auto& device = Engine::current()->device();
            auto stream = Engine::current()->stream();

            auto kernel = device.compile<1>([
                body_ids = m_body_ids.view(),
                l_basis = m_l_basis.view(),
                r_basis = m_r_basis.view(),
                ref_qs = m_ref_qs.view(),
                lowers = m_lowers.view(),
                uppers = m_uppers.view(),
                strengths = m_strengths.view(),
                qs = qs,
                q_prevs = q_prevs] $mutable
            {
                set_block_size(256);
                auto I = $dispatch_id.x;
                $if(I < body_ids.size())
                {
                    Vector2i bid = body_ids[I];
                    Vector6 lb = l_basis[I];
                    Vector6 rb = r_basis[I];
                    Vector24 ref_q = ref_qs[I];

                    // Extract q values for body k and l
                    Vector12 qk, q_prevk, q_refk;
                    Vector12 ql, q_prevl, q_refl;

                    for(int i = 0; i < 12; ++i)
                    {
                        qk[i] = qs[bid[0]][i];
                        q_prevk[i] = q_prevs[bid[0]][i];
                        q_refk[i] = ref_q[i];

                        ql[i] = qs[bid[1]][i];
                        q_prevl[i] = q_prevs[bid[1]][i];
                        q_refl[i] = ref_q[i + 12];
                    }

                    // Compute angle delta
                    Float theta_prev = 0.0f;
                    Float delta = 0.0f;

                    sym::affine_body_revolute_joint_limit::DeltaTheta(theta_prev, lb, q_prevk, q_refk, rb, q_prevl, q_refl);
                    sym::affine_body_revolute_joint_limit::DeltaTheta(delta, lb, qk, q_prevk, rb, ql, q_prevl);

                    Float x = theta_prev + delta;
                    Float E = uipc::backend::luisa::joint_limit::eval_penalty_energy<Float>(x, lowers[I], uppers[I], strengths[I]);

                    // Energy contribution is handled by the manager
                };
            });

            stream << kernel(body_ids.size()).dispatch();
        }

        void do_compute_gradient_hessian(BufferView<Vector12> qs,
                                          BufferView<Vector12> q_prevs,
                                          BufferView<Vector12> G12s,
                                          BufferView<Matrix12x12> H12x12s,
                                          Bool gradient_only)
        {
            using namespace luisa;
            using namespace luisa::compute;

            auto& device = Engine::current()->device();
            auto stream = Engine::current()->stream();

            auto kernel = device.compile<1>([
                body_ids = m_body_ids.view(),
                l_basis = m_l_basis.view(),
                r_basis = m_r_basis.view(),
                ref_qs = m_ref_qs.view(),
                lowers = m_lowers.view(),
                uppers = m_uppers.view(),
                strengths = m_strengths.view(),
                qs = qs,
                q_prevs = q_prevs,
                G12s = G12s,
                H12x12s = H12x12s,
                gradient_only = gradient_only] $mutable
            {
                set_block_size(256);
                auto I = $dispatch_id.x;
                $if(I < body_ids.size())
                {
                    Vector2i bid = body_ids[I];
                    Vector6 lb = l_basis[I];
                    Vector6 rb = r_basis[I];
                    Vector24 ref_q = ref_qs[I];

                    // Extract q values
                    Vector12 qk, q_prevk, q_refk;
                    Vector12 ql, q_prevl, q_refl;

                    for(int i = 0; i < 12; ++i)
                    {
                        qk[i] = qs[bid[0]][i];
                        q_prevk[i] = q_prevs[bid[0]][i];
                        q_refk[i] = ref_q[i];

                        ql[i] = qs[bid[1]][i];
                        q_prevl[i] = q_prevs[bid[1]][i];
                        q_refl[i] = ref_q[i + 12];
                    }

                    // Compute angle delta
                    Float theta_prev = 0.0f;
                    Float delta = 0.0f;

                    sym::affine_body_revolute_joint_limit::DeltaTheta(theta_prev, lb, q_prevk, q_refk, rb, q_prevl, q_refl);
                    sym::affine_body_revolute_joint_limit::DeltaTheta(delta, lb, qk, q_prevk, rb, ql, q_prevl);

                    Float x = theta_prev + delta;

                    // Evaluate penalty derivatives
                    Float dE_dx = 0.0f;
                    Float d2E_dx2 = 0.0f;
                    uipc::backend::luisa::joint_limit::eval_penalty_derivatives<Float>(x, lowers[I], uppers[I], strengths[I], dE_dx, d2E_dx2);

                    // Compute dDeltaTheta/dQ
                    Vector24 dDeltaTheta_dQ;
                    sym::affine_body_revolute_joint_limit::dDeltaTheta_dQ(dDeltaTheta_dQ, lb, qk, q_prevk, rb, ql, q_prevl);

                    // Compute gradient: G = dE/dx * dDeltaTheta/dQ
                    Vector24 G;
                    for(int i = 0; i < 24; ++i)
                    {
                        G[i] = dE_dx * dDeltaTheta_dQ[i];
                    }

                    // Split gradient into G12s for body k and l
                    for(int i = 0; i < 12; ++i)
                    {
                        G12s[bid[0]][i] += G[i];
                        G12s[bid[1]][i] += G[i + 12];
                    }

                    // Compute Hessian if needed
                    $if(!gradient_only)
                    {
                        // H = d2E/dx2 * (dDeltaTheta/dQ)^T * (dDeltaTheta/dQ) + dE/dx * d2DeltaTheta/dQ2

                        // First term: d2E/dx2 * outer product
                        Matrix24x24 H;
                        for(int i = 0; i < 24; ++i)
                        {
                            for(int j = 0; j < 24; ++j)
                            {
                                H(i, j) = d2E_dx2 * dDeltaTheta_dQ[i] * dDeltaTheta_dQ[j];
                            }
                        }

                        // Second term: dE/dx * d2DeltaTheta/dQ2 (only if dE_dx != 0)
                        $if(dE_dx != 0.0f)
                        {
                            // Compute F = J_r * Q and F_prev = J_r * Q_prev
                            Vector12 F;
                            Vector12 F_prev;
                            sym::affine_body_revolute_joint_limit::F(F, lb, qk, rb, ql);
                            sym::affine_body_revolute_joint_limit::F(F_prev, lb, q_prevk, rb, q_prevl);

                            // Compute d2DeltaTheta/dF2
                            Matrix12x12 d2DeltaTheta_dF2;
                            sym::affine_body_revolute_joint_limit::ddDeltaTheta_ddF(d2DeltaTheta_dF2, F, F_prev);

                            // Compute correction term: J_r^T * d2DeltaTheta/dF2 * J_r
                            Matrix24x24 correction;
                            sym::affine_body_revolute_joint_limit::JT_H_J(correction, d2DeltaTheta_dF2, lb, rb, lb, rb);

                            for(int i = 0; i < 24; ++i)
                            {
                                for(int j = 0; j < 24; ++j)
                                {
                                    H(i, j) += dE_dx * correction(i, j);
                                }
                            }
                        };

                        // Add to H12x12s for body k and l
                        // Block (0,0): body k with itself
                        for(int i = 0; i < 12; ++i)
                        {
                            for(int j = 0; j < 12; ++j)
                            {
                                H12x12s[bid[0]](i, j) += H(i, j);
                            }
                        }

                        // Block (0,1): body k with body l
                        for(int i = 0; i < 12; ++i)
                        {
                            for(int j = 0; j < 12; ++j)
                            {
                                H12x12s[bid[1]](i, j) += H(i, j + 12);
                            }
                        }

                        // Block (1,1): body l with itself
                        for(int i = 0; i < 12; ++i)
                        {
                            for(int j = 0; j < 12; ++j)
                            {
                                H12x12s[bid[1]](i, j) += H(i + 12, j + 12);
                            }
                        }
                    };
                };
            });

            stream << kernel(body_ids.size()).dispatch();
        }

      private:
        Buffer<Vector2i> m_body_ids;
        Buffer<Vector6> m_l_basis;
        Buffer<Vector6> m_r_basis;
        Buffer<Vector24> m_ref_qs;
        Buffer<Float> m_lowers;
        Buffer<Float> m_uppers;
        Buffer<Float> m_strengths;
    };

    Impl m_impl;

    virtual void do_build() override
    {
        m_impl.do_build(this);
    }

    virtual void do_step() override
    {
        m_impl.do_step(this);
    }
};

REGISTER_SIM_SYSTEM(AffineBodyRevoluteJointLimit);
}  // namespace uipc::backend::luisa
