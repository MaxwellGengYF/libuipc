#include <dytopo_effect_system/dytopo_classify_info.h>
#include <uipc/common/log.h>

namespace uipc::backend::luisa
{
void DyTopoClassifyInfo::range(const luisa::int2& LRange, const luisa::int2& RRange)
{
    m_type             = Type::Range;
    m_hessian_i_range  = LRange;
    m_hessian_j_range  = RRange;
    m_gradient_i_range = luisa::int2{0, 0};
}

void DyTopoClassifyInfo::range(const luisa::int2& Range)
{
    m_type             = Type::Range;
    m_gradient_i_range = Range;
    m_hessian_i_range  = Range;
    m_hessian_j_range  = Range;
}

bool DyTopoClassifyInfo::is_empty() const
{
    return m_hessian_i_range.x == m_hessian_i_range.y
           || m_hessian_j_range.x == m_hessian_j_range.y;
}

bool DyTopoClassifyInfo::is_diag() const
{
    return m_gradient_i_range.x != m_gradient_i_range.y;
}

void DyTopoClassifyInfo::sanity_check()
{
    if(is_diag())
    {
        UIPC_ASSERT(m_gradient_i_range.x <= m_gradient_i_range.y,
                    "Diagonal Contact Gradient Range is invalid");

        UIPC_ASSERT(m_hessian_i_range.x == m_hessian_j_range.x
                        && m_hessian_i_range.y == m_hessian_j_range.y,
                    "Diagonal Contact Hessian must have the same i_range and j_range");
    }
    else
    {
        UIPC_ASSERT(m_gradient_i_range.x == m_gradient_i_range.y,
                    "Off-Diagonal Contact must not have Gradient Part");
    }

    UIPC_ASSERT(m_hessian_i_range.x <= m_hessian_i_range.y,
                "Contact Hessian Range-i is invalid");
    UIPC_ASSERT(m_hessian_j_range.x <= m_hessian_j_range.y,
                "Contact Hessian Range-j is invalid");
}
}  // namespace uipc::backend::luisa
