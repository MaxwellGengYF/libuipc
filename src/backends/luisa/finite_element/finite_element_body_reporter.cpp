#include <finite_element/finite_element_body_reporter.h>
#include <luisa_device/linear/linear.h>
#include <frame_stream.h>

namespace uipc::backend::luisa
{
namespace
{
namespace lla_ = luisa_device::linalg;

/// `coindices(i) = i` (the reference's iota kernel).
auto& report_attributes_kernel()
{
    static auto& k = lla_::detail::cached_kernel<lcrt::Buffer<IndexT>, uint32_t>(
        [](lcrt::BufferVar<IndexT> coindices, lcrt::Var<uint32_t> n) noexcept
        {
            auto i = lcrt::dispatch_x();
            $if(i < n) { coindices->write(i, lcrt::cast<IndexT>(i)); };
        });
    return k;
}
}  // namespace

REGISTER_SIM_SYSTEM(FiniteElementBodyReporter);

void FiniteElementBodyReporter::do_build(BuildInfo& info)
{
    m_impl.finite_element_method = &require<FiniteElementMethod>();
}

void FiniteElementBodyReporter::do_init(InitInfo& info)
{
    // do nothing
}

void FiniteElementBodyReporter::do_report_count(BodyCountInfo& info)
{
    m_impl.report_count(info);
}

void FiniteElementBodyReporter::do_report_attributes(BodyAttributeInfo& info)
{
    m_impl.report_attributes(info);
}

void FiniteElementBodyReporter::Impl::report_count(BodyCountInfo& info)
{
    auto N = finite_element_method->m_impl.h_body_self_collision.size();
    info.count(N);
    info.changeable(false);
}

void FiniteElementBodyReporter::Impl::report_attributes(BodyAttributeInfo& info)
{
    auto& k = report_attributes_kernel();
    auto  n = static_cast<uint32_t>(info.coindices().size());
    if(n > 0)
    {
        frame_stream().record(k(info.coindices(), n).dispatch(n));
    }

    span<const IndexT> self_collision = finite_element_method->m_impl.h_body_self_collision;

    UIPC_ASSERT(self_collision.size() == info.self_collision().size(),
                "Size mismatch in self-collision data, info size: {}, self_collision size: {}",
                info.self_collision().size(),
                self_collision.size());

    info.self_collision().copy_from(self_collision);
}
}  // namespace uipc::backend::luisa
