#include <affine_body/affine_body_body_reporter.h>
#include <luisa_device/linear/linear.h>
#include <frame_stream.h>

/********************************************************************
 * T6.1a — AffineBodyBodyReporter (reference:
 * src/backends/cuda/affine_body/affine_body_body_reporter.cu).
 *
 * The reference's single kernel is an iota over `info.coindices()`
 * (`coindices(i) = i`); the self-collision flags are then a plain host->device
 * copy of `h_body_id_to_self_collision` (a host vector on both sides).
 *********************************************************************/
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

REGISTER_SIM_SYSTEM(AffineBodyBodyReporter);

void AffineBodyBodyReporter::do_build(BuildInfo& info)
{
    m_impl.affine_body_dynamics = &require<AffineBodyDynamics>();
}

void AffineBodyBodyReporter::do_init(InitInfo& info)
{
    // do nothing
}

void AffineBodyBodyReporter::do_report_count(BodyCountInfo& info)
{
    m_impl.report_count(info);
}

void AffineBodyBodyReporter::do_report_attributes(BodyAttributeInfo& info)
{
    m_impl.report_attributes(info);
}

void AffineBodyBodyReporter::Impl::report_count(BodyCountInfo& info)
{
    auto N = affine_body_dynamics->m_impl.body_count();
    info.count(N);
    info.changeable(false);
}

void AffineBodyBodyReporter::Impl::report_attributes(BodyAttributeInfo& info)
{
    auto& k = report_attributes_kernel();
    auto  n = static_cast<uint32_t>(info.coindices().size());
    if(n > 0)
    {
        frame_stream().record(k(info.coindices(), n).dispatch(n));
    }

    span<const IndexT> self_collision =
        affine_body_dynamics->m_impl.h_body_id_to_self_collision;

    UIPC_ASSERT(self_collision.size() == info.self_collision().size(),
                "Size mismatch in self-collision data, info size: {}, self_collision size: {}",
                info.self_collision().size(),
                self_collision.size());

    info.self_collision().copy_from(self_collision);
}
}  // namespace uipc::backend::luisa
