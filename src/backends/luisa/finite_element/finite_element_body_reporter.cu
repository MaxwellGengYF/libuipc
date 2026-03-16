#include <finite_element/finite_element_body_reporter.h>

namespace uipc::backend::luisa
{
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
    using namespace luisa::compute;

    // Fill coindices with iota (0, 1, 2, ...)
    auto coindices = info.coindices();
    auto count = coindices.size();

    Kernel1D iota_kernel = [&](BufferVar<IndexT> coindices_buffer) noexcept
    {
        auto i = dispatch_x();
        coindices_buffer.write(i, cast<IndexT>(i));
    };

    auto shader = sim_engine->device().compile(iota_kernel);
    sim_engine->stream() << shader(coindices.buffer_view()).dispatch(count);

    span<const IndexT> self_collision = finite_element_method->m_impl.h_body_self_collision;

    UIPC_ASSERT(self_collision.size() == info.self_collision().size(),
                "Size mismatch in self-collision data, info size: {}, self_collision size: {}",
                info.self_collision().size(),
                self_collision.size());

    info.self_collision().copy_from(self_collision.data());
}
}  // namespace uipc::backend::luisa
