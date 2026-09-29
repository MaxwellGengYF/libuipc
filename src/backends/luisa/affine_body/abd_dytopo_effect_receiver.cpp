#include <affine_body/abd_dytopo_effect_receiver.h>

/*************************************************************************************************
* T6.1b — ABDDyTopoEffectReceiver (port of
* src/backends/cuda/affine_body/abd_dytopo_effect_receiver.cu; namespace + the
* T3.1 view types are the only changes).
*
* `do_report` declares the receiver's range as the ABD *vertex* slice
* `[vertex_offset, vertex_offset + vertex_count)`; `do_receive` caches the
* classified effect the manager hands back.  Both are the reference's bodies,
* unchanged.
*************************************************************************************************/
namespace uipc::backend::luisa
{
REGISTER_SIM_SYSTEM(ABDDyTopoEffectReceiver);

void ABDDyTopoEffectReceiver::do_build(DyTopoEffectReceiver::BuildInfo& info)
{
    (void)info;
    m_impl.affine_body_vertex_reporter = &require<AffineBodyVertexReporter>();
}

void ABDDyTopoEffectReceiver::do_report(GlobalDyTopoEffectManager::ClassifyInfo& info)
{
    auto vertex_offset = m_impl.affine_body_vertex_reporter->vertex_offset();
    auto vertex_count  = m_impl.affine_body_vertex_reporter->vertex_count();
    auto v_begin       = vertex_offset;
    auto v_end         = vertex_offset + vertex_count;
    info.range({v_begin, v_end});
}

void ABDDyTopoEffectReceiver::Impl::receive(
    GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info)
{
    gradients = info.gradients();
    hessians  = info.hessians();
}

void ABDDyTopoEffectReceiver::do_receive(
    GlobalDyTopoEffectManager::ClassifiedDyTopoEffectInfo& info)
{
    m_impl.receive(info);
}
}  // namespace uipc::backend::luisa
