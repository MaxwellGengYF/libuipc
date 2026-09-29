/********************************************************************
 * @file finite_element_animator.inl
 * @brief T5.3 — the out-of-line `FilteredInfo::for_each` of the FEM animator
 * (reference: src/backends/cuda/finite_element/details/finite_element_animator.inl).
 * Included at the end of finite_element_animator.h; identical to the reference
 * apart from the namespace.
 *********************************************************************/
namespace uipc::backend::luisa
{
template <typename ForEach, typename ViewGetter>
void FiniteElementAnimator::FilteredInfo::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                                   ViewGetter&& view_getter,
                                                   ForEach&& for_each_action) noexcept
{
    auto geo_infos = anim_geo_infos();
    FiniteElementMethod::_for_each(geo_infos,
                                   geo_slots,
                                   std::forward<ViewGetter>(view_getter),
                                   std::forward<ForEach>(for_each_action));
}
}  // namespace uipc::backend::luisa
