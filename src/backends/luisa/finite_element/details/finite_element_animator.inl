namespace uipc::backend::luisa
{
template <typename ForEach, typename ViewGetter>
void FiniteElementAnimator::FilteredInfo::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                                   ViewGetter&&                    view_getter,
                                                   ForEach&&                       for_each_action) noexcept
{
    auto geo_infos = anim_geo_infos();
    // Note: The _for_each implementation will be provided by FiniteElementMethod
    // This is a placeholder that references the CUDA backend pattern
    // In the full implementation, this should call the appropriate for_each method
    // from FiniteElementMethod or a similar utility
}
}  // namespace uipc::backend::luisa
