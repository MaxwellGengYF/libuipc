/********************************************************************
 * @file affine_body/details/inter_affine_body_constitution_manager.inl
 * @brief T6.1b — the out-of-line `for_each` templates of the inter-affine-body
 * constitution manager's `FilteredInfo` and the static `_for_each` they share
 * (reference: src/backends/cuda/affine_body/details/inter_affine_body_constitution_manager.inl).
 * Included at the end of the manager's header; identical to the reference apart
 * from the namespace.
 *********************************************************************/
namespace uipc::backend::luisa
{
template <typename ForEachGeometry>
void InterAffineBodyConstitutionManager::FilteredInfo::for_each(
    span<S<geometry::GeometrySlot>> geo_slots, ForEachGeometry&& for_every_geometry) const
{
    InterAffineBodyConstitutionManager::_for_each(
        geo_slots, this->inter_geo_infos(), std::forward<ForEachGeometry>(for_every_geometry));
}

template <typename ForEachGeometry>
void InterAffineBodyConstitutionManager::_for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                                   span<const InterGeoInfo>        geo_infos,
                                                   ForEachGeometry&& for_every_geometry)
{
    ForEachInfo foreach_info;

    for(auto&& geo_info : geo_infos)
    {
        auto  geo_index = geo_info.geo_slot_index;
        auto& geo       = geo_slots[geo_index]->geometry();

        if constexpr(std::is_invocable_v<ForEachGeometry, const ForEachInfo&, geometry::Geometry&>)
        {
            foreach_info.m_geo_info = &geo_info;
            for_every_geometry(foreach_info, geo);
            ++foreach_info.m_index;
        }
        else if constexpr(std::is_invocable_v<ForEachGeometry, geometry::Geometry&>)
        {
            for_every_geometry(geo);
        }
        else
        {
            static_assert(always_false_v<ForEachGeometry>, "Invalid ForEachGeometry");
        }
    }
}
}  // namespace uipc::backend::luisa
