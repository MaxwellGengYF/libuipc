/********************************************************************
 * @file affine_body/details/affine_body_dynamics.inl
 * @brief T6.1a — the template members of AffineBodyDynamics
 * (port of src/backends/cuda/affine_body/details/affine_body_dynamics.inl).
 *
 * Included at the end of affine_body/affine_body_dynamics.h.  The bodies are the
 * reference's, with the two mechanical mappings:
 *   `cuda_tool::BufferView<T>::view(offset, count)` -> `LDeviceBuffer::view(...)`
 *   (an LBufferView *is* a windowed handle, so the window arithmetic is LC's own,
 *    exactly like T5.4b's copy fallback);
 *   `span{buffer}.subspan(offset, count)` stays a host span.
 *********************************************************************/
#include <uipc/common/zip.h>

namespace uipc::backend::luisa
{
template <typename ViewGetterF, typename ForEachF>
void AffineBodyDynamics::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                  ViewGetterF&&                   getter,
                                  ForEachF&&                      for_each)
{
    m_impl.for_each(  // NOTE: the reference writes `m_impl.template for_each(...)`;
                      // MSVC rejects the `template` disambiguator on a non-dependent
                      // member name (measured C2187 "unexpected 'template'" at
                      // instantiation), and it is not needed: `m_impl` has no
                      // dependent type, so overload resolution is unambiguous.
        geo_slots,
        std::forward<ViewGetterF>(getter),
        std::forward<ForEachF>(for_each));
}

template <typename ForEachGeometry>
void AffineBodyDynamics::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                  ForEachGeometry&&               for_every_geometry)
{
    m_impl.for_each(  // see the note above (`template` disambiguator dropped)
        geo_slots,
        std::forward<ForEachGeometry>(for_every_geometry));
}

template <typename ViewGetterF, typename ForEachF>
void AffineBodyDynamics::Impl::_for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                         span<const GeoInfo>             geo_infos,
                                         ViewGetterF&&                   getter,
                                         ForEachF&&                      for_each)
{
    ForEachInfo foreach_info;

    for(auto&& geo_info : geo_infos)
    {
        auto geo_index = geo_info.geo_slot_index;
        auto sc        = geo_slots[geo_index]->geometry().template as<geometry::SimplicialComplex>();
        UIPC_ASSERT(sc,
                    "Geometry({}) is not a simplicial complex, why can it happen?",
                    sc->type());

        auto attr_view = getter(*sc);
        foreach_info.m_local_index = 0;
        foreach_info.m_geo_info    = &geo_info;
        for(auto&& value : attr_view)
        {
            for_each(foreach_info, value);

            ++foreach_info.m_local_index;
            ++foreach_info.m_global_index;
        }
    }
}

template <typename ForEachGeometry>
void AffineBodyDynamics::Impl::_for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                         span<const GeoInfo>             geo_infos,
                                         ForEachGeometry&&               for_every_geometry)
{
    ForEachInfo foreach_info;

    for(auto&& geo_info : geo_infos)
    {
        auto geo_index = geo_info.geo_slot_index;
        auto sc        = geo_slots[geo_index]->geometry().template as<geometry::SimplicialComplex>();
        UIPC_ASSERT(sc,
                    "Geometry({}) is not a simplicial complex, why can it happen?",
                    sc->type());

        if constexpr(std::is_invocable_v<ForEachGeometry, const ForEachInfo&, geometry::SimplicialComplex&>)
        {
            foreach_info.m_geo_info = &geo_info;
            for_every_geometry(foreach_info, *sc);
            ++foreach_info.m_global_index;
        }
        else if constexpr(std::is_invocable_v<ForEachGeometry, geometry::SimplicialComplex&>)
        {
            for_every_geometry(*sc);
        }
        else
        {
            static_assert(always_false_v<ForEachGeometry>, "Invalid ForEachGeometry");
        }
    }
}

template <typename ForEachGeometry>
void AffineBodyDynamics::Impl::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                        ForEachGeometry&&               for_every_geometry)
{
    _for_each(geo_slots, geo_infos, std::forward<ForEachGeometry>(for_every_geometry));
}

template <typename ViewGetterF, typename ForEachF>
void AffineBodyDynamics::Impl::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                        ViewGetterF&&                   getter,
                                        ForEachF&&                      for_each)
{
    _for_each(geo_slots,
              geo_infos,
              std::forward<ViewGetterF>(getter),
              std::forward<ForEachF>(for_each));
}

template <typename T>
lud::LBufferView<T> AffineBodyDynamics::Impl::subview(lud::LDeviceBuffer<T>& buffer,
                                                      SizeT constitution_index) const noexcept
{
    auto& constitution_info = constitution_infos[constitution_index];
    return buffer.view(constitution_info.body_offset, constitution_info.body_count);
}

template <typename T>
span<T> AffineBodyDynamics::Impl::subview(vector<T>& buffer, SizeT constitution_index) const noexcept
{
    auto& constitution_info = constitution_infos[constitution_index];
    return span{buffer}.subspan(constitution_info.body_offset, constitution_info.body_count);
}

template <typename ViewGetterF, typename ForEachF>
void AffineBodyDynamics::FilteredInfo::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                                ViewGetterF&&                   getter,
                                                ForEachF&&                      for_each) const
{
    m_impl->_for_each(geo_slots,
                      geo_infos(),
                      std::forward<ViewGetterF>(getter),
                      std::forward<ForEachF>(for_each));
}

template <typename ForEachGeometry>
void AffineBodyDynamics::FilteredInfo::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                                ForEachGeometry&&               for_every_geometry) const
{
    m_impl->_for_each(geo_slots, geo_infos(), std::forward<ForEachGeometry>(for_every_geometry));
}
}  // namespace uipc::backend::luisa
