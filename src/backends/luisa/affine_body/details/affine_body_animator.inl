/********************************************************************
 * @file affine_body/details/affine_body_animator.inl
 * @brief T6.1b — the out-of-line `FilteredInfo::for_each` templates of the ABD
 * animator (reference: src/backends/cuda/affine_body/details/affine_body_animator.inl).
 * Included at the end of affine_body/affine_body_animator.h; identical to the
 * reference apart from the namespace.
 *********************************************************************/
namespace uipc::backend::luisa
{
template <typename ViewGetterF, typename ForEachF>
void AffineBodyAnimator::FilteredInfo::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                               ViewGetterF&&                   getter,
                                               ForEachF&&                      for_each)
{
    AffineBodyDynamics::Impl::_for_each(geo_slots,
                                        this->anim_geo_infos(),
                                        std::forward<ViewGetterF>(getter),
                                        std::forward<ForEachF>(for_each));
}

template <typename ForEachGeometry>
void AffineBodyAnimator::FilteredInfo::for_each(span<S<geometry::GeometrySlot>> geo_slots,
                                               ForEachGeometry&&               for_every_geometry)
{
    AffineBodyDynamics::Impl::_for_each(geo_slots,
                                        this->anim_geo_infos(),
                                        std::forward<ForEachGeometry>(for_every_geometry));
}
}  // namespace uipc::backend::luisa
