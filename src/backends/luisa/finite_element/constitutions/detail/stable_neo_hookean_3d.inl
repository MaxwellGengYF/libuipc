/********************************************************************
 * @file stable_neo_hookean_3d.inl
 * @brief T5.2 (batch a) — the `sym::stable_neo_hookean_3d` energy/gradient/Hessian
 *        (reference: src/backends/cuda/finite_element/constitutions/detail/
 *        stable_neo_hookean_3d.inl).
 *
 * Included *inside namespace uipc::backend::luisa::sym::stable_neo_hookean_3d*
 * by `stable_neo_hookean_3d_function.h` -- no includes allowed here.
 *
 * `E` and `dEdVecF` are flat scalar algebra and are transcribed by
 * `scratch/gen_fem_cons_a.py` (verbatim statement order / association; only the
 * operator spelling and the Eigen->lps type names change).
 *
 * `ddEddVecF` mixes containers and scalars (`H1 = 2*I`, the `g1`/`gJ` block
 * assignments, the 3x3 `f*hat` comma initialisers, `g1*g1.transpose()`), so its
 * Eigen plumbing is hand written here with the arithmetic kept verbatim; the
 * `f0hat << ...` comma initialiser becomes one element assignment per entry in
 * Eigen's own (row-major) traversal order.
 ********************************************************************/

// >>>GENERATED:snh_3d>>>
template <typename T>
inline void E(T& R, const T& mu, const T& lambda, const lps::LMat<T, 3, 3>& F)
{
    auto J = F.determinant();
    auto Ic = F.squaredNorm();
    auto alpha = lps::detail::add(1.0, lps::detail::div(lps::detail::mul(0.75, mu), lambda));
    R = lps::detail::sub(lps::detail::add(lps::detail::mul(lps::detail::mul(lps::detail::mul(0.5, lambda), lps::detail::sub(J, alpha)), lps::detail::sub(J, alpha)), lps::detail::mul(lps::detail::mul(0.5, mu), lps::detail::sub(Ic, 3.0))), lps::detail::mul(lps::detail::mul(0.5, mu), lps::log(lps::detail::add(Ic, 1.0))));
}
template <typename T>
inline void dEdVecF(lps::LMat<T, 3, 3>& PEPF, const T& mu, const T& lambda, const lps::LMat<T, 3, 3>& F)
{
    auto J = F.determinant();
    auto Ic = F.squaredNorm();
    lps::LMat<T, 3, 3> pJpF;
    pJpF(0, 0) = lps::detail::sub(lps::detail::mul(F(1, 1), F(2, 2)), lps::detail::mul(F(1, 2), F(2, 1)));
    pJpF(0, 1) = lps::detail::sub(lps::detail::mul(F(1, 2), F(2, 0)), lps::detail::mul(F(1, 0), F(2, 2)));
    pJpF(0, 2) = lps::detail::sub(lps::detail::mul(F(1, 0), F(2, 1)), lps::detail::mul(F(1, 1), F(2, 0)));
    pJpF(1, 0) = lps::detail::sub(lps::detail::mul(F(2, 1), F(0, 2)), lps::detail::mul(F(2, 2), F(0, 1)));
    pJpF(1, 1) = lps::detail::sub(lps::detail::mul(F(2, 2), F(0, 0)), lps::detail::mul(F(2, 0), F(0, 2)));
    pJpF(1, 2) = lps::detail::sub(lps::detail::mul(F(2, 0), F(0, 1)), lps::detail::mul(F(2, 1), F(0, 0)));
    pJpF(2, 0) = lps::detail::sub(lps::detail::mul(F(0, 1), F(1, 2)), lps::detail::mul(F(1, 1), F(0, 2)));
    pJpF(2, 1) = lps::detail::sub(lps::detail::mul(F(0, 2), F(1, 0)), lps::detail::mul(F(0, 0), F(1, 2)));
    pJpF(2, 2) = lps::detail::sub(lps::detail::mul(F(0, 0), F(1, 1)), lps::detail::mul(F(0, 1), F(1, 0)));
    PEPF = ((lps::detail::mul(mu, lps::detail::sub(1.0, lps::detail::div(1.0, lps::detail::add(Ic, 1.0)))) * F) + (lps::detail::mul(lambda, lps::detail::sub(lps::detail::sub(J, 1.0), lps::detail::div(lps::detail::mul(0.75, mu), lambda))) * pJpF));
}
// <<<GENERATED:snh_3d<<<

/// Unit-volume Hessian `d^2 E / dF^2` of the `sym::stable_neo_hookean_3d` energy
/// (`detail/stable_neo_hookean_3d.inl`'s `E`), i.e. the *unprojected* Hessian.
///
/// `HJ` is `d cof(F) / dF` (flattened 9x9, the reference's `f*hat` blocks);
/// `g1 = 2F` and `gJ = cof(F)` are the two outer-product terms.
template <typename T>
inline void ddEddVecF(lps::LMat<T, 9, 9>&       R,
                      const T&                  mu,
                      const T&                  lambda,
                      const lps::LMat<T, 3, 3>& F)
{
    auto J  = F.determinant();
    auto Ic = F.squaredNorm();

    lps::LMat<T, 9, 9> H1 = lps::LMat<T, 9, 9>::Identity() * 2.0;
    lps::LVec<T, 9>    g1;
    g1.segment<3>(0) = F.col(0) * 2.0;
    g1.segment<3>(3) = F.col(1) * 2.0;
    g1.segment<3>(6) = F.col(2) * 2.0;
    lps::LVec<T, 9> gJ;
    gJ.segment<3>(0) = lps::cross(F.col(1), F.col(2));
    gJ.segment<3>(3) = lps::cross(F.col(2), F.col(0));
    gJ.segment<3>(6) = lps::cross(F.col(0), F.col(1));

    // f0hat << 0, -F(2,0), F(1,0), F(2,0), 0, -F(0,0), -F(1,0), F(0,0), 0;
    lps::LMat<T, 3, 3> f0hat;
    f0hat(0, 0) = 0.0;
    f0hat(0, 1) = -F(2, 0);
    f0hat(0, 2) = F(1, 0);
    f0hat(1, 0) = F(2, 0);
    f0hat(1, 1) = 0.0;
    f0hat(1, 2) = -F(0, 0);
    f0hat(2, 0) = -F(1, 0);
    f0hat(2, 1) = F(0, 0);
    f0hat(2, 2) = 0.0;
    // f1hat << 0, -F(2,1), F(1,1), F(2,1), 0, -F(0,1), -F(1,1), F(0,1), 0;
    lps::LMat<T, 3, 3> f1hat;
    f1hat(0, 0) = 0.0;
    f1hat(0, 1) = -F(2, 1);
    f1hat(0, 2) = F(1, 1);
    f1hat(1, 0) = F(2, 1);
    f1hat(1, 1) = 0.0;
    f1hat(1, 2) = -F(0, 1);
    f1hat(2, 0) = -F(1, 1);
    f1hat(2, 1) = F(0, 1);
    f1hat(2, 2) = 0.0;
    // f2hat << 0, -F(2,2), F(1,2), F(2,2), 0, -F(0,2), -F(1,2), F(0,2), 0;
    lps::LMat<T, 3, 3> f2hat;
    f2hat(0, 0) = 0.0;
    f2hat(0, 1) = -F(2, 2);
    f2hat(0, 2) = F(1, 2);
    f2hat(1, 0) = F(2, 2);
    f2hat(1, 1) = 0.0;
    f2hat(1, 2) = -F(0, 2);
    f2hat(2, 0) = -F(1, 2);
    f2hat(2, 1) = F(0, 2);
    f2hat(2, 2) = 0.0;

    lps::LMat<T, 9, 9> HJ;
    HJ.block<3, 3>(0, 0) = lps::LMat<T, 3, 3>::Zero();
    HJ.block<3, 3>(0, 3) = -f2hat;
    HJ.block<3, 3>(0, 6) = f1hat;
    HJ.block<3, 3>(3, 0) = f2hat;
    HJ.block<3, 3>(3, 3) = lps::LMat<T, 3, 3>::Zero();
    HJ.block<3, 3>(3, 6) = -f0hat;
    HJ.block<3, 3>(6, 0) = -f1hat;
    HJ.block<3, 3>(6, 3) = f0hat;
    HJ.block<3, 3>(6, 6) = lps::LMat<T, 3, 3>::Zero();

    // The reference's single expression, in its own association order: the scalar
    // sub-expressions go through the trampolines (RULE M1), the container products
    // keep the numerics layer's own operators.
    R = lps::detail::div(lps::detail::mul(Ic, mu),
                         lps::detail::mul(2.0, lps::detail::add(Ic, 1.0)))
            * H1
        + lps::detail::mul(lambda,
                           lps::detail::sub(lps::detail::sub(J, 1.0),
                                            lps::detail::div(lps::detail::mul(3.0, mu),
                                                             lps::detail::mul(4.0, lambda))))
              * HJ
        + lps::detail::div(mu,
                           lps::detail::mul(lps::detail::mul(2.0, lps::detail::add(Ic, 1.0)),
                                            lps::detail::add(Ic, 1.0)))
              * g1 * g1.transpose()
        + lambda * gJ * gJ.transpose();
}
