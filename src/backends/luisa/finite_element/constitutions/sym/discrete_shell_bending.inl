/********************************************************************
 * @file discrete_shell_bending.inl
 * @brief T5.2 (batch b) — the SymEigen.py-generated discrete-shell-bending energy
 *        and its theta derivatives (reference:
 *        src/backends/cuda/finite_element/constitutions/sym/
 *        discrete_shell_bending.inl).
 *
 * Included *inside namespace uipc::backend::luisa::sym::discrete_shell_bending* by
 * `discrete_shell_bending_function.h` — no includes allowed here.
 *
 * TRANSCRIPTION: the whole file is machine generated (SymEigen.py) and is
 * transcribed by `scratch/gen_fem_cons_b.py` — never by hand.  Only the spelling
 * of the operators changes (RULE M1, cookbook §7.2):
 *   `a * b`          -> `lps::detail::mul(a, b)` (scalar trampoline)
 *   `std::pow(x, 2)` -> `lps::pow(x, 2)` (`lps::pow` is `std::pow` on the host and
 *                      `lcrt::pow` on the DSL)
 * (same temporary names, same statement order, same association order).
 ********************************************************************/
// >>>GENERATED:dsb>>>
template <typename T>
inline void E(T& R, const T& kappa, const T& theta, const T& theta_bar, const T& L0, const T& h_bar)
{
    R = lps::detail::div(lps::detail::mul(lps::detail::mul(L0, kappa), lps::pow(lps::detail::sub(theta, theta_bar), 2.0)), h_bar);
}
template <typename T>
inline void dEdtheta(T& R, const T& kappa, const T& theta, const T& theta_bar, const T& L0, const T& h_bar)
{
    R = lps::detail::div(lps::detail::mul(lps::detail::mul(L0, kappa), lps::detail::sub(lps::detail::mul(2.0, theta), lps::detail::mul(2.0, theta_bar))), h_bar);
}
template <typename T>
inline void ddEddtheta(T& R, const T& kappa, const T& theta, const T& theta_bar, const T& L0, const T& h_bar)
{
    R = lps::detail::div(lps::detail::mul(lps::detail::mul(2.0, L0), kappa), h_bar);
}
// <<<GENERATED:dsb<<<
