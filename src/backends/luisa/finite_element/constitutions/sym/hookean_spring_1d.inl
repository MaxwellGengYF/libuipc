/********************************************************************
 * @file hookean_spring_1d.inl
 * @brief T5.2 (batch c) — the SymEigen.py-generated Hookean-spring energy,
 * gradient, Hessian and parameter-gradient (reference:
 * src/backends/cuda/finite_element/constitutions/sym/hookean_spring_1d.inl).
 *
 * Included *inside namespace uipc::backend::luisa::sym::hookean_spring_1d* by
 * `hookean_spring_1d_function.h` — no includes allowed here.
 *
 * TRANSCRIPTION: the whole file is machine generated (SymEigen.py) and is
 * transcribed by `scratch/gen_fem_cons_c.py` — never by hand.  Only the spelling of
 * the operators changes (RULE M1, cookbook §7.2):
 *   `a * b`                    -> `lps::detail::mul(a, b)` (scalar trampoline)
 *   `std::pow(x, 2)`           -> `lps::pow(x, 2)`
 *   `Eigen::Vector<T,6>`       -> `lps::LVec<T,6>`
 *   `Eigen::Matrix<T,6,6>`     -> `lps::LMat<T,6,6>`
 * (same temporary names, same statement order, same association order).
 *
 * The mathematical content is derived independently in the unit test: with
 * `L = |X1 - X0|`,
 *     E     = k (L - L0)^2 / (2 L0^2)          (the generated `E`)
 *     dE/dX = k (L - L0) / (L0^2 L) * (+e, -e),  e = X0 - X1
 * i.e. exactly the 1d Hookean spring law the root `_function.h` wraps.
 * `pGpk` is the parameter gradient `dE/dk` (unused by the constitution itself; it is
 * the diff-sim surface, transcribed for completeness like batch a's unused siblings).
 ********************************************************************/
// >>>GENERATED:hs1d>>>
template <typename T>
inline void E(T& R, const T& k, const lps::LVec<T,6>& X, const T& L0)
{
    R = lps::detail::div(lps::detail::mul(lps::detail::mul(lps::detail::div(1.0, 2.0), k), lps::pow(lps::detail::add(lps::detail::neg(L0), lps::pow(lps::detail::add(lps::detail::add(lps::pow(lps::detail::add(lps::detail::neg(X(0)), X(3)), 2.0), lps::pow(lps::detail::add(lps::detail::neg(X(1)), X(4)), 2.0)), lps::pow(lps::detail::add(lps::detail::neg(X(2)), X(5)), 2.0)), lps::detail::div(1.0, 2.0))), 2.0)), lps::pow(L0, 2.0));
}
template <typename T>
inline void dEdX(lps::LVec<T,6>& R, const T& k, const lps::LVec<T,6>& X, const T& L0)
{
    auto x0 = lps::detail::sub(X(0), X(3));
    auto x1 = lps::detail::neg(x0);
    auto x2 = lps::detail::sub(X(1), X(4));
    auto x3 = lps::detail::neg(x2);
    auto x4 = lps::detail::sub(X(2), X(5));
    auto x5 = lps::detail::neg(x4);
    auto x6 = lps::pow(lps::detail::add(lps::detail::add(lps::pow(x1, 2.0), lps::pow(x3, 2.0)), lps::pow(x5, 2.0)), lps::detail::div(1.0, 2.0));
    auto x7 = lps::detail::div(lps::detail::mul(k, lps::detail::add(lps::detail::neg(L0), x6)), lps::detail::mul(lps::pow(L0, 2.0), x6));
    R(0) = lps::detail::mul(x0, x7);
    R(1) = lps::detail::mul(x2, x7);
    R(2) = lps::detail::mul(x4, x7);
    R(3) = lps::detail::mul(x1, x7);
    R(4) = lps::detail::mul(x3, x7);
    R(5) = lps::detail::mul(x5, x7);
}
template <typename T>
inline void ddEddX(lps::LMat<T,6,6>& R, const T& k, const lps::LVec<T,6>& X, const T& L0)
{
    auto x0 = lps::detail::sub(X(0), X(3));
    auto x1 = lps::pow(x0, 2.0);
    auto x2 = lps::detail::neg(x0);
    auto x3 = lps::pow(x2, 2.0);
    auto x4 = lps::detail::sub(X(1), X(4));
    auto x5 = lps::detail::neg(x4);
    auto x6 = lps::pow(x5, 2.0);
    auto x7 = lps::detail::sub(X(2), X(5));
    auto x8 = lps::detail::neg(x7);
    auto x9 = lps::pow(x8, 2.0);
    auto x10 = lps::detail::add(lps::detail::add(x3, x6), x9);
    auto x11 = lps::detail::div(k, lps::pow(L0, 2.0));
    auto x12 = lps::detail::div(x11, x10);
    auto x13 = lps::pow(x10, lps::detail::div(1.0, 2.0));
    auto x14 = lps::detail::mul(x11, lps::detail::add(lps::detail::neg(L0), x13));
    auto x15 = lps::detail::div(x14, x13);
    auto x16 = lps::detail::div(x14, lps::pow(x10, lps::detail::div(3.0, 2.0)));
    auto x17 = lps::detail::mul(x0, x16);
    auto x18 = lps::detail::add(x15, lps::detail::mul(x17, x2));
    auto x19 = lps::detail::mul(x0, x12);
    auto x20 = lps::detail::mul(x19, x4);
    auto x21 = lps::detail::mul(x17, x5);
    auto x22 = lps::detail::mul(x19, x7);
    auto x23 = lps::detail::mul(x17, x8);
    auto x24 = lps::detail::neg(x15);
    auto x25 = lps::detail::add(lps::detail::mul(x19, x2), x24);
    auto x26 = lps::detail::mul(x19, x5);
    auto x27 = lps::detail::mul(x17, x4);
    auto x28 = lps::detail::mul(x19, x8);
    auto x29 = lps::detail::mul(x17, x7);
    auto x30 = lps::detail::mul(x16, x4);
    auto x31 = lps::detail::mul(x2, x30);
    auto x32 = lps::pow(x4, 2.0);
    auto x33 = lps::detail::add(x15, lps::detail::mul(x30, x5));
    auto x34 = lps::detail::mul(x12, x4);
    auto x35 = lps::detail::mul(x34, x7);
    auto x36 = lps::detail::mul(x30, x8);
    auto x37 = lps::detail::mul(x2, x34);
    auto x38 = lps::detail::add(x24, lps::detail::mul(x34, x5));
    auto x39 = lps::detail::mul(x34, x8);
    auto x40 = lps::detail::mul(x30, x7);
    auto x41 = lps::detail::mul(x16, x7);
    auto x42 = lps::detail::mul(x2, x41);
    auto x43 = lps::detail::mul(x41, x5);
    auto x44 = lps::pow(x7, 2.0);
    auto x45 = lps::detail::add(x15, lps::detail::mul(x41, x8));
    auto x46 = lps::detail::mul(x12, x7);
    auto x47 = lps::detail::mul(x2, x46);
    auto x48 = lps::detail::mul(x46, x5);
    auto x49 = lps::detail::add(x24, lps::detail::mul(x46, x8));
    auto x50 = lps::detail::mul(x16, x2);
    auto x51 = lps::detail::mul(x5, x50);
    auto x52 = lps::detail::mul(x50, x8);
    auto x53 = lps::detail::mul(x12, x2);
    auto x54 = lps::detail::mul(x5, x53);
    auto x55 = lps::detail::mul(x53, x8);
    auto x56 = lps::detail::mul(x5, x8);
    auto x57 = lps::detail::mul(x16, x56);
    auto x58 = lps::detail::mul(x12, x56);
    R(0, 0) = lps::detail::add(lps::detail::mul(x1, x12), x18);
    R(0, 1) = lps::detail::add(x20, x21);
    R(0, 2) = lps::detail::add(x22, x23);
    R(0, 3) = lps::detail::add(lps::detail::mul(x1, x16), x25);
    R(0, 4) = lps::detail::add(x26, x27);
    R(0, 5) = lps::detail::add(x28, x29);
    R(1, 0) = lps::detail::add(x20, x31);
    R(1, 1) = lps::detail::add(lps::detail::mul(x12, x32), x33);
    R(1, 2) = lps::detail::add(x35, x36);
    R(1, 3) = lps::detail::add(x27, x37);
    R(1, 4) = lps::detail::add(lps::detail::mul(x16, x32), x38);
    R(1, 5) = lps::detail::add(x39, x40);
    R(2, 0) = lps::detail::add(x22, x42);
    R(2, 1) = lps::detail::add(x35, x43);
    R(2, 2) = lps::detail::add(lps::detail::mul(x12, x44), x45);
    R(2, 3) = lps::detail::add(x29, x47);
    R(2, 4) = lps::detail::add(x40, x48);
    R(2, 5) = lps::detail::add(lps::detail::mul(x16, x44), x49);
    R(3, 0) = lps::detail::add(lps::detail::mul(x16, x3), x25);
    R(3, 1) = lps::detail::add(x37, x51);
    R(3, 2) = lps::detail::add(x47, x52);
    R(3, 3) = lps::detail::add(lps::detail::mul(x12, x3), x18);
    R(3, 4) = lps::detail::add(x31, x54);
    R(3, 5) = lps::detail::add(x42, x55);
    R(4, 0) = lps::detail::add(x26, x51);
    R(4, 1) = lps::detail::add(lps::detail::mul(x16, x6), x38);
    R(4, 2) = lps::detail::add(x48, x57);
    R(4, 3) = lps::detail::add(x21, x54);
    R(4, 4) = lps::detail::add(lps::detail::mul(x12, x6), x33);
    R(4, 5) = lps::detail::add(x43, x58);
    R(5, 0) = lps::detail::add(x28, x52);
    R(5, 1) = lps::detail::add(x39, x57);
    R(5, 2) = lps::detail::add(lps::detail::mul(x16, x9), x49);
    R(5, 3) = lps::detail::add(x23, x55);
    R(5, 4) = lps::detail::add(x36, x58);
    R(5, 5) = lps::detail::add(lps::detail::mul(x12, x9), x45);
}
template <typename T>
inline void pGpk(lps::LVec<T,6>& R, const T& k, const lps::LVec<T,6>& X, const T& L0)
{
    auto x0 = lps::detail::sub(X(0), X(3));
    auto x1 = lps::detail::neg(x0);
    auto x2 = lps::detail::sub(X(1), X(4));
    auto x3 = lps::detail::neg(x2);
    auto x4 = lps::detail::sub(X(2), X(5));
    auto x5 = lps::detail::neg(x4);
    auto x6 = lps::pow(lps::detail::add(lps::detail::add(lps::pow(x1, 2.0), lps::pow(x3, 2.0)), lps::pow(x5, 2.0)), lps::detail::div(1.0, 2.0));
    auto x7 = lps::detail::div(lps::detail::add(lps::detail::neg(L0), x6), lps::detail::mul(lps::pow(L0, 2.0), x6));
    R(0) = lps::detail::mul(x0, x7);
    R(1) = lps::detail::mul(x2, x7);
    R(2) = lps::detail::mul(x4, x7);
    R(3) = lps::detail::mul(x1, x7);
    R(4) = lps::detail::mul(x3, x7);
    R(5) = lps::detail::mul(x5, x7);
}
// <<<GENERATED:hs1d<<<
