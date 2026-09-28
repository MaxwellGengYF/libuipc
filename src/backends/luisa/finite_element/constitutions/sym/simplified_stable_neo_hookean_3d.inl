/********************************************************************
 * @file simplified_stable_neo_hookean_3d.inl
 * @brief T5.2 (batch a) — the SymEigen.py-generated simplified stable
 *        Neo-Hookean 3D energy/gradient/Hessian (reference:
 *        src/backends/cuda/finite_element/constitutions/sym/
 *        simplified_stable_neo_hookean_3d.inl).
 *
 * TRANSCRIPTION: the whole file is machine generated (SymEigen.py) and is
 * transcribed by `scratch/gen_fem_cons_a.py` -- never by hand.  Only the
 * spelling of the operators changes (RULE M1, cookbook §7.2):
 *   `a * b`      -> `lps::detail::mul(a, b)`   (scalar trampoline)
 *   `std::pow(x, 2)` -> `lps::pow(x, 2)`       (`lps::pow` is `std::pow` on the
 *                                               host and `lcrt::pow` on the DSL)
 *   `Eigen::Vector<T,9>` -> `lps::LVec<T,9>`   (same statement order, same
 *                                               temporary names, same
 *                                               association order)
 *
 * NOTE (reference parity): `stable_neo_hookean_3d_function.h` does *not* include
 * this file -- the reference's `StableNeoHookean3D` uses the Stiff-GIPC `snk1`
 * variant, and this SymEigen-generated sibling is dead code in the reference as
 * well.  It is ported (the batch's brief names `sym/*.inl`) and unit-tested, but
 * no constitution consumes it.
 ********************************************************************/
// >>>GENERATED:simplified_snh_3d>>>
template <typename T>
inline void E(T& R, const T& mu, const T& lambda, const lps::LVec<T,9>& VecF)
{
    auto x0 = lps::detail::sub(lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::sub(lps::detail::sub(lps::detail::mul(lps::detail::mul(VecF(0), VecF(4)), VecF(8)), lps::detail::mul(lps::detail::mul(VecF(0), VecF(7)), VecF(5))), lps::detail::mul(lps::detail::mul(VecF(3), VecF(1)), VecF(8))), lps::detail::mul(lps::detail::mul(VecF(3), VecF(7)), VecF(2))), lps::detail::mul(lps::detail::mul(VecF(6), VecF(1)), VecF(5))), lps::detail::mul(lps::detail::mul(VecF(6), VecF(4)), VecF(2))), 1.0);
    R = lps::detail::add(lps::detail::add(lps::detail::sub(lps::detail::mul(lps::detail::mul(lps::detail::div(1.0, 2.0), lambda), lps::pow(x0, 2.0)), lps::detail::mul(mu, x0)), lps::detail::mul(lps::detail::mul(lps::detail::div(1.0, 2.0), mu), lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::add(lps::pow(VecF(0), 2.0), lps::pow(VecF(3), 2.0)), lps::pow(VecF(6), 2.0)), lps::pow(VecF(1), 2.0)), lps::pow(VecF(4), 2.0)), lps::pow(VecF(7), 2.0)), lps::pow(VecF(2), 2.0)), lps::pow(VecF(5), 2.0)), lps::pow(VecF(8), 2.0)), 3.0))), lps::detail::div(lps::pow(mu, 2.0), lps::pow(lambda, 2.0)));
}
template <typename T>
inline void dEdVecF(lps::LVec<T,9>& R, const T& mu, const T& lambda, const lps::LVec<T,9>& VecF)
{
    auto x0 = lps::detail::mul(VecF(4), VecF(8));
    auto x1 = lps::detail::mul(VecF(7), VecF(5));
    auto x2 = lps::detail::mul(VecF(3), VecF(8));
    auto x3 = lps::detail::mul(VecF(6), VecF(4));
    auto x4 = lps::detail::mul(lps::detail::mul(lps::detail::div(1.0, 2.0), lambda), lps::detail::sub(lps::detail::sub(lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::sub(lps::detail::mul(lps::detail::mul(VecF(0), VecF(4)), VecF(8)), lps::detail::mul(VecF(0), x1)), lps::detail::mul(lps::detail::mul(VecF(3), VecF(7)), VecF(2))), lps::detail::mul(lps::detail::mul(VecF(6), VecF(1)), VecF(5))), lps::detail::mul(VecF(1), x2)), lps::detail::mul(VecF(2), x3)), 1.0));
    auto x5 = lps::detail::mul(VecF(3), VecF(7));
    auto x6 = lps::detail::mul(VecF(1), VecF(8));
    auto x7 = lps::detail::mul(VecF(0), VecF(8));
    auto x8 = lps::detail::mul(VecF(6), VecF(2));
    auto x9 = lps::detail::mul(VecF(0), VecF(7));
    auto x10 = lps::detail::mul(VecF(1), VecF(5));
    auto x11 = lps::detail::mul(VecF(4), VecF(2));
    auto x12 = lps::detail::mul(VecF(0), VecF(5));
    auto x13 = lps::detail::mul(VecF(0), VecF(4));
    auto x14 = lps::detail::mul(VecF(3), VecF(1));
    R(0) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(0), mu), lps::detail::mul(mu, lps::detail::sub(x0, x1))), lps::detail::mul(x4, lps::detail::sub(lps::detail::mul(2.0, x0), lps::detail::mul(2.0, x1))));
    R(1) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(1), mu), lps::detail::mul(mu, lps::detail::sub(lps::detail::mul(VecF(6), VecF(5)), x2))), lps::detail::mul(x4, lps::detail::sub(lps::detail::mul(lps::detail::mul(2.0, VecF(6)), VecF(5)), lps::detail::mul(2.0, x2))));
    R(2) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(2), mu), lps::detail::mul(mu, lps::detail::add(lps::detail::neg(x3), x5))), lps::detail::mul(x4, lps::detail::add(lps::detail::mul(lps::detail::neg(2.0), x3), lps::detail::mul(2.0, x5))));
    R(3) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(3), mu), lps::detail::mul(mu, lps::detail::sub(lps::detail::mul(VecF(7), VecF(2)), x6))), lps::detail::mul(x4, lps::detail::sub(lps::detail::mul(lps::detail::mul(2.0, VecF(7)), VecF(2)), lps::detail::mul(2.0, x6))));
    R(4) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(4), mu), lps::detail::mul(mu, lps::detail::sub(x7, x8))), lps::detail::mul(x4, lps::detail::sub(lps::detail::mul(2.0, x7), lps::detail::mul(2.0, x8))));
    R(5) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(5), mu), lps::detail::mul(mu, lps::detail::sub(lps::detail::mul(VecF(6), VecF(1)), x9))), lps::detail::mul(x4, lps::detail::sub(lps::detail::mul(lps::detail::mul(2.0, VecF(6)), VecF(1)), lps::detail::mul(2.0, x9))));
    R(6) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(6), mu), lps::detail::mul(mu, lps::detail::sub(x10, x11))), lps::detail::mul(x4, lps::detail::sub(lps::detail::mul(2.0, x10), lps::detail::mul(2.0, x11))));
    R(7) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(7), mu), lps::detail::mul(mu, lps::detail::sub(lps::detail::mul(VecF(3), VecF(2)), x12))), lps::detail::mul(x4, lps::detail::sub(lps::detail::mul(lps::detail::mul(2.0, VecF(3)), VecF(2)), lps::detail::mul(2.0, x12))));
    R(8) = lps::detail::add(lps::detail::sub(lps::detail::mul(VecF(8), mu), lps::detail::mul(mu, lps::detail::sub(x13, x14))), lps::detail::mul(x4, lps::detail::sub(lps::detail::mul(2.0, x13), lps::detail::mul(2.0, x14))));
}
template <typename T>
inline void ddEddVecF(lps::LMat<T,9,9>& R, const T& mu, const T& lambda, const lps::LVec<T,9>& VecF)
{
    auto x0 = lps::detail::mul(VecF(4), VecF(8));
    auto x1 = lps::detail::mul(VecF(7), VecF(5));
    auto x2 = lps::detail::sub(x0, x1);
    auto x3 = lps::detail::mul(lps::detail::div(1.0, 2.0), lambda);
    auto x4 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(2.0, x0), lps::detail::mul(2.0, x1)));
    auto x5 = lps::detail::mul(VecF(3), VecF(8));
    auto x6 = lps::detail::sub(lps::detail::mul(VecF(6), VecF(5)), x5);
    auto x7 = lps::detail::mul(VecF(3), VecF(7));
    auto x8 = lps::detail::mul(VecF(6), VecF(4));
    auto x9 = lps::detail::sub(x7, x8);
    auto x10 = lps::detail::mul(VecF(1), VecF(8));
    auto x11 = lps::detail::sub(lps::detail::mul(VecF(7), VecF(2)), x10);
    auto x12 = lps::detail::mul(VecF(0), VecF(8));
    auto x13 = lps::detail::mul(VecF(6), VecF(2));
    auto x14 = lps::detail::sub(x12, x13);
    auto x15 = lps::detail::mul(VecF(8), mu);
    auto x16 = lps::detail::mul(lambda, lps::detail::sub(lps::detail::sub(lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::sub(lps::detail::mul(lps::detail::mul(VecF(0), VecF(4)), VecF(8)), lps::detail::mul(VecF(0), x1)), lps::detail::mul(lps::detail::mul(VecF(3), VecF(7)), VecF(2))), lps::detail::mul(lps::detail::mul(VecF(6), VecF(1)), VecF(5))), lps::detail::mul(VecF(1), x5)), lps::detail::mul(VecF(2), x8)), 1.0));
    auto x17 = lps::detail::mul(VecF(8), x16);
    auto x18 = lps::detail::add(lps::detail::neg(x15), x17);
    auto x19 = lps::detail::mul(VecF(0), VecF(7));
    auto x20 = lps::detail::sub(lps::detail::mul(VecF(6), VecF(1)), x19);
    auto x21 = lps::detail::mul(VecF(7), mu);
    auto x22 = lps::detail::mul(VecF(7), x16);
    auto x23 = lps::detail::sub(x21, x22);
    auto x24 = lps::detail::mul(VecF(1), VecF(5));
    auto x25 = lps::detail::mul(VecF(4), VecF(2));
    auto x26 = lps::detail::sub(x24, x25);
    auto x27 = lps::detail::mul(VecF(0), VecF(5));
    auto x28 = lps::detail::sub(lps::detail::mul(VecF(3), VecF(2)), x27);
    auto x29 = lps::detail::mul(VecF(5), mu);
    auto x30 = lps::detail::mul(VecF(5), x16);
    auto x31 = lps::detail::sub(x29, x30);
    auto x32 = lps::detail::mul(VecF(0), VecF(4));
    auto x33 = lps::detail::mul(VecF(3), VecF(1));
    auto x34 = lps::detail::sub(x32, x33);
    auto x35 = lps::detail::mul(VecF(4), mu);
    auto x36 = lps::detail::mul(VecF(4), x16);
    auto x37 = lps::detail::add(lps::detail::neg(x35), x36);
    auto x38 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(lps::detail::mul(2.0, VecF(6)), VecF(5)), lps::detail::mul(2.0, x5)));
    auto x39 = lps::detail::sub(x15, x17);
    auto x40 = lps::detail::mul(VecF(6), mu);
    auto x41 = lps::detail::mul(VecF(6), x16);
    auto x42 = lps::detail::add(lps::detail::neg(x40), x41);
    auto x43 = lps::detail::add(lps::detail::neg(x29), x30);
    auto x44 = lps::detail::mul(VecF(3), mu);
    auto x45 = lps::detail::mul(VecF(3), x16);
    auto x46 = lps::detail::sub(x44, x45);
    auto x47 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(2.0, x7), lps::detail::mul(2.0, x8)));
    auto x48 = lps::detail::add(lps::detail::neg(x21), x22);
    auto x49 = lps::detail::sub(x40, x41);
    auto x50 = lps::detail::sub(x35, x36);
    auto x51 = lps::detail::add(lps::detail::neg(x44), x45);
    auto x52 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(lps::detail::mul(2.0, VecF(7)), VecF(2)), lps::detail::mul(2.0, x10)));
    auto x53 = lps::detail::mul(VecF(2), mu);
    auto x54 = lps::detail::mul(VecF(2), x16);
    auto x55 = lps::detail::add(lps::detail::neg(x53), x54);
    auto x56 = lps::detail::mul(VecF(1), mu);
    auto x57 = lps::detail::mul(VecF(1), x16);
    auto x58 = lps::detail::sub(x56, x57);
    auto x59 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(2.0, x12), lps::detail::mul(2.0, x13)));
    auto x60 = lps::detail::sub(x53, x54);
    auto x61 = lps::detail::mul(VecF(0), mu);
    auto x62 = lps::detail::mul(VecF(0), x16);
    auto x63 = lps::detail::add(lps::detail::neg(x61), x62);
    auto x64 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(lps::detail::mul(2.0, VecF(6)), VecF(1)), lps::detail::mul(2.0, x19)));
    auto x65 = lps::detail::add(lps::detail::neg(x56), x57);
    auto x66 = lps::detail::sub(x61, x62);
    auto x67 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(2.0, x24), lps::detail::mul(2.0, x25)));
    auto x68 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(lps::detail::mul(2.0, VecF(3)), VecF(2)), lps::detail::mul(2.0, x27)));
    auto x69 = lps::detail::mul(x3, lps::detail::sub(lps::detail::mul(2.0, x32), lps::detail::mul(2.0, x33)));
    R(0, 0) = lps::detail::add(mu, lps::detail::mul(x2, x4));
    R(0, 1) = lps::detail::mul(x4, x6);
    R(0, 2) = lps::detail::mul(x4, x9);
    R(0, 3) = lps::detail::mul(x11, x4);
    R(0, 4) = lps::detail::add(lps::detail::mul(x14, x4), x18);
    R(0, 5) = lps::detail::add(lps::detail::mul(x20, x4), x23);
    R(0, 6) = lps::detail::mul(x26, x4);
    R(0, 7) = lps::detail::add(lps::detail::mul(x28, x4), x31);
    R(0, 8) = lps::detail::add(lps::detail::mul(x34, x4), x37);
    R(1, 0) = lps::detail::mul(x2, x38);
    R(1, 1) = lps::detail::add(mu, lps::detail::mul(x38, x6));
    R(1, 2) = lps::detail::mul(x38, x9);
    R(1, 3) = lps::detail::add(lps::detail::mul(x11, x38), x39);
    R(1, 4) = lps::detail::mul(x14, x38);
    R(1, 5) = lps::detail::add(lps::detail::mul(x20, x38), x42);
    R(1, 6) = lps::detail::add(lps::detail::mul(x26, x38), x43);
    R(1, 7) = lps::detail::mul(x28, x38);
    R(1, 8) = lps::detail::add(lps::detail::mul(x34, x38), x46);
    R(2, 0) = lps::detail::mul(x2, x47);
    R(2, 1) = lps::detail::mul(x47, x6);
    R(2, 2) = lps::detail::add(mu, lps::detail::mul(x47, x9));
    R(2, 3) = lps::detail::add(lps::detail::mul(x11, x47), x48);
    R(2, 4) = lps::detail::add(lps::detail::mul(x14, x47), x49);
    R(2, 5) = lps::detail::mul(x20, x47);
    R(2, 6) = lps::detail::add(lps::detail::mul(x26, x47), x50);
    R(2, 7) = lps::detail::add(lps::detail::mul(x28, x47), x51);
    R(2, 8) = lps::detail::mul(x34, x47);
    R(3, 0) = lps::detail::mul(x2, x52);
    R(3, 1) = lps::detail::add(x39, lps::detail::mul(x52, x6));
    R(3, 2) = lps::detail::add(x48, lps::detail::mul(x52, x9));
    R(3, 3) = lps::detail::add(mu, lps::detail::mul(x11, x52));
    R(3, 4) = lps::detail::mul(x14, x52);
    R(3, 5) = lps::detail::mul(x20, x52);
    R(3, 6) = lps::detail::mul(x26, x52);
    R(3, 7) = lps::detail::add(lps::detail::mul(x28, x52), x55);
    R(3, 8) = lps::detail::add(lps::detail::mul(x34, x52), x58);
    R(4, 0) = lps::detail::add(x18, lps::detail::mul(x2, x59));
    R(4, 1) = lps::detail::mul(x59, x6);
    R(4, 2) = lps::detail::add(x49, lps::detail::mul(x59, x9));
    R(4, 3) = lps::detail::mul(x11, x59);
    R(4, 4) = lps::detail::add(mu, lps::detail::mul(x14, x59));
    R(4, 5) = lps::detail::mul(x20, x59);
    R(4, 6) = lps::detail::add(lps::detail::mul(x26, x59), x60);
    R(4, 7) = lps::detail::mul(x28, x59);
    R(4, 8) = lps::detail::add(lps::detail::mul(x34, x59), x63);
    R(5, 0) = lps::detail::add(lps::detail::mul(x2, x64), x23);
    R(5, 1) = lps::detail::add(x42, lps::detail::mul(x6, x64));
    R(5, 2) = lps::detail::mul(x64, x9);
    R(5, 3) = lps::detail::mul(x11, x64);
    R(5, 4) = lps::detail::mul(x14, x64);
    R(5, 5) = lps::detail::add(mu, lps::detail::mul(x20, x64));
    R(5, 6) = lps::detail::add(lps::detail::mul(x26, x64), x65);
    R(5, 7) = lps::detail::add(lps::detail::mul(x28, x64), x66);
    R(5, 8) = lps::detail::mul(x34, x64);
    R(6, 0) = lps::detail::mul(x2, x67);
    R(6, 1) = lps::detail::add(x43, lps::detail::mul(x6, x67));
    R(6, 2) = lps::detail::add(x50, lps::detail::mul(x67, x9));
    R(6, 3) = lps::detail::mul(x11, x67);
    R(6, 4) = lps::detail::add(lps::detail::mul(x14, x67), x60);
    R(6, 5) = lps::detail::add(lps::detail::mul(x20, x67), x65);
    R(6, 6) = lps::detail::add(mu, lps::detail::mul(x26, x67));
    R(6, 7) = lps::detail::mul(x28, x67);
    R(6, 8) = lps::detail::mul(x34, x67);
    R(7, 0) = lps::detail::add(lps::detail::mul(x2, x68), x31);
    R(7, 1) = lps::detail::mul(x6, x68);
    R(7, 2) = lps::detail::add(x51, lps::detail::mul(x68, x9));
    R(7, 3) = lps::detail::add(lps::detail::mul(x11, x68), x55);
    R(7, 4) = lps::detail::mul(x14, x68);
    R(7, 5) = lps::detail::add(lps::detail::mul(x20, x68), x66);
    R(7, 6) = lps::detail::mul(x26, x68);
    R(7, 7) = lps::detail::add(mu, lps::detail::mul(x28, x68));
    R(7, 8) = lps::detail::mul(x34, x68);
    R(8, 0) = lps::detail::add(lps::detail::mul(x2, x69), x37);
    R(8, 1) = lps::detail::add(x46, lps::detail::mul(x6, x69));
    R(8, 2) = lps::detail::mul(x69, x9);
    R(8, 3) = lps::detail::add(lps::detail::mul(x11, x69), x58);
    R(8, 4) = lps::detail::add(lps::detail::mul(x14, x69), x63);
    R(8, 5) = lps::detail::mul(x20, x69);
    R(8, 6) = lps::detail::mul(x26, x69);
    R(8, 7) = lps::detail::mul(x28, x69);
    R(8, 8) = lps::detail::add(mu, lps::detail::mul(x34, x69));
}
// <<<GENERATED:simplified_snh_3d<<<
