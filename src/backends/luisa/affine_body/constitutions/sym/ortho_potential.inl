/********************************************************************
 * @file affine_body/constitutions/sym/ortho_potential.inl
 * @brief T6.3a (ABD) — the SymEigen.py-generated orthogonal-potential
 *        energy/gradient/Hessian of the ABD constitution (reference:
 *        src/backends/cuda/affine_body/constitutions/sym/ortho_potential.inl).
 *
 * TRANSCRIPTION: the whole file is machine generated (SymEigen.py output, "the only
 * sanctioned way to produce a generated body") and is transcribed by
 * `scratch/gen_abd_cons_a.py` — never by hand, and never re-spelled beyond the
 * operator spelling (cookbook §7.17 rule 34).  `scratch/gen_abd_cons_a.py` imports
 * the T5.2 transcriber `scratch/gen_fem_cons_a.py` (tokenizer/parser/emitter), so
 * the two families cannot drift apart; only its `REF_DIR`/`DST_DIR` and its region
 * list differ.
 *
 * Operator re-spelling (RULE M1, cookbook §7.2):
 *   `a * b` (scalars)      -> `lps::detail::mul(a, b)`
 *   `a + b` / `a - b`      -> `lps::detail::add` / `lps::detail::sub`
 *   `std::pow(x, 2)`       -> `lps::pow(x, 2)` (`lps::pow` is `std::pow` on the host
 *                             and `lcrt::pow` on the DSL)
 *   `Eigen::Vector<T,12>`  -> `lps::LVec<T,12>`; `Eigen::Matrix<T,9,9>` -> `lps::LMat<T,9,9>`
 *   integer literals       -> double literals (value-identical; keeps LC's
 *                             `[[deprecated]]` fp/integral default-confusion warning
 *                             helper out of the DSL instantiation, §7.16 rule 10)
 * Same statement order, same temporary names (`x0`, `x1`, ...), same association
 * order.
 *
 * WHAT THE GENERATED BODY SAYS (the reading the test's hand-derived oracle uses;
 * `q(3..11)` are the three rows of `A`, i.e. `A(i,j) = q(3 + 3i + j)`):
 *   with `C = AᵀA - I`,
 *       E  = kappa * ||AᵀA - I||_F^2   (the `2*(...)²` terms are the two off-diagonal
 *                                       entries, the `(|a_i|² - 1)²` terms the diagonal)
 *       dE/dA = 4 kappa A C            (each `R(i)` row of the generated gradient is
 *                                       `4 kappa * (A C)(row, ·)`)
 *   and `ddEddq` is the exact second derivative of that polynomial.  E/G/H are
 *   therefore finite everywhere (the reference applies no input guard) and the
 *   *unclamped* `ddEddq` is indefinite for `AᵀA < I` — which is why the reference's
 *   kernel clamps it with `make_spd` (see ortho_potential.h).
 ********************************************************************/
// >>>GENERATED:abd_ortho_potential>>>
template <typename T>
inline void E(T& R, const T& kappa, const lps::LVec<T,12>& q)
{
    R = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::mul(2.0, lps::pow(lps::detail::add(lps::detail::add(lps::detail::mul(q(10), q(4)), lps::detail::mul(q(11), q(5))), lps::detail::mul(q(3), q(9))), 2.0)), lps::detail::mul(2.0, lps::pow(lps::detail::add(lps::detail::add(lps::detail::mul(q(10), q(7)), lps::detail::mul(q(11), q(8))), lps::detail::mul(q(6), q(9))), 2.0))), lps::detail::mul(2.0, lps::pow(lps::detail::add(lps::detail::add(lps::detail::mul(q(3), q(6)), lps::detail::mul(q(4), q(7))), lps::detail::mul(q(5), q(8))), 2.0))), lps::pow(lps::detail::sub(lps::detail::add(lps::detail::add(lps::pow(q(10), 2.0), lps::pow(q(11), 2.0)), lps::pow(q(9), 2.0)), 1.0), 2.0)), lps::pow(lps::detail::sub(lps::detail::add(lps::detail::add(lps::pow(q(3), 2.0), lps::pow(q(4), 2.0)), lps::pow(q(5), 2.0)), 1.0), 2.0)), lps::pow(lps::detail::sub(lps::detail::add(lps::detail::add(lps::pow(q(6), 2.0), lps::pow(q(7), 2.0)), lps::pow(q(8), 2.0)), 1.0), 2.0)));
}
template <typename T>
inline void dEdq(lps::LVec<T,9>& R, const T& kappa, const lps::LVec<T,12>& q)
{
    auto x0 = lps::detail::add(lps::detail::add(lps::detail::mul(lps::detail::mul(4.0, q(3)), q(6)), lps::detail::mul(lps::detail::mul(4.0, q(4)), q(7))), lps::detail::mul(lps::detail::mul(4.0, q(5)), q(8)));
    auto x1 = lps::detail::add(lps::detail::add(lps::detail::mul(lps::detail::mul(4.0, q(10)), q(4)), lps::detail::mul(lps::detail::mul(4.0, q(11)), q(5))), lps::detail::mul(lps::detail::mul(4.0, q(3)), q(9)));
    auto x2 = lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::mul(4.0, lps::pow(q(3), 2.0)), lps::detail::mul(4.0, lps::pow(q(4), 2.0))), lps::detail::mul(4.0, lps::pow(q(5), 2.0))), 4.0);
    auto x3 = lps::detail::add(lps::detail::add(lps::detail::mul(lps::detail::mul(4.0, q(10)), q(7)), lps::detail::mul(lps::detail::mul(4.0, q(11)), q(8))), lps::detail::mul(lps::detail::mul(4.0, q(6)), q(9)));
    auto x4 = lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::mul(4.0, lps::pow(q(6), 2.0)), lps::detail::mul(4.0, lps::pow(q(7), 2.0))), lps::detail::mul(4.0, lps::pow(q(8), 2.0))), 4.0);
    auto x5 = lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::mul(4.0, lps::pow(q(10), 2.0)), lps::detail::mul(4.0, lps::pow(q(11), 2.0))), lps::detail::mul(4.0, lps::pow(q(9), 2.0))), 4.0);
    R(0) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(3), x2), lps::detail::mul(q(6), x0)), lps::detail::mul(q(9), x1)));
    R(1) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(10), x1), lps::detail::mul(q(4), x2)), lps::detail::mul(q(7), x0)));
    R(2) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(11), x1), lps::detail::mul(q(5), x2)), lps::detail::mul(q(8), x0)));
    R(3) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(3), x0), lps::detail::mul(q(6), x4)), lps::detail::mul(q(9), x3)));
    R(4) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(10), x3), lps::detail::mul(q(4), x0)), lps::detail::mul(q(7), x4)));
    R(5) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(11), x3), lps::detail::mul(q(5), x0)), lps::detail::mul(q(8), x4)));
    R(6) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(3), x1), lps::detail::mul(q(6), x3)), lps::detail::mul(q(9), x5)));
    R(7) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(10), x5), lps::detail::mul(q(4), x1)), lps::detail::mul(q(7), x3)));
    R(8) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(11), x5), lps::detail::mul(q(5), x1)), lps::detail::mul(q(8), x3)));
}
template <typename T>
inline void ddEddq(lps::LMat<T,9,9>& R, const T& kappa, const lps::LVec<T,12>& q)
{
    auto x0 = lps::pow(q(3), 2.0);
    auto x1 = lps::pow(q(4), 2.0);
    auto x2 = lps::detail::mul(4.0, x1);
    auto x3 = lps::pow(q(6), 2.0);
    auto x4 = lps::detail::mul(4.0, x3);
    auto x5 = lps::detail::add(x2, x4);
    auto x6 = lps::pow(q(9), 2.0);
    auto x7 = lps::detail::mul(4.0, x6);
    auto x8 = lps::pow(q(5), 2.0);
    auto x9 = lps::detail::sub(lps::detail::mul(4.0, x8), 4.0);
    auto x10 = lps::detail::add(x7, x9);
    auto x11 = lps::detail::mul(4.0, q(9));
    auto x12 = lps::detail::mul(q(10), x11);
    auto x13 = lps::detail::mul(8.0, q(3));
    auto x14 = lps::detail::mul(4.0, q(6));
    auto x15 = lps::detail::mul(q(7), x14);
    auto x16 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(4), x13), x12), x15));
    auto x17 = lps::detail::mul(q(11), x11);
    auto x18 = lps::detail::mul(q(8), x14);
    auto x19 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(5), x13), x17), x18));
    auto x20 = lps::detail::mul(4.0, q(4));
    auto x21 = lps::detail::mul(q(7), x20);
    auto x22 = lps::detail::mul(4.0, q(5));
    auto x23 = lps::detail::mul(q(8), x22);
    auto x24 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(6), x13), x21), x23));
    auto x25 = lps::detail::mul(kappa, x14);
    auto x26 = lps::detail::mul(q(4), x25);
    auto x27 = lps::detail::mul(q(5), x25);
    auto x28 = lps::detail::mul(q(10), x20);
    auto x29 = lps::detail::mul(q(11), x22);
    auto x30 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(9), x13), x28), x29));
    auto x31 = lps::detail::mul(kappa, x11);
    auto x32 = lps::detail::mul(q(4), x31);
    auto x33 = lps::detail::mul(q(5), x31);
    auto x34 = lps::pow(q(10), 2.0);
    auto x35 = lps::detail::mul(4.0, x34);
    auto x36 = lps::detail::mul(4.0, x0);
    auto x37 = lps::pow(q(7), 2.0);
    auto x38 = lps::detail::mul(4.0, x37);
    auto x39 = lps::detail::add(x36, x38);
    auto x40 = lps::detail::mul(4.0, q(10));
    auto x41 = lps::detail::mul(q(11), x40);
    auto x42 = lps::detail::mul(8.0, q(4));
    auto x43 = lps::detail::mul(4.0, q(8));
    auto x44 = lps::detail::mul(q(7), x43);
    auto x45 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(5), x42), x41), x44));
    auto x46 = lps::detail::mul(kappa, q(7));
    auto x47 = lps::detail::mul(4.0, q(3));
    auto x48 = lps::detail::mul(x46, x47);
    auto x49 = lps::detail::mul(q(3), x14);
    auto x50 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(7), x42), x23), x49));
    auto x51 = lps::detail::mul(x22, x46);
    auto x52 = lps::detail::mul(kappa, q(3));
    auto x53 = lps::detail::mul(x40, x52);
    auto x54 = lps::detail::mul(q(3), x11);
    auto x55 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(10), x42), x29), x54));
    auto x56 = lps::detail::mul(lps::detail::mul(kappa, q(10)), x22);
    auto x57 = lps::pow(q(8), 2.0);
    auto x58 = lps::detail::mul(4.0, x57);
    auto x59 = lps::detail::sub(x58, 4.0);
    auto x60 = lps::pow(q(11), 2.0);
    auto x61 = lps::detail::mul(4.0, x60);
    auto x62 = lps::detail::add(x36, x61);
    auto x63 = lps::detail::mul(x43, x52);
    auto x64 = lps::detail::mul(kappa, x20);
    auto x65 = lps::detail::mul(q(8), x64);
    auto x66 = lps::detail::mul(8.0, q(5));
    auto x67 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(8), x66), x21), x49));
    auto x68 = lps::detail::mul(lps::detail::mul(kappa, q(11)), x47);
    auto x69 = lps::detail::mul(q(11), x64);
    auto x70 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(11), x66), x28), x54));
    auto x71 = lps::detail::mul(q(3), x20);
    auto x72 = lps::detail::mul(8.0, q(6));
    auto x73 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(7), x72), x12), x71));
    auto x74 = lps::detail::mul(q(3), x22);
    auto x75 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(8), x72), x17), x74));
    auto x76 = lps::detail::mul(q(7), x40);
    auto x77 = lps::detail::mul(q(11), x43);
    auto x78 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(9), x72), x76), x77));
    auto x79 = lps::detail::mul(q(7), x31);
    auto x80 = lps::detail::mul(q(8), x31);
    auto x81 = lps::detail::mul(q(5), x20);
    auto x82 = lps::detail::mul(8.0, q(7));
    auto x83 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(8), x82), x41), x81));
    auto x84 = lps::detail::mul(q(10), x25);
    auto x85 = lps::detail::mul(q(6), x11);
    auto x86 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(10), x82), x77), x85));
    auto x87 = lps::detail::mul(lps::detail::mul(kappa, q(8)), x40);
    auto x88 = lps::detail::add(x38, x61);
    auto x89 = lps::detail::mul(q(11), x25);
    auto x90 = lps::detail::mul(lps::detail::mul(4.0, q(11)), x46);
    auto x91 = lps::detail::mul(8.0, q(11));
    auto x92 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(8), x91), x76), x85));
    auto x93 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(lps::detail::mul(8.0, q(10)), q(9)), x15), x71));
    auto x94 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(9), x91), x18), x74));
    auto x95 = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(q(10), x91), x44), x81));
    R(0, 0) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::mul(12.0, x0), x10), x5));
    R(0, 1) = x16;
    R(0, 2) = x19;
    R(0, 3) = x24;
    R(0, 4) = x26;
    R(0, 5) = x27;
    R(0, 6) = x30;
    R(0, 7) = x32;
    R(0, 8) = x33;
    R(1, 0) = x16;
    R(1, 1) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::mul(12.0, x1), x35), x39), x9));
    R(1, 2) = x45;
    R(1, 3) = x48;
    R(1, 4) = x50;
    R(1, 5) = x51;
    R(1, 6) = x53;
    R(1, 7) = x55;
    R(1, 8) = x56;
    R(2, 0) = x19;
    R(2, 1) = x45;
    R(2, 2) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::add(x2, x59), x62), lps::detail::mul(12.0, x8)));
    R(2, 3) = x63;
    R(2, 4) = x65;
    R(2, 5) = x67;
    R(2, 6) = x68;
    R(2, 7) = x69;
    R(2, 8) = x70;
    R(3, 0) = x24;
    R(3, 1) = x48;
    R(3, 2) = x63;
    R(3, 3) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::add(lps::detail::mul(12.0, x3), x39), x59), x7));
    R(3, 4) = x73;
    R(3, 5) = x75;
    R(3, 6) = x78;
    R(3, 7) = x79;
    R(3, 8) = x80;
    R(4, 0) = x26;
    R(4, 1) = x50;
    R(4, 2) = x65;
    R(4, 3) = x73;
    R(4, 4) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::add(x35, lps::detail::mul(12.0, x37)), x5), x59));
    R(4, 5) = x83;
    R(4, 6) = x84;
    R(4, 7) = x86;
    R(4, 8) = x87;
    R(5, 0) = x27;
    R(5, 1) = x51;
    R(5, 2) = x67;
    R(5, 3) = x75;
    R(5, 4) = x83;
    R(5, 5) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::add(x4, lps::detail::mul(12.0, x57)), x88), x9));
    R(5, 6) = x89;
    R(5, 7) = x90;
    R(5, 8) = x92;
    R(6, 0) = x30;
    R(6, 1) = x53;
    R(6, 2) = x68;
    R(6, 3) = x78;
    R(6, 4) = x84;
    R(6, 5) = x89;
    R(6, 6) = lps::detail::mul(kappa, lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::add(x35, x4), lps::detail::mul(12.0, x6)), x62), 4.0));
    R(6, 7) = x93;
    R(6, 8) = x94;
    R(7, 0) = x32;
    R(7, 1) = x55;
    R(7, 2) = x69;
    R(7, 3) = x79;
    R(7, 4) = x86;
    R(7, 5) = x90;
    R(7, 6) = x93;
    R(7, 7) = lps::detail::mul(kappa, lps::detail::sub(lps::detail::add(lps::detail::add(lps::detail::add(x2, lps::detail::mul(12.0, x34)), x7), x88), 4.0));
    R(7, 8) = x95;
    R(8, 0) = x33;
    R(8, 1) = x56;
    R(8, 2) = x70;
    R(8, 3) = x80;
    R(8, 4) = x87;
    R(8, 5) = x92;
    R(8, 6) = x94;
    R(8, 7) = x95;
    R(8, 8) = lps::detail::mul(kappa, lps::detail::add(lps::detail::add(lps::detail::add(x10, x35), x58), lps::detail::mul(12.0, x60)));
}
// <<<GENERATED:abd_ortho_potential<<<
