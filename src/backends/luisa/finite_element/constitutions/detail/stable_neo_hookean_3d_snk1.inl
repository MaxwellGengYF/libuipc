// Stiff-GIPC SNK1 constitution (ported from StiffGIPC/femEnergy.cu):
//   energy   = __cal_StabbleNHK_energy1_3D
//   gradient = __computePEPF_StableNHK3D1_double
//   Hessian  = __project_StabbleNHK_H_3D1_makePD (analytic SPD projection)
//
// Unit-volume energy:  Psi = 0.5*mu*(Ic - 3) + 0.5*lambda*(J - 1 - mu/lambda)^2
// with Ic = |F|_F^2, J = det(F). The Hessian uses Stiff's analytic
// eigensystem (twist/flip modes from the SVD of F + a 3x3 direct eigensolve)
// instead of a generic 9x9 EVD + clamp.
// Included inside namespace uipc::backend::luisa::snk1 by
// stable_neo_hookean_3d_function.h — no includes allowed here.
//
// T5.2 (batch a) port notes:
//  * `E` and `dEdF` are flat scalar algebra and are transcribed by
//    `scratch/gen_fem_cons_a.py` (verbatim statement order / association; only
//    the operator spelling and the Eigen->lps type names change).
//  * `build_twist_flip_eigenvectors` and `ddEddF_spd` mix containers and control
//    flow, so their Eigen plumbing is hand written with verbatim arithmetic.  The
//    documented, semantics-preserving rewrites are:
//      - `Eigen::Map<M3>(Q.data() + k) = M;` — the reference maps a *contiguous*
//        nine doubles of the 9x9 matrix as a column-major 3x3 block, i.e. a flat
//        copy of `M`'s own storage — becomes `set_block_flat(Q, k, M)`;
//      - `M3 A; A << a00, a01, ...;` — Eigen's comma initialiser fills row by row
//        (`M3` is column-major), so `A(i, j) = a_ij` one entry at a time;
//      - `if(eigenvalues(i) < 0.0) eigenvalues(i) = 0.0;` becomes a `lps::max`
//        *select* (RULE M2: a device value cannot drive host control flow; for a
//        host `double` the two forms agree, including the -0.0 case);
//      - `math::qr_svd` -> `lps::qr_svd`, `cuda_tool::eigen::evd` -> `lps::evd`,
//        scalar arithmetic -> the `lps::detail::{add,sub,mul,div}` trampolines
//        (RULE M1) and integer literals -> double literals (exact for these
//        values, and it keeps LC's fp/integral implicit-conversion helper out of
//        the DSL instantiation — cookbook §7.16 rule 10).

// >>>GENERATED:snk1>>>
template <typename T>
inline void E(T& R, const T& mu, const T& lambda, const lps::LMat<T, 3, 3>& F)
{
    T J = F.determinant();
    T Ic = F.squaredNorm();
    T Jm = lps::detail::sub(lps::detail::sub(J, 1.0), lps::detail::div(mu, lambda));
    R = lps::detail::mul(0.5, lps::detail::add(lps::detail::mul(mu, lps::detail::sub(Ic, 3.0)), lps::detail::mul(lps::detail::mul(lambda, Jm), Jm)));
}
template <typename T>
inline void dEdF(lps::LMat<T, 3, 3>& PEPF, const T& mu, const T& lambda, const lps::LMat<T, 3, 3>& F)
{
    T J = F.determinant();
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
    PEPF = ((mu * F) + (lps::detail::sub(lps::detail::mul(lambda, lps::detail::sub(J, 1.0)), mu) * pJpF));
}
// <<<GENERATED:snk1<<<

/// `Eigen::Map<M3>(Q.data() + k) = M;` — a flat, column-major nine-element write.
template <typename T>
inline void set_block_flat(lps::LMat<T, 9, 9>& Q, int k, const lps::LMat<T, 3, 3>& M)
{
    for(int i = 0; i < 9; ++i)
        Q.data[static_cast<std::size_t>(k + i)] = M.data[static_cast<std::size_t>(i)];
}

/// Orthonormal twist/flip eigenvectors of the SNK1 Hessian, written into `Q`'s
/// columns 0..5 (the reference's `A << ...` comma initialisers are expanded into
/// one element assignment per entry, in Eigen's own row-major traversal order).
template <typename T>
inline void build_twist_flip_eigenvectors(const lps::LMat<T, 3, 3>& U,
                                          const lps::LMat<T, 3, 3>& V,
                                          lps::LMat<T, 9, 9>&       Q)
{
    const T scale = lps::detail::div(lps::one_of<T>(), lps::sqrt(lps::make_scalar<T>(2)));

    const lps::LMat<T, 3, 3> sV = V * scale;

    // M3 A; A << sV(0,2)*U(0,1), sV(1,2)*U(0,1), sV(2,2)*U(0,1), ...  ==> A(i,j) = sV(j,2)*U(i,1)
    lps::LMat<T, 3, 3> A;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            A(i, j) = lps::detail::mul(sV(j, 2), U(i, 1));

    // M3 B; B << sV(0,1)*U(0,2), ...  ==> B(i,j) = sV(j,1)*U(i,2)
    lps::LMat<T, 3, 3> B;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            B(i, j) = lps::detail::mul(sV(j, 1), U(i, 2));

    // M3 C; C << sV(0,2)*U(0,0), ...  ==> C(i,j) = sV(j,2)*U(i,0)
    lps::LMat<T, 3, 3> C;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            C(i, j) = lps::detail::mul(sV(j, 2), U(i, 0));

    // M3 D; D << sV(0,0)*U(0,2), ...  ==> D(i,j) = sV(j,0)*U(i,2)
    lps::LMat<T, 3, 3> D;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            D(i, j) = lps::detail::mul(sV(j, 0), U(i, 2));

    // M3 E; E << sV(0,1)*U(0,0), ...  ==> E(i,j) = sV(j,1)*U(i,0)
    lps::LMat<T, 3, 3> E;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            E(i, j) = lps::detail::mul(sV(j, 1), U(i, 0));

    // M3 F_; F_ << sV(0,0)*U(0,1), ...  ==> F_(i,j) = sV(j,0)*U(i,1)
    lps::LMat<T, 3, 3> F_;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            F_(i, j) = lps::detail::mul(sV(j, 0), U(i, 1));

    // Twist eigenvectors
    set_block_flat(Q, 0, B - A);
    set_block_flat(Q, 9, D - C);
    set_block_flat(Q, 18, F_ - E);

    // Flip eigenvectors
    set_block_flat(Q, 27, A + B);
    set_block_flat(Q, 36, C + D);
    set_block_flat(Q, 45, E + F_);
}

// Analytic SPD-projected 9x9 energy Hessian (replaces generic EVD + clamp).
template <typename T>
inline void ddEddF_spd(lps::LMat<T, 9, 9>&       H,
                       const T&                      mu,
                       const T&                      lambda,
                       const lps::LMat<T, 3, 3>& F)
{
    const T J = F.determinant();

    lps::LMat<T, 3, 3> U, V;
    lps::LVec<T, 3>    S;
    lps::qr_svd(F, S, U, V);

    lps::LVec<T, 9>    eigenvalues;
    lps::LMat<T, 9, 9> Q;

    // Twist & flip eigenvalues: mu +/- sigma_i * (lambda*(J-1) - mu)
    const T evScale = lps::detail::sub(lps::detail::mul(lambda, lps::detail::sub(J, 1.0)), mu);
    eigenvalues.segment<3>(0) = S * evScale;
    eigenvalues.segment<3>(3) = -(S * evScale);
    eigenvalues.segment<6>(0).array() += mu;

    build_twist_flip_eigenvectors(U, V, Q);

    // Remaining three eigenpairs from the 3x3 stretch block
    {
        lps::LMat<T, 3, 3> A;
        const T            s0s0 = lps::detail::mul(S(0), S(0));
        const T            s1s1 = lps::detail::mul(S(1), S(1));
        const T            s2s2 = lps::detail::mul(S(2), S(2));
        A(0, 0)                 = lps::detail::add(mu, lps::detail::mul(lps::detail::mul(lambda, s1s1), s2s2));
        A(1, 1)                 = lps::detail::add(mu, lps::detail::mul(lps::detail::mul(lambda, s0s0), s2s2));
        A(2, 2)                 = lps::detail::add(mu, lps::detail::mul(lps::detail::mul(lambda, s0s0), s1s1));
        const T evScale2 = lps::detail::sub(lps::detail::mul(lambda, lps::detail::sub(lps::detail::mul(2.0, J), 1.0)), mu);
        A(0, 1)          = lps::detail::mul(evScale2, S(2));
        A(1, 0)          = A(0, 1);
        A(0, 2)          = lps::detail::mul(evScale2, S(1));
        A(2, 0)          = A(0, 2);
        A(1, 2)          = lps::detail::mul(evScale2, S(0));
        A(2, 1)          = A(1, 2);

        lps::LVec<T, 3>    block_values;
        lps::LMat<T, 3, 3> block_vectors;
        lps::evd(A, block_values, block_vectors);
        eigenvalues.segment<3>(6) = block_values;

        for(int k = 0; k < 3; ++k)
        {
            const lps::LVec<T, 3> vk = block_vectors.col(k);
            set_block_flat(Q, 54 + k * 9, U * vk.asDiagonal() * V.transpose());
        }
    }

    for(int i = 0; i < 9; i++)
    {
        // reference: `if(eigenvalues(i) < 0.0) eigenvalues(i) = 0.0;`
        // `(eigenvalues(i) > 0) ? eigenvalues(i) : 0` -- a select, never a branch
        // (RULE M2: a device value cannot drive host control flow).
        eigenvalues(i) = lps::max(eigenvalues(i), lps::zero_of<T>());
    }
    H = Q * eigenvalues.asDiagonal() * Q.transpose();
}
