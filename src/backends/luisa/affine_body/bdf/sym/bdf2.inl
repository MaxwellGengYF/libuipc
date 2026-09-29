/********************************************************************
 * @file affine_body/bdf/sym/bdf2.inl
 * @brief T6.2 — the BDF2 affine-body helper functions (reference:
 * src/backends/cuda/affine_body/bdf/sym/bdf2.inl).
 *
 * Included *inside namespace uipc::backend::luisa::sym::abd_bdf2* by
 * `bdf2_function.h` — no includes allowed here.
 *
 * TRANSCRIPTION: like the FEM twin (`finite_element/bdf/sym/bdf2.inl`, T5.3) this
 * file is **hand-written in the reference** (44 lines, not SymEigen.py output), so
 * it is transcribed by **hand** — there is no generator for it.  The transcription
 * is literal: the same temporaries (`c_qn`, `c_qn_1`, `c_v`, `c_g`), the same
 * statement order and the same association order as the reference.  The only change
 * is the spelling of the vector type, following the T5.2/T5.3 precedent exactly:
 * `Eigen::Vector<T, 12>` -> `lps::LVec<T, 12>`
 * `T` is `double` on the host and a DSL scalar (`lcrt::Var<double>`) inside a
 * `compile1` kernel, so the same three functions serve both sides (cookbook §7.4).
 * The `__host__ __device__` qualifiers have no counterpart in the port (kernels are
 * `device.compile<N>` lambdas) and are dropped; `inline` takes their place for the
 * header-only definitions.
 *
 * The three formulas, written out (h = dt, all vectors are 12-component, and the
 * *3-dof* sibling of every one of them is the FEM `.inl`'s — the formulas are
 * literally the same, only the container width differs):
 * compute_q_tilde(R, q_n, q_n_1, q_v_n, q_v_n_1, h)
 *   R = 4/3 q_n - 1/3 q_{n-1} + 2/9 h (4 q_v_n - q_v_{n-1})
 * compute_q_tilde(R, q_n, q_n_1, q_v_n, q_v_n_1, g, h)
 *   R = 4/9 h^2 g + 4/3 q_n - 1/3 q_{n-1} + 2/9 h (4 q_v_n - q_v_{n-1})
 * compute_qv(R, q, q_n, q_n_1, h)
 *   R = 1/(2h) (3 q - 4 q_n + q_{n-1})
 * The host-oracle case of `luisa_abd_dynamics_test.cpp`
 * (`abd_dynamics_bdf2_host_oracle`) asserts these against hand-derived values and
 * expects ~1e-16 (they are exact rational multiples of the inputs).
 ********************************************************************/
// BDF2 affine body helper functions.

template <typename T>
inline void compute_q_tilde(lps::LVec<T, 12>&       R,
                            const lps::LVec<T, 12>& q_n,
                            const lps::LVec<T, 12>& q_n_1,
                            const lps::LVec<T, 12>& qv_n,
                            const lps::LVec<T, 12>& qv_n_1,
                            const T&                h)
{
    const T c_qn   = static_cast<T>(4.0 / 3.0);
    const T c_qn_1 = static_cast<T>(1.0 / 3.0);
    const T c_v    = static_cast<T>(2.0 / 9.0) * h;

    R = c_qn * q_n - c_qn_1 * q_n_1 + c_v * (static_cast<T>(4.0) * qv_n - qv_n_1);
}

template <typename T>
inline void compute_q_tilde(lps::LVec<T, 12>&       R,
                            const lps::LVec<T, 12>& q_n,
                            const lps::LVec<T, 12>& q_n_1,
                            const lps::LVec<T, 12>& qv_n,
                            const lps::LVec<T, 12>& qv_n_1,
                            const lps::LVec<T, 12>& g,
                            const T&                h)
{
    const T c_qn   = static_cast<T>(4.0 / 3.0);
    const T c_qn_1 = static_cast<T>(1.0 / 3.0);
    const T c_v    = static_cast<T>(2.0 / 9.0) * h;
    const T c_g    = static_cast<T>(4.0 / 9.0) * h * h;

    R = c_g * g + c_qn * q_n - c_qn_1 * q_n_1 + c_v * (static_cast<T>(4.0) * qv_n - qv_n_1);
}

template <typename T>
inline void compute_qv(lps::LVec<T, 12>&       R,
                       const lps::LVec<T, 12>& q,
                       const lps::LVec<T, 12>& q_n,
                       const lps::LVec<T, 12>& q_n_1,
                       const T&                h)
{
    const T c = static_cast<T>(0.5) / h;
    R = c * (static_cast<T>(3.0) * q - static_cast<T>(4.0) * q_n + q_n_1);
}
