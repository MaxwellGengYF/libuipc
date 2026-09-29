/********************************************************************
 * @file bdf2.inl
 * @brief T5.3 — the BDF2 finite-element helper functions (reference:
 * src/backends/cuda/finite_element/bdf/sym/bdf2.inl).
 *
 * Included *inside namespace uipc::backend::luisa::sym::fem_bdf2* by
 * `bdf2_function.h` — no includes allowed here.
 *
 * TRANSCRIPTION: unlike the `sym/*.inl` siblings this file is **hand-written in
 * the reference** (44 lines, not SymEigen.py output), so it is transcribed by
 * **hand** (the T5.3 brief's §2.2) — there is no generator for it.  The
 * transcription is literal: the same temporaries, the same statement order and
 * the same association order as the reference.  The only change is the spelling
 * of the vector type, following the T5.2 precedent exactly (cookbook §7.17
 * rule 7's type mapping):
 *   `Eigen::Vector<T, 3>` -> `lps::LVec<T, 3>`
 * `T` is `double` on the host and a DSL scalar (`lcrt::Var<double>`) inside a
 * `compile1` kernel, so the same three functions serve both sides (cookbook
 * §7.4).  The `__host__ __device__` qualifiers have no counterpart in the port
 * (kernels are `device.compile<N>` lambdas, cookbook §3.1) and are dropped;
 * `inline` takes their place for the header-only definitions.
 *
 * The three formulas, written out (h = dt, all vectors are 3-component):
 *   compute_x_tilde(R, x_n, x_n_1, v_n, v_n_1, h)
 *       R = 4/3 x_n - 1/3 x_{n-1} + 2/9 h (4 v_n - v_{n-1})
 *   compute_x_tilde(R, x_n, x_n_1, v_n, v_n_1, g, h)
 *       R = 4/9 h^2 g + 4/3 x_n - 1/3 x_{n-1} + 2/9 h (4 v_n - v_{n-1})
 *   compute_v(R, x, x_n, x_n_1, h)
 *       R = 1/(2h) (3 x - 4 x_n + x_{n-1})
 * The host-oracle case of `luisa_fem_dynamics_test.cpp`
 * (`fem_dynamics_bdf2_host_oracle`) asserts these against hand-derived values
 * and expects ~1e-16 (they are exact rational multiples of the inputs).
 ********************************************************************/
// BDF2 finite element helper functions.

template <typename T>
inline void compute_x_tilde(lps::LVec<T, 3>&       R,
                            const lps::LVec<T, 3>& x_n,
                            const lps::LVec<T, 3>& x_n_1,
                            const lps::LVec<T, 3>& v_n,
                            const lps::LVec<T, 3>& v_n_1,
                            const T&               h)
{
    const T c_xn   = static_cast<T>(4.0 / 3.0);
    const T c_xn_1 = static_cast<T>(1.0 / 3.0);
    const T c_v    = static_cast<T>(2.0 / 9.0) * h;

    R = c_xn * x_n - c_xn_1 * x_n_1 + c_v * (static_cast<T>(4.0) * v_n - v_n_1);
}

template <typename T>
inline void compute_x_tilde(lps::LVec<T, 3>&       R,
                            const lps::LVec<T, 3>& x_n,
                            const lps::LVec<T, 3>& x_n_1,
                            const lps::LVec<T, 3>& v_n,
                            const lps::LVec<T, 3>& v_n_1,
                            const lps::LVec<T, 3>& g,
                            const T&               h)
{
    const T c_xn   = static_cast<T>(4.0 / 3.0);
    const T c_xn_1 = static_cast<T>(1.0 / 3.0);
    const T c_v    = static_cast<T>(2.0 / 9.0) * h;
    const T c_g    = static_cast<T>(4.0 / 9.0) * h * h;

    R = c_g * g + c_xn * x_n - c_xn_1 * x_n_1 + c_v * (static_cast<T>(4.0) * v_n - v_n_1);
}

template <typename T>
inline void compute_v(lps::LVec<T, 3>&       R,
                      const lps::LVec<T, 3>& x,
                      const lps::LVec<T, 3>& x_n,
                      const lps::LVec<T, 3>& x_n_1,
                      const T&               h)
{
    const T c = static_cast<T>(0.5) / h;
    R = c * (static_cast<T>(3.0) * x - static_cast<T>(4.0) * x_n + x_n_1);
}
