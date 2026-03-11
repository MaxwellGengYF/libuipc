

// Helper function to flatten a 3x3 matrix to a 9-element vector (column-major)
template <typename T>
inline luisa::Vector<T, 9> vec(const luisa::Matrix<T, 3>& mat)
{
    return luisa::Vector<T, 9>{
        mat[0][0], mat[0][1], mat[0][2],  // Column 0
        mat[1][0], mat[1][1], mat[1][2],  // Column 1
        mat[2][0], mat[2][1], mat[2][2]   // Column 2
    };
}

// Helper to create a 3x3 matrix from column vectors
template <typename T>
inline luisa::Matrix<T, 3> mat_from_cols(const luisa::Vector<T, 3>& c0,
                                          const luisa::Vector<T, 3>& c1,
                                          const luisa::Vector<T, 3>& c2)
{
    luisa::Matrix<T, 3> m;
    m[0] = c0;
    m[1] = c1;
    m[2] = c2;
    return m;
}

// Helper to get outer product of two vectors (result is a matrix)
template <typename T>
inline luisa::Matrix<T, 3> outer(const luisa::Vector<T, 3>& a, const luisa::Vector<T, 3>& b)
{
    luisa::Matrix<T, 3> m;
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            m[i][j] = a[j] * b[i];  // Note: m[col][row]
        }
    }
    return m;
}

// Helper to compute cofactor matrix (gradient of determinant)
template <typename T>
inline luisa::Matrix<T, 3> cofactor_matrix(const luisa::Matrix<T, 3>& F)
{
    luisa::Matrix<T, 3> pJpF;
    
    // Column 0: cross product of columns 1 and 2
    pJpF[0] = luisa::cross(F[1], F[2]);
    // Column 1: cross product of columns 2 and 0
    pJpF[1] = luisa::cross(F[2], F[0]);
    // Column 2: cross product of columns 0 and 1
    pJpF[2] = luisa::cross(F[0], F[1]);
    
    return pJpF;
}

// Helper to compute skew-symmetric (hat) matrix from a vector
template <typename T>
inline luisa::Matrix<T, 3> hat_matrix(const luisa::Vector<T, 3>& v)
{
    luisa::Matrix<T, 3> m;
    m[0] = luisa::Vector<T, 3>{T(0), v[2], -v[1]};
    m[1] = luisa::Vector<T, 3>{-v[2], T(0), v[0]};
    m[2] = luisa::Vector<T, 3>{v[1], -v[0], T(0)};
    return m;
}

template <typename T>
inline void E(T& R, const T& mu, const T& lambda, const luisa::Matrix<T, 3>& F)
{
    auto J = luisa::determinant(F);
    auto vecF = vec(F);
    auto Ic = luisa::dot(vecF, vecF);
    auto alpha = T(1) + T(0.75) * mu / lambda;
    R = T(0.5) * lambda * (J - alpha) * (J - alpha) + T(0.5) * mu * (Ic - T(3)) - T(0.5) * mu * luisa::log(Ic + T(1));
}

template <typename T>
inline void dEdVecF(luisa::Matrix<T, 3>& PEPF, const T& mu, const T& lambda, const luisa::Matrix<T, 3>& F)
{
    auto J = luisa::determinant(F);
    auto vecF = vec(F);
    auto Ic = luisa::dot(vecF, vecF);
    
    auto pJpF = cofactor_matrix(F);
    
    PEPF = mu * (T(1) - T(1) / (Ic + T(1))) * F + (lambda * (J - T(1) - T(0.75) * mu / lambda)) * pJpF;
}

template <typename T>
inline void ddEddVecF(luisa::Matrix<T, 9>& R, const T& mu, const T& lambda, const luisa::Matrix<T, 3>& F)
{
    auto J = luisa::determinant(F);
    auto vecF = vec(F);
    auto Ic = luisa::dot(vecF, vecF);
    
    // g1: gradient of Ic = 2*vec(F)
    luisa::Vector<T, 9> g1 = T(2) * vecF;
    
    // gJ: gradient of J (cofactor matrix flattened)
    auto cof = cofactor_matrix(F);
    luisa::Vector<T, 9> gJ = vec(cof);
    
    // f0hat, f1hat, f2hat: hat matrices of columns of F
    auto f0hat = hat_matrix(F[0]);
    auto f1hat = hat_matrix(F[1]);
    auto f2hat = hat_matrix(F[2]);
    
    // HJ is a 9x9 matrix representing the Hessian of J
    // We store it as a matrix where columns are the 9-element vectors
    // HJ[col][row] format
    luisa::Matrix<T, 9> HJ = luisa::Matrix<T, 9>::zero();
    
    // Fill HJ using the hat matrices
    // HJ.block<3, 3>(0, 3) = -f2hat;  -> columns 0-2, rows 3-5
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            HJ[0 + i][3 + j] = -f2hat[i][j];
        }
    }
    
    // HJ.block<3, 3>(0, 6) = f1hat;  -> columns 0-2, rows 6-8
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            HJ[0 + i][6 + j] = f1hat[i][j];
        }
    }
    
    // HJ.block<3, 3>(3, 0) = f2hat;  -> columns 3-5, rows 0-2
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            HJ[3 + i][0 + j] = f2hat[i][j];
        }
    }
    
    // HJ.block<3, 3>(3, 6) = -f0hat;  -> columns 3-5, rows 6-8
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            HJ[3 + i][6 + j] = -f0hat[i][j];
        }
    }
    
    // HJ.block<3, 3>(6, 0) = -f1hat;  -> columns 6-8, rows 0-2
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            HJ[6 + i][0 + j] = -f1hat[i][j];
        }
    }
    
    // HJ.block<3, 3>(6, 3) = f0hat;  -> columns 6-8, rows 3-5
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            HJ[6 + i][3 + j] = f0hat[i][j];
        }
    }
    
    // H1 is 2*Identity matrix
    luisa::Matrix<T, 9> H1 = luisa::Matrix<T, 9>::zero();
    for(int i = 0; i < 9; ++i)
    {
        H1[i][i] = T(2);
    }
    
    // g1 * g1^T and gJ * gJ^T are outer products
    // R = (Ic * mu) / (2 * (Ic + 1)) * H1 + lambda * (J - 1 - (3 * mu) / (4 * lambda)) * HJ 
    //     + (mu / (2 * (Ic + 1) * (Ic + 1))) * g1 * g1^T + lambda * gJ * gJ^T
    
    T coeff1 = (Ic * mu) / (T(2) * (Ic + T(1)));
    T coeff2 = lambda * (J - T(1) - (T(3) * mu) / (T(4) * lambda));
    T coeff3 = mu / (T(2) * (Ic + T(1)) * (Ic + T(1)));
    T coeff4 = lambda;
    
    for(int col = 0; col < 9; ++col)
    {
        for(int row = 0; row < 9; ++row)
        {
            R[col][row] = coeff1 * H1[col][row] + coeff2 * HJ[col][row]
                        + coeff3 * g1[row] * g1[col] + coeff4 * gJ[row] * gJ[col];
        }
    }
}
