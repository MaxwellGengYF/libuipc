#pragma once
#include <finite_element/matrix_utils.h>

namespace uipc::backend::luisa::fem
{
LUISA_DEVICE Float invariant2(const Matrix3x3& F);
LUISA_DEVICE Float invariant2(const Vector3& Sigma);
LUISA_DEVICE Float invariant3(const Matrix3x3& F);
LUISA_DEVICE Float invariant3(const Vector3& Sigma);
LUISA_DEVICE Float invariant4(const Matrix3x3& F, const Vector3& a);
LUISA_DEVICE Float invariant5(const Matrix3x3& F, const Vector3& a);

//tex: $\frac{\partial \det(\mathbf{F})}{\partial F}$
LUISA_DEVICE Matrix3x3 dJdF(const Matrix3x3& F);

// inverse material coordinates
LUISA_DEVICE Matrix3x3 Dm_inv(const Vector3& X0, const Vector3& X1, const Vector3& X2, const Vector3& X3);
LUISA_DEVICE Matrix3x3 Ds(const Vector3& X0, const Vector3& X1, const Vector3& X2, const Vector3& X3);
LUISA_DEVICE Matrix9x12 dFdx(const Matrix3x3& DmInv);
LUISA_DEVICE Matrix3x3    F(const Vector3& x0,
                          const Vector3& x1,
                          const Vector3& x2,
                          const Vector3& x3,
                          const Matrix3x3& DmInv);
}  // namespace uipc::backend::luisa::fem
