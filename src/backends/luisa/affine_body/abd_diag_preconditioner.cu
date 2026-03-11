#include <linear_system/local_preconditioner.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/abd_linear_subsystem.h>
#include <linear_system/global_linear_system.h>
#include <finite_element/matrix_utils.h>
#include <kernel_cout.h>
#include <luisa/dsl/dsl.h>

namespace uipc::backend::luisa
{
/**
 * @brief Compute the inverse of a 12x12 matrix using LU decomposition with partial pivoting.
 * 
 * This function is adapted for LuisaCompute DSL and uses the Matrix12x12 type
 * defined in finite_element/matrix_utils.h.
 * 
 * @param A Input 12x12 matrix
 * @return Inverse of A
 */
LUISA_DEVICE inline Matrix12x12 inverse_12x12(const Matrix12x12& A) noexcept
{
    // Create working copy and identity matrix for inverse
    Matrix12x12 LU = A;
    Matrix12x12 inv = {};
    
    // Initialize inverse as identity matrix
    for(int i = 0; i < 12; i++)
        for(int j = 0; j < 12; j++)
            inv[i][j] = (i == j) ? 1.0f : 0.0f;
    
    // Row indices for pivoting
    int pivot[12];
    for(int i = 0; i < 12; i++)
        pivot[i] = i;
    
    // LU decomposition with partial pivoting
    for(int k = 0; k < 12; k++)
    {
        // Find pivot
        float max_val = luisa::abs(LU[k][k]);
        int max_row = k;
        for(int i = k + 1; i < 12; i++)
        {
            float val = luisa::abs(LU[i][k]);
            if(val > max_val)
            {
                max_val = val;
                max_row = i;
            }
        }
        
        // Swap rows k and max_row in LU
        if(max_row != k)
        {
            for(int j = 0; j < 12; j++)
            {
                float temp = LU[k][j];
                LU[k][j] = LU[max_row][j];
                LU[max_row][j] = temp;
            }
            // Swap pivot indices
            int temp_pivot = pivot[k];
            pivot[k] = pivot[max_row];
            pivot[max_row] = temp_pivot;
        }
        
        // Check for singular matrix
        float pivot_val = LU[k][k];
        if(luisa::abs(pivot_val) < 1e-10f)
        {
            // Return identity for near-singular matrices (fallback)
            Matrix12x12 identity = {};
            for(int i = 0; i < 12; i++)
                identity[i][i] = 1.0f;
            return identity;
        }
        
        // Compute multipliers and eliminate column k
        for(int i = k + 1; i < 12; i++)
        {
            LU[i][k] /= pivot_val;
            for(int j = k + 1; j < 12; j++)
            {
                LU[i][j] -= LU[i][k] * LU[k][j];
            }
        }
    }
    
    // Solve for inverse columns using forward/back substitution
    for(int col = 0; col < 12; col++)
    {
        // Create right-hand side (column of identity, with pivoting applied)
        float b[12];
        for(int i = 0; i < 12; i++)
        {
            b[i] = (pivot[i] == col) ? 1.0f : 0.0f;
        }
        
        // Forward substitution: solve Ly = b
        float y[12];
        for(int i = 0; i < 12; i++)
        {
            y[i] = b[i];
            for(int j = 0; j < i; j++)
            {
                y[i] -= LU[i][j] * y[j];
            }
        }
        
        // Back substitution: solve Ux = y
        float x[12];
        for(int i = 11; i >= 0; i--)
        {
            x[i] = y[i];
            for(int j = i + 1; j < 12; j++)
            {
                x[i] -= LU[i][j] * x[j];
            }
            x[i] /= LU[i][i];
        }
        
        // Store result as column of inverse
        for(int i = 0; i < 12; i++)
        {
            inv[i][col] = x[i];
        }
    }
    
    return inv;
}

class ABDDiagPreconditioner final : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    ABDLinearSubsystem* abd_linear_subsystem = nullptr;

    // LuisaCompute buffer for storing diagonal inverse matrices
    luisa::compute::Buffer<Matrix12x12> diag_inv;

    virtual void do_build(BuildInfo& info) override
    {
        auto& global_linear_system = require<GlobalLinearSystem>();
        abd_linear_subsystem       = &require<ABDLinearSubsystem>();

        info.connect(abd_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        using namespace luisa::compute;

        auto diag_hessian = abd_linear_subsystem->diag_hessian();
        
        // Recreate buffer if size changed (LuisaCompute buffers don't have resize)
        if(diag_inv.size() != diag_hessian.size())
        {
            diag_inv = device().create_buffer<Matrix12x12>(diag_hessian.size());
        }

        // Kernel to compute inverse of diagonal Hessian blocks
        Kernel1D assemble_kernel = [&](BufferVar<const Matrix12x12> diag_hessian_buf,
                                       BufferVar<Matrix12x12> diag_inv_buf) noexcept
        {
            auto idx = dispatch_id().x;
            $if(idx < diag_inv_buf.size())
            {
                // Read the diagonal Hessian block
                Var<Matrix12x12> H = diag_hessian_buf.read(idx);
                
                // Compute inverse using LU decomposition
                Var<Matrix12x12> H_inv = inverse_12x12(H);
                
                // Write result
                diag_inv_buf.write(idx, H_inv);
            };
        };

        // Compile and dispatch kernel
        auto assemble_shader = device().compile(assemble_kernel);
        stream() << assemble_shader(diag_hessian, diag_inv).dispatch(diag_inv.size());
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        using namespace luisa::compute;

        auto r_view = info.r();
        auto z_view = info.z();

        // Kernel to apply preconditioner: z = diag_inv * r
        // r and z are dense vectors with 12 floats per body
        Kernel1D apply_kernel = [&](BufferVar<const Float> r_buf,
                                    BufferVar<Float> z_buf,
                                    BufferVar<const Matrix12x12> diag_inv_buf) noexcept
        {
            auto idx = dispatch_id().x;
            $if(idx < diag_inv_buf.size())
            {
                // Read the inverse diagonal block
                Var<Matrix12x12> H_inv = diag_inv_buf.read(idx);
                
                // Read 12-element segment from r (input vector) and compute z = H_inv * r
                float z_vec[12];
                for(int i = 0; i < 12; i++)
                {
                    z_vec[i] = 0.0f;
                    for(int j = 0; j < 12; j++)
                    {
                        z_vec[i] += H_inv[i][j] * r_buf.read(idx * 12u + cast<uint>(j));
                    }
                }
                
                // Write 12-element segment to z (output vector)
                for(int i = 0; i < 12; i++)
                {
                    z_buf.write(idx * 12u + cast<uint>(i), z_vec[i]);
                }
            };
        };

        // Compile and dispatch kernel
        auto apply_shader = device().compile(apply_kernel);
        stream() << apply_shader(r_view, z_view, diag_inv).dispatch(diag_inv.size());
    }
};

REGISTER_SIM_SYSTEM(ABDDiagPreconditioner);
}  // namespace uipc::backend::luisa
