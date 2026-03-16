#include <diff_sim/diff_dof_reporter.h>
#include <linear_system/global_linear_system.h>
#include <utils/matrix_unpacker.h>
#include <kernel_cout.h>
#include <algorithm/matrix_converter.h>

namespace uipc::backend::luisa
{
/**
 * @brief Compute the Diff Dof for the current frame
 * 
 * This class directly take the result from the global linear system,
 * because the global linear system's full hessian matrix is exactly the 
 * 
 * $$
 * \frac{\partial^2 E}{\partial X^{[i]} \partial X^{[i]}}
 * $$
 * 
 * where $^{[i]}$ is the i-th frame
 */
class CurrentFrameDiffDofReporter : public DiffDofReporter
{
  public:
    using DiffDofReporter::DiffDofReporter;

    GlobalLinearSystem* global_linear_system = nullptr;
    IndexT              triplet_count        = 0;

    CBCOOMatrixView<Float, 3> bcoo_A()
    {
        return global_linear_system->m_impl.bcoo_A.view();
    }

    virtual void do_build(BuildInfo& info) override
    {
        global_linear_system = &require<GlobalLinearSystem>();
    }

    virtual void do_report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info) override
    {
        auto A        = bcoo_A();
        triplet_count = A.block_count;

        // 3x3 block coo matrix -> 1x1 coo matrix
        auto coo_triplet_count = triplet_count * 3 * 3;
        // report the triplet count
        info.triplet_count(coo_triplet_count);
    }

    virtual void do_assemble(GlobalDiffSimManager::DiffDofInfo& info) override
    {
        auto frame      = info.frame();
        auto dof_offset = info.dof_offset(frame);
        auto dof_count  = info.dof_count(frame);

        // Get the H matrix for the current frame
        // This is a scalar (1x1 block) COO matrix that will store the expanded Hessian
        auto H = info.H();
        
        // Get the BCOO matrix A from the global linear system
        // This contains 3x3 blocks
        auto A = bcoo_A();

        // Create kernel for unpacking matrix blocks
        // Each 3x3 block in A is expanded to 9 scalar entries in H
        // The dof_offset is added to position the entries correctly in the global matrix
        Kernel1D unpack_kernel = [](BufferVar<int> bcoo_rows,
                                     BufferVar<int> bcoo_cols,
                                     BufferVar<Matrix3x3> bcoo_vals,
                                     BufferVar<int> out_rows,
                                     BufferVar<int> out_cols,
                                     BufferVar<Matrix1x1> out_vals,
                                     UInt count,
                                     Int dof_off) {
            UInt I = dispatch_id().x;
            $if(I < count) {
                Int i = bcoo_rows.read(I);
                Int j = bcoo_cols.read(I);
                Matrix3x3 H3x3 = bcoo_vals.read(I);

                // Write 9 scalar entries for this 3x3 block
                // Each entry in the 3x3 block becomes a separate COO entry
                // The global row/col indices include the dof_offset
                $for(ii, 3) {
                    $for(jj, 3) {
                        UInt out_idx = I * 9 + ii * 3 + jj;
                        out_rows.write(out_idx, (i * 3 + ii) + dof_off);
                        out_cols.write(out_idx, (j * 3 + jj) + dof_off);
                        Matrix1x1 val;
                        val(0, 0) = H3x3(ii, jj);
                        out_vals.write(out_idx, val);
                    };
                };
            };
        };

        auto& device = global_linear_system->m_impl.bcoo_A.block_row_indices.device();
        auto stream = device.create_stream();
        auto unpack_shader = device.compile(unpack_kernel);
        
        // Get views to the output H matrix
        auto H_view = H.view();
        
        stream << unpack_shader(A.block_row_indices,
                                 A.block_col_indices,
                                 A.block_values,
                                 H_view.row_indices,
                                 H_view.col_indices,
                                 H_view.values,
                                 static_cast<uint>(A.block_count),
                                 static_cast<int>(dof_offset))
                      .dispatch(A.block_count);
    }
};

REGISTER_SIM_SYSTEM(CurrentFrameDiffDofReporter);
}  // namespace uipc::backend::luisa
