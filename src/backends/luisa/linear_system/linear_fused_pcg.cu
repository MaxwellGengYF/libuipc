#include <linear_system/linear_fused_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <uipc/common/timer.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
using namespace luisa::compute;

REGISTER_SIM_SYSTEM(LinearFusedPCG);

void LinearFusedPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto        solver_attr = config.find<std::string>("linear_system/solver");
    std::string solver_name =
        solver_attr ? solver_attr->view()[0] : std::string{"fused_pcg"};
    if(solver_name != "fused_pcg")
    {
        throw SimSystemException("LinearFusedPCG unused");
    }

    auto& global_linear_system = require<GlobalLinearSystem>();

    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<Float>("linear_system/tol_rate");
    global_tol_rate    = tol_rate_attr->view()[0];

    auto check_attr = config.find<IndexT>("linear_system/check_interval");
    if(check_attr)
        check_interval = check_attr->view()[0];

    // Allocate device scalar buffers (1 element each)
    auto& device = engine().device();
    d_rz     = device.create_buffer<Float>(1);
    d_pAp    = device.create_buffer<Float>(1);
    d_rz_new = device.create_buffer<Float>(1);
    d_converged = device.create_buffer<int>(1);

    logger::info("LinearFusedPCG: max_iter_ratio = {}, tol_rate = {}, check_interval = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 check_interval);
}

void LinearFusedPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    // x = 0
    engine().stream() << x.fill(0.0f);

    auto N = x.size();
    
    // Resize buffers if needed
    if(r.size() < N)
    {
        auto M = static_cast<size_t>(reserve_ratio * N);
        auto& device = engine().device();
        r  = device.create_buffer<Float>(M);
        z  = device.create_buffer<Float>(M);
        p  = device.create_buffer<Float>(M);
        Ap = device.create_buffer<Float>(M);
    }

    auto iter = fused_pcg(x, b, max_iter_ratio * b.size());

    info.iter_count(iter);
}

// Helper kernel for computing dot product x^T * y
static void compute_dot_product(BufferView<const Float> x,
                                BufferView<const Float> y,
                                BufferView<Float>       result,
                                Stream&                 stream,
                                Device&                 device)
{
    // Zero result buffer
    stream << result.fill(0.0f);

    auto n = x.size();
    
    // Use atomic operations for reduction
    Kernel1D dot_kernel = [&](BufferVar<const Float> x_buf,
                              BufferVar<const Float> y_buf,
                              BufferVar<Float>       result_buf,
                              UInt                   count) noexcept
    {
        auto idx = dispatch_id().x;
        $if(idx < count)
        {
            Float val = x_buf.read(idx) * y_buf.read(idx);
            result_buf.atomic(0).fetch_add(val);
        };
    };

    auto shader = device.compile(dot_kernel);
    stream << shader(x, y, result, static_cast<uint>(n)).dispatch(n);
}

// Helper kernel for computing vector norm ||x||
static Float compute_norm(BufferView<const Float> x,
                          Buffer<Float>&          temp_buffer,
                          Stream&                 stream,
                          Device&                 device)
{
    // Compute x^T * x
    compute_dot_product(x, x, temp_buffer.view(), stream, device);
    
    // Copy result to host
    Float result;
    stream << temp_buffer.view().copy_to(&result) << synchronize();
    
    return std::sqrt(result);
}

void LinearFusedPCG::check_init_rz_nan_inf(Float rz)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto& device = engine().device();
        auto& stream = engine().stream();
        
        auto norm_r = compute_norm(r.view(), d_rz, stream, device);
        auto norm_z = compute_norm(z.view(), d_rz, stream, device);
        bool r_bad  = !std::isfinite(norm_r);
        auto hint = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                            "preconditioner failed, likely due to inverse matrix calculation failure";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void LinearFusedPCG::check_iter_rz_nan_inf(Float rz, SizeT k)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto& device = engine().device();
        auto& stream = engine().stream();
        
        auto norm_r = compute_norm(r.view(), d_rz, stream, device);
        auto norm_z = compute_norm(z.view(), d_rz, stream, device);
        bool r_ok   = std::isfinite(norm_r);
        bool z_bad  = !std::isfinite(norm_z);
        auto hint   = (r_ok && z_bad) ?
                          "preconditioner failed, likely due to inverse matrix calculation failure" :
                          "PCG iteration diverged";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, FusedPCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    k,
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

// Kernel: x += alpha*p, r -= alpha*Ap where alpha = rz/pAp
static void fused_update_xr(BufferView<Float>       d_rz,
                            BufferView<Float>       d_pAp,
                            BufferView<Float>       x,
                            BufferView<const Float> p,
                            BufferView<Float>       r,
                            BufferView<const Float> Ap,
                            Stream&                 stream,
                            Device&                 device)
{
    auto n = x.size();
    
    Kernel1D update_kernel = [&](BufferVar<const Float> d_rz_buf,
                                 BufferVar<const Float> d_pAp_buf,
                                 BufferVar<Float>       x_buf,
                                 BufferVar<const Float> p_buf,
                                 BufferVar<Float>       r_buf,
                                 BufferVar<const Float> Ap_buf,
                                 UInt                   count) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < count)
        {
            Float alpha = d_rz_buf.read(0) / d_pAp_buf.read(0);
            Float x_val = x_buf.read(i) + alpha * p_buf.read(i);
            Float r_val = r_buf.read(i) - alpha * Ap_buf.read(i);
            x_buf.write(i, x_val);
            r_buf.write(i, r_val);
        };
    };

    auto shader = device.compile(update_kernel);
    stream << shader(d_rz, d_pAp, x, p, r, Ap, static_cast<uint>(n)).dispatch(n);
}

// Kernel: p = z + beta*p where beta = rz_new/rz
static void fused_update_p(BufferView<Float>       d_rz_new,
                           BufferView<Float>       d_rz,
                           BufferView<Float>       p,
                           BufferView<const Float> z,
                           Stream&                 stream,
                           Device&                 device)
{
    auto n = p.size();
    
    Kernel1D update_kernel = [&](BufferVar<const Float> d_rz_new_buf,
                                 BufferVar<const Float> d_rz_buf,
                                 BufferVar<Float>       p_buf,
                                 BufferVar<const Float> z_buf,
                                 UInt                   count) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < count)
        {
            Float beta = d_rz_new_buf.read(0) / d_rz_buf.read(0);
            Float p_val = z_buf.read(i) + beta * p_buf.read(i);
            p_buf.write(i, p_val);
        };
    };

    auto shader = device.compile(update_kernel);
    stream << shader(d_rz_new, d_rz, p, z, static_cast<uint>(n)).dispatch(n);
}

// Kernel: swap rz values (d_rz = d_rz_new)
static void fused_swap_rz(BufferView<Float> d_rz_new,
                          BufferView<Float> d_rz,
                          Stream&           stream,
                          Device&           device)
{
    Kernel1D swap_kernel = [&](BufferVar<const Float> d_rz_new_buf,
                               BufferVar<Float>       d_rz_buf) noexcept
    {
        d_rz_buf.write(0, d_rz_new_buf.read(0));
    };

    auto shader = device.compile(swap_kernel);
    stream << shader(d_rz_new, d_rz).dispatch(1);
}

// Kernel: copy vector (p = z)
static void copy_vector(BufferView<Float>       dst,
                        BufferView<const Float> src,
                        Stream&                 stream,
                        Device&                 device)
{
    stream << dst.copy_from(src);
}

SizeT LinearFusedPCG::fused_pcg(BufferView<Float>       x,
                                BufferView<const Float> b,
                                SizeT                   max_iter)
{
    Timer pcg_timer{"FusedPCG"};
    auto& stream = engine().stream();
    auto& device = engine().device();

    SizeT k = 0;
    
    // d_converged = 0
    stream << d_converged.view().fill(0);

    // r = b - A*x, but x0 = 0 so r = b
    stream << r.view().copy_from(b);

    // z = P^{-1} * r
    {
        Timer timer{"Apply Preconditioner"};
        apply_preconditioner(z.view(), r.view());
    }

    // p = z
    copy_vector(p.view(), z.view(), stream, device);

    // rz = r^T * z
    compute_dot_product(r.view(), z.view(), d_rz.view(), stream, device);
    
    // Copy rz to host
    Float rz_host;
    stream << d_rz.view().copy_to(&rz_host) << synchronize();
    
    check_init_rz_nan_inf(rz_host);
    Float abs_rz0 = std::abs(rz_host);

    if(abs_rz0 == Float{0.0})
        return 0;

    Float rz_tol = global_tol_rate * abs_rz0;
    SizeT effective_check_interval = check_interval > 0 ? check_interval : SizeT{1};

    for(k = 1; k < max_iter; ++k)
    {
        // Ap = A * p,  pAp = p^T * Ap
        {
            Timer timer{"SpMV"};
            spmv_dot(p.view(), Ap.view(), d_pAp.view());
        }

        // alpha = rz / pAp,  x += alpha * p,  r -= alpha * Ap
        fused_update_xr(d_rz.view(), d_pAp.view(), x, p.view(), r.view(), Ap.view(), stream, device);

        // z = P^{-1} * r
        {
            Timer timer{"Apply Preconditioner"};
            apply_preconditioner(z.view(), r.view());
        }

        // rz_new = r^T * z
        compute_dot_product(r.view(), z.view(), d_rz_new.view(), stream, device);

        // Check error ratio periodically to avoid per-iteration D2H synchronization.
        bool do_check = (k % effective_check_interval == 0) || (k + 1 == max_iter);
        if(do_check)
        {
            Float rz_new_host;
            stream << d_rz_new.view().copy_to(&rz_new_host) << synchronize();
            
            check_iter_rz_nan_inf(rz_new_host, k);
            if((std::abs(rz_new_host) / abs_rz0) <= global_tol_rate)
                break;
        }

        // p = z + beta * p, then rz = rz_new
        fused_update_p(d_rz_new.view(), d_rz.view(), p.view(), z.view(), stream, device);
        fused_swap_rz(d_rz_new.view(), d_rz.view(), stream, device);
    }

    return k;
}
}  // namespace uipc::backend::luisa
