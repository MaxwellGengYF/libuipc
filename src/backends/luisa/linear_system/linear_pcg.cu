#include <linear_system/linear_pcg.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>
#include <utils/matrix_market.h>
#include <backends/common/backend_path_tool.h>
#include <uipc/common/timer.h>
#include <luisa/luisa-compute.h>

namespace uipc::backend::luisa
{
using namespace luisa::compute;

REGISTER_SIM_SYSTEM(LinearPCG);

void LinearPCG::do_build(BuildInfo& info)
{
    auto& config = world().scene().config();

    auto solver_attr = config.find<std::string>("linear_system/solver");
    UIPC_ASSERT(solver_attr, "linear_system/solver not found");
    if(solver_attr->view()[0] != "linear_pcg")
    {
        throw SimSystemException("LinearPCG unused");
    }

    auto& global_linear_system = require<GlobalLinearSystem>();

    // TODO: get info from the scene, now we just use the default value
    max_iter_ratio = 2;

    auto tol_rate_attr = config.find<Float>("linear_system/tol_rate");
    UIPC_ASSERT(tol_rate_attr, "linear_system/tol_rate not found");
    global_tol_rate    = tol_rate_attr->view()[0];

    auto dump_attr  = config.find<IndexT>("extras/debug/dump_linear_pcg");
    UIPC_ASSERT(dump_attr, "extras/debug/dump_linear_pcg not found");
    need_debug_dump = dump_attr->view()[0];
    
    // Allocate the converged flag buffer (always false)
    auto& device = engine().device();
    d_converged_false = device.create_buffer<int>(1);
    engine().stream() << d_converged_false.view().fill(0);

    logger::info("LinearPCG: max_iter_ratio = {}, tol_rate = {}, debug_dump = {}",
                 max_iter_ratio,
                 global_tol_rate,
                 need_debug_dump);
}

void LinearPCG::do_solve(GlobalLinearSystem::SolvingInfo& info)
{
    auto x = info.x();
    auto b = info.b();

    engine().stream() << x.fill(0.0f);

    auto N = x.size();
    if(z.size() < N)
    {
        auto M = static_cast<size_t>(reserve_ratio * N);
        auto& device = engine().device();
        z  = device.create_buffer<Float>(M);
        p  = device.create_buffer<Float>(M);
        r  = device.create_buffer<Float>(M);
        Ap = device.create_buffer<Float>(M);
    }
    
    // Reset converged flag to false
    engine().stream() << d_converged_false.view().fill(0);

    auto iter = pcg(x, b, max_iter_ratio * b.size());

    info.iter_count(iter);
}

void LinearPCG::dump_r_z(SizeT k)
{
    auto path_tool     = BackendPathTool(workspace());
    auto output_path   = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");
    auto output_path_r = fmt::format("{}r.{}.{}.{}.mtx",
                                     output_path.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k);

    export_vector_market(output_path_r, r.view(), engine().stream());
    logger::info("Dumped PCG r to {}", output_path_r);

    auto output_path_z = fmt::format("{}z.{}.{}.{}.mtx",
                                     output_path.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k);

    export_vector_market(output_path_z, z.view(), engine().stream());
    logger::info("Dumped PCG z to {}", output_path_z);
}

void LinearPCG::dump_p_Ap(SizeT k)
{
    auto path_tool     = BackendPathTool(workspace());
    auto output_folder = path_tool.workspace(UIPC_RELATIVE_SOURCE_FILE, "debug");

    auto output_path_p = fmt::format("{}p.{}.{}.{}.mtx",
                                     output_folder.string(),
                                     engine().frame(),
                                     engine().newton_iter(),
                                     k);

    export_vector_market(output_path_p, p.view(), engine().stream());
    logger::info("Dumped PCG p to {}", output_path_p);

    auto output_path_Ap = fmt::format("{}Ap.{}.{}.{}.mtx",
                                      output_folder.string(),
                                      engine().frame(),
                                      engine().newton_iter(),
                                      k);
    export_vector_market(output_path_Ap, Ap.view(), engine().stream());
    logger::info("Dumped PCG Ap to {}", output_path_Ap);
}

// Helper: compute dot product using atomic reduction
static Float compute_dot(BufferView<const Float> x,
                         BufferView<const Float> y,
                         Buffer<Float>&          temp_buffer,
                         Stream&                 stream,
                         Device&                 device)
{
    // Zero temp buffer
    stream << temp_buffer.view().fill(0.0f);
    
    auto n = x.size();
    
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
    stream << shader(x, y, temp_buffer.view(), static_cast<uint>(n)).dispatch(n);
    
    Float result;
    stream << temp_buffer.view().copy_to(&result) << synchronize();
    return result;
}

// Helper: compute vector norm
static Float compute_norm(BufferView<const Float> x,
                          Buffer<Float>&          temp_buffer,
                          Stream&                 stream,
                          Device&                 device)
{
    return std::sqrt(compute_dot(x, x, temp_buffer, stream, device));
}

void LinearPCG::check_init_rz_nan_inf(Float rz)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto& device = engine().device();
        auto& stream = engine().stream();
        
        // Use z buffer as temp for norm computation
        auto norm_r = compute_norm(r.view(), z, stream, device);
        auto norm_z = compute_norm(z.view(), z, stream, device);
        bool r_bad  = !std::isfinite(norm_r);
        auto hint = r_bad ? "gradient assembling produced NaN values, likely due to error in formula implementation" :
                            "preconditioner failed, likely due to inverse matrix calculation failure";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, PCG Init: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
                    "Hint: {}.",
                    engine().frame(),
                    engine().newton_iter(),
                    rz,
                    norm_r,
                    norm_z,
                    hint);
    }
}

void LinearPCG::check_iter_rz_nan_inf(Float rz, SizeT k)
{
    if(!std::isfinite(rz)) [[unlikely]]
    {
        auto& device = engine().device();
        auto& stream = engine().stream();
        
        auto norm_r = compute_norm(r.view(), z, stream, device);
        auto norm_z = compute_norm(z.view(), z, stream, device);
        bool r_ok   = std::isfinite(norm_r);
        bool z_bad  = !std::isfinite(norm_z);
        auto hint = (r_ok && z_bad) ?
                        "preconditioner failed, likely due to inverse matrix calculation failure" :
                        "PCG iteration diverged";
        UIPC_ASSERT(false,
                    "Frame {}, Newton {}, PCG Iter {}: r^T*z = {}, norm(r) = {}, norm(z) = {}. "
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

// Kernel: x += alpha*p, r -= alpha*Ap
static void update_xr(Float                   alpha,
                      BufferView<Float>       x,
                      BufferView<const Float> p,
                      BufferView<Float>       r,
                      BufferView<const Float> Ap,
                      Stream&                 stream,
                      Device&                 device)
{
    auto n = x.size();
    
    Kernel1D update_kernel = [&](BufferVar<Float>       x_buf,
                                 BufferVar<const Float> p_buf,
                                 BufferVar<Float>       r_buf,
                                 BufferVar<const Float> Ap_buf,
                                 Float                  a,
                                 UInt                   count) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < count)
        {
            Float x_val = x_buf.read(i) + a * p_buf.read(i);
            Float r_val = r_buf.read(i) - a * Ap_buf.read(i);
            x_buf.write(i, x_val);
            r_buf.write(i, r_val);
        };
    };

    auto shader = device.compile(update_kernel);
    stream << shader(x, p, r, Ap, alpha, static_cast<uint>(n)).dispatch(n);
}

// Kernel: p = z + beta*p
static void update_p(BufferView<Float>       p,
                     BufferView<const Float> z,
                     Float                   beta,
                     Stream&                 stream,
                     Device&                 device)
{
    auto n = p.size();
    
    Kernel1D update_kernel = [&](BufferVar<Float>       p_buf,
                                 BufferVar<const Float> z_buf,
                                 Float                  b,
                                 UInt                   count) noexcept
    {
        auto i = dispatch_id().x;
        $if(i < count)
        {
            Float p_val = z_buf.read(i) + b * p_buf.read(i);
            p_buf.write(i, p_val);
        };
    };

    auto shader = device.compile(update_kernel);
    stream << shader(p, z, beta, static_cast<uint>(n)).dispatch(n);
}

SizeT LinearPCG::pcg(BufferView<Float> x, CBufferViewFloat b, SizeT max_iter)
{
    Timer pcg_timer{"PCG"};
    auto& stream = engine().stream();
    auto& device = engine().device();

    SizeT k = 0;
    // r = b - A * x
    {
        // r = b;
        stream << r.view().copy_from(b);

        // x == 0, so we don't need to do the following
        // r = - A * x + r
        //spmv(-1.0, x.as_const(), 1.0, r.view());
    }

    Float alpha, beta, rz, abs_rz0;

    // z = P * r (apply preconditioner)
    {
        Timer timer{"Apply Preconditioner"};
        apply_preconditioner(z.view(), r.view());
    }

    if(need_debug_dump) [[unlikely]]
        dump_r_z(k);

    // p = z
    stream << p.view().copy_from(z.view());

    // init rz
    // rz = r^T * z
    rz = compute_dot(r.view(), z.view(), z, stream, device);
    check_init_rz_nan_inf(rz);

    abs_rz0 = std::abs(rz);

    // check convergence
    if(accuracy_statisfied(r.view()) && abs_rz0 == Float{0.0})
        return 0;

    for(k = 1; k < max_iter; ++k)
    {
        {
            Timer timer{"SpMV"};
            spmv(p.view(), Ap.view());
        }

        if(need_debug_dump) [[unlikely]]
            dump_p_Ap(k);

        // alpha = rz / p^T * Ap
        alpha = rz / compute_dot(p.view(), Ap.view(), z, stream, device);

        // x = x + alpha * p
        // r = r - alpha * Ap
        update_xr(alpha, x, p.view(), r.view(), Ap.view(), stream, device);

        // z = P * r (apply preconditioner)
        {
            Timer timer{"Apply Preconditioner"};
            apply_preconditioner(z.view(), r.view());
        }

        if(need_debug_dump) [[unlikely]]
            dump_r_z(k);

        // rz_new = r^T * z
        Float rz_new = compute_dot(r.view(), z.view(), z, stream, device);
        check_iter_rz_nan_inf(rz_new, k);

        // check convergence
        if(accuracy_statisfied(r.view()) && std::abs(rz_new) <= global_tol_rate * abs_rz0)
            break;

        // beta = rz_new / rz
        beta = rz_new / rz;

        // p = z + beta * p
        update_p(p.view(), z.view(), beta, stream, device);

        rz = rz_new;
    }

    return k;
}
}  // namespace uipc::backend::luisa
