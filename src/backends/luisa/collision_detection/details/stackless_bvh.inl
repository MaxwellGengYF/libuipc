#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

namespace uipc::backend::luisa
{
namespace culbvh
{
using aabb          = uipc::backend::luisa::AABB;
using stacklessnode = uipc::backend::luisa::StacklessBVH::Node;
using Vector2i      = uipc::Vector2i;

using uint   = uint32_t;
using ullint = unsigned long long int;

constexpr int K_THREADS = 256;
constexpr int K_WARPS   = K_THREADS >> 5;

constexpr int K_REDUCTION_LAYER  = 5;
constexpr int K_REDUCTION_NUM    = 1 << K_REDUCTION_LAYER;
constexpr int K_REDUCTION_MODULO = K_REDUCTION_NUM - 1;

constexpr int    aabbBits  = 15;
constexpr int    aabbRes   = (1 << aabbBits) - 2;
constexpr int    indexBits = 64 - 3 * aabbBits;
constexpr int    offset3   = aabbBits * 3;
constexpr int    offset2   = aabbBits * 2;
constexpr int    offset1   = aabbBits * 1;
constexpr ullint indexMask = 0xFFFFFFFFFFFFFFFFu << offset3;
constexpr uint   aabbMask  = 0xFFFFFFFFu >> (32 - aabbBits);
constexpr uint   MaxIndex  = 0xFFFFFFFFFFFFFFFFu >> offset3;

constexpr uint MAX_CD_NUM_PER_VERT = 64;
constexpr int  MAX_RES_PER_BLOCK   = 1024;

struct PlainAABB
{
    float3 _min, _max;
};

LUISA_GENERIC LUISA_INLINE PlainAABB toPlainAABB(const aabb& box)
{
    PlainAABB res;
    res._min = make_float3(box.min().x(), box.min().y(), box.min().z());
    res._max = make_float3(box.max().x(), box.max().y(), box.max().z());
    return res;
}

LUISA_GENERIC LUISA_INLINE aabb fromPlainAABB(const PlainAABB& box)
{
    aabb aabb;
    aabb.min() = Vector3(box._min.x, box._min.y, box._min.z);
    aabb.max() = Vector3(box._max.x, box._max.y, box._max.z);
    return aabb;
}

struct intAABB
{
    int3 _min, _max;

    LUISA_GENERIC LUISA_INLINE void convertFrom(const aabb& other, float3& origin, float3& delta)
    {
        _min.x = static_cast<int>((other.min().x() - origin.x) / delta.x);
        _min.y = static_cast<int>((other.min().y() - origin.y) / delta.y);
        _min.z = static_cast<int>((other.min().z() - origin.z) / delta.z);
        _max.x = static_cast<int>(ceilf((other.max().x() - origin.x) / delta.x));
        _max.y = static_cast<int>(ceilf((other.max().y() - origin.y) / delta.y));
        _max.z = static_cast<int>(ceilf((other.max().z() - origin.z) / delta.z));
    }
};

template <typename T>
LUISA_GENERIC LUISA_INLINE T __mm_min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
LUISA_GENERIC LUISA_INLINE T __mm_max(T a, T b)
{
    return a > b ? a : b;
}

LUISA_DEVICE LUISA_INLINE float atomicMinf(luisa::compute::Float* addr, float value)
{
    float old;
    old = (value >= 0) ?
              luisa::compute::atomic_cast<float>(luisa::compute::atomic_min(
                  reinterpret_cast<luisa::compute::Int*>(addr), 
                  luisa::compute::atomic_cast<int>(value))) :
              luisa::compute::atomic_cast<float>(luisa::compute::atomic_max(
                  reinterpret_cast<luisa::compute::UInt*>(addr), 
                  luisa::compute::atomic_cast<uint>(value)));
    return old;
}

LUISA_DEVICE LUISA_INLINE float atomicMaxf(luisa::compute::Float* addr, float value)
{
    float old;
    old = (value >= 0) ?
              luisa::compute::atomic_cast<float>(luisa::compute::atomic_max(
                  reinterpret_cast<luisa::compute::Int*>(addr), 
                  luisa::compute::atomic_cast<int>(value))) :
              luisa::compute::atomic_cast<float>(luisa::compute::atomic_min(
                  reinterpret_cast<luisa::compute::UInt*>(addr), 
                  luisa::compute::atomic_cast<uint>(value)));
    return old;
}

LUISA_GENERIC LUISA_INLINE uint expandBits(uint v)
{  ///< Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

LUISA_GENERIC LUISA_INLINE uint morton3D(float x, float y, float z)
{  ///< Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
    x       = ::fmin(::fmax(x * 1024.0f, 0.0f), 1023.0f);
    y       = ::fmin(::fmax(y * 1024.0f, 0.0f), 1023.0f);
    z       = ::fmin(::fmax(z * 1024.0f, 0.0f), 1023.0f);
    uint xx = expandBits((uint)x);
    uint yy = expandBits((uint)y);
    uint zz = expandBits((uint)z);
    return (xx * 4 + yy * 2 + zz);
}

// Custom comparison for int3 based on lexicographical ordering
LUISA_GENERIC LUISA_INLINE bool lessThan(const int3& a, const int3& b)
{
    if(a.x != b.x)
        return a.x < b.x;
    if(a.y != b.y)
        return a.y < b.y;
    return a.z < b.z;
}

LUISA_GENERIC LUISA_INLINE Vector2i to_eigen(int2 v)
{
    return Vector2i{v.x, v.y};
}

LUISA_GENERIC LUISA_INLINE int2 make_ordered_pair(int a, int b)
{
    if(a < b)
        return int2{a, b};
    else
        return int2{b, a};
}

LUISA_GENERIC LUISA_INLINE float3 operator-(const float3& v0, const float3& v1)
{
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}

LUISA_GENERIC LUISA_INLINE void SafeCopyTo(int2* sharedRes,
                                           int   totalResInBlock,

                                           Vector2i* globalRes,
                                           int       globalIdx,
                                           int       maxRes)
{
    if(globalIdx >= maxRes      // Out of memory for results.
       || totalResInBlock == 0  // No results to write
    )
        return;

    auto CopyCount = std::min(totalResInBlock, maxRes - globalIdx);

    // Copy full blocks
    int fullBlocks = (CopyCount - 1) / (int)blockDim.x;
    for(int i = 0; i < fullBlocks; i++)
    {
        int offset                    = i * blockDim.x + threadIdx.x;
        globalRes[globalIdx + offset] = to_eigen(sharedRes[offset]);
    }

    // Copy the rest
    int offset = fullBlocks * blockDim.x + threadIdx.x;
    if(offset < CopyCount)
        globalRes[globalIdx + offset] = to_eigen(sharedRes[offset]);
}

}  // namespace culbvh

LUISA_INLINE void StacklessBVH::Impl::calcMaxBVFromBox(luisa::compute::BufferView<AABB> aabbs,
                                                      luisa::compute::BufferView<AABB> scene_box)
{
    using namespace culbvh;
    using namespace luisa::compute;

    auto numQuery = aabbs.size();
    
    if(numQuery == 0) return;

    // Kernel for computing scene box from AABBs
    Kernel1D kernel = [&](BufferVar<AABB> box, BufferVar<AABB> _bv) noexcept {
        set_block_size(K_THREADS, 1u, 1u);
        
        $shared<PlainAABB> aabbData{K_WARPS};
        
        $int idx = dispatch_x();
        $int warpTid = thread_x() % 32;
        $int warpId  = thread_x() >> 5;
        
        $if(idx == 0) {
            _bv.write(0, AABB());
        };
        
        sync_block();
        
        $if(idx < cast<int>(numQuery)) {
            auto bv = box.read(idx);
            PlainAABB temp;
            temp._min = make_float3(bv.min().x(), bv.min().y(), bv.min().z());
            temp._max = make_float3(bv.max().x(), bv.max().y(), bv.max().z());
            
            sync_block();
            
            // Warp shuffle reduction
            $float tempMinX = temp._min.x;
            $float tempMinY = temp._min.y;
            $float tempMinZ = temp._min.z;
            $float tempMaxX = temp._max.x;
            $float tempMaxY = temp._max.y;
            $float tempMaxZ = temp._max.z;
            
            for($int i = 1; i < 32; i = i << 1) {
                $float otherMinX = warp_shuffle_down(tempMinX, cast<uint>(i));
                $float otherMinY = warp_shuffle_down(tempMinY, cast<uint>(i));
                $float otherMinZ = warp_shuffle_down(tempMinZ, cast<uint>(i));
                $float otherMaxX = warp_shuffle_down(tempMaxX, cast<uint>(i));
                $float otherMaxY = warp_shuffle_down(tempMaxY, cast<uint>(i));
                $float otherMaxZ = warp_shuffle_down(tempMaxZ, cast<uint>(i));
                tempMinX = __mm_min(tempMinX, otherMinX);
                tempMinY = __mm_min(tempMinY, otherMinY);
                tempMinZ = __mm_min(tempMinZ, otherMinZ);
                tempMaxX = __mm_max(tempMaxX, otherMaxX);
                tempMaxY = __mm_max(tempMaxY, otherMaxY);
                tempMaxZ = __mm_max(tempMaxZ, otherMaxZ);
            }
            
            $if(warpTid == 0) {
                PlainAABB reduced;
                reduced._min = make_float3(tempMinX, tempMinY, tempMinZ);
                reduced._max = make_float3(tempMaxX, tempMaxY, tempMaxZ);
                aabbData[warpId] = reduced;
            };
        };
        
        sync_block();
        
        // Final reduction
        $int warpNum = cast<int>((numQuery + 31) / 32);
        $if(thread_x() < cast<uint>(warpNum)) {
            auto temp = aabbData[thread_x()];
            $float tempMinX = temp._min.x;
            $float tempMinY = temp._min.y;
            $float tempMinZ = temp._min.z;
            $float tempMaxX = temp._max.x;
            $float tempMaxY = temp._max.y;
            $float tempMaxZ = temp._max.z;
            
            for($int i = 1; i < warpNum; i = i << 1) {
                $float otherMinX = warp_shuffle_down(tempMinX, cast<uint>(i));
                $float otherMinY = warp_shuffle_down(tempMinY, cast<uint>(i));
                $float otherMinZ = warp_shuffle_down(tempMinZ, cast<uint>(i));
                $float otherMaxX = warp_shuffle_down(tempMaxX, cast<uint>(i));
                $float otherMaxY = warp_shuffle_down(tempMaxY, cast<uint>(i));
                $float otherMaxZ = warp_shuffle_down(tempMaxZ, cast<uint>(i));
                tempMinX = __mm_min(tempMinX, otherMinX);
                tempMinY = __mm_min(tempMinY, otherMinY);
                tempMinZ = __mm_min(tempMinZ, otherMinZ);
                tempMaxX = __mm_max(tempMaxX, otherMaxX);
                tempMaxY = __mm_max(tempMaxY, otherMaxY);
                tempMaxZ = __mm_max(tempMaxZ, otherMaxZ);
            }
            
            $if(thread_x() == 0) {
                auto bv = _bv.read(0);
                // Atomic min/max operations
                auto minPtr = reinterpret_cast<Float*>(&(bv.min().x()));
                auto maxPtr = reinterpret_cast<Float*>(&(bv.max().x()));
                // Note: In luisa-compute, we use buffer atomic operations
            };
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(aabbs, scene_box).dispatch(numQuery);
}

LUISA_INLINE void StacklessBVH::Impl::calcMCsFromBox(luisa::compute::BufferView<AABB>     aabbs,
                                                    luisa::compute::BufferView<AABB>     scene_box,
                                                    luisa::compute::BufferView<uint32_t> codes)
{
    using namespace culbvh;
    using namespace luisa::compute;

    Kernel1D kernel = [&](BufferVar<AABB> box, BufferVar<AABB> scene, BufferVar<uint32_t> mcodes) noexcept {
        $int idx = dispatch_x();
        $if(idx < cast<int>(aabbs.size())) {
            AABB bv = box.read(idx);
            auto center = bv.center();
            float3 c = make_float3(center.x(), center.y(), center.z());
            
            auto sceneMin = scene.read(0).min();
            float3 sceneMinVec = make_float3(sceneMin.x(), sceneMin.y(), sceneMin.z());
            float3 offset = c - sceneMinVec;
            
            auto sceneSize = scene.read(0).sizes();
            mcodes.write(idx, morton3D(offset.x / sceneSize.x(),
                                       offset.y / sceneSize.y(),
                                       offset.z / sceneSize.z()));
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(aabbs, scene_box, codes).dispatch(aabbs.size());
}

/// incoherent access, thus poor performance
LUISA_INLINE void StacklessBVH::Impl::calcInverseMapping()
{
    using namespace luisa::compute;

    Kernel1D kernel = [&](BufferVar<int32_t> map, BufferVar<int32_t> invMap) noexcept {
        $int idx = dispatch_x();
        $if(idx < cast<int>(sorted_id.size())) {
            invMap.write(map.read(idx), idx);
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(sorted_id.view(), primMap.view()).dispatch(sorted_id.size());
}

LUISA_INLINE void StacklessBVH::Impl::buildPrimitivesFromBox(luisa::compute::BufferView<AABB> aabbs)
{  ///< update idx-th _bxs to idx-th leaf
    using namespace luisa::compute;
    
    Kernel1D kernel = [&](BufferVar<int32_t> _primIdx,
                          BufferVar<AABB>     _primBox,
                          BufferVar<int32_t>  _primMap,
                          BufferVar<AABB>     box) noexcept {
        $int idx = dispatch_x();
        $if(idx < cast<int>(aabbs.size())) {
            int newIdx = _primMap.read(idx);
            AABB bv = box.read(idx);
            _primIdx.write(newIdx, idx);
            _primBox.write(newIdx, bv);
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(ext_idx.view(), ext_aabb.view(), primMap.view(), 
                     BufferVar<AABB>{aabbs.native_handle(), aabbs.handle(), aabbs.stride(), 
                                    aabbs.offset_bytes(), aabbs.size(), aabbs.total_size()})
           .dispatch(aabbs.size());
}

LUISA_INLINE void StacklessBVH::Impl::calcExtNodeSplitMetrics()
{
    using namespace luisa::compute;
    
    Kernel1D kernel = [&](BufferVar<uint32_t> _codes, BufferVar<int32_t> _metrics, $int extsize) noexcept {
        $int idx = dispatch_x();
        $if(idx < extsize) {
            $int metric = idx != extsize - 1 ?
                              32 - clz(_codes.read(idx) ^ _codes.read(idx + 1)) :
                              33;
            _metrics.write(idx, metric);
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(mtcode.view(), metric.view(), cast<int>(mtcode.size()))
           .dispatch(mtcode.size());
}

LUISA_INLINE void StacklessBVH::Impl::buildIntNodes(int size)
{
    using namespace luisa::compute;

    Kernel1D kernel = [&](
        $int size,
        // leaf nodes
        BufferVar<uint32_t> _depths,
        BufferVar<int32_t>  _lvs_lca,
        BufferVar<int32_t>  _lvs_metric,
        BufferVar<uint32_t> _lvs_par,
        BufferVar<uint32_t> _lvs_mark,
        BufferVar<AABB>     _lvs_box,
        // internal nodes
        BufferVar<int32_t>  _tks_rc,
        BufferVar<int32_t>  _tks_lc,
        BufferVar<int32_t>  _tks_range_y,
        BufferVar<int32_t>  _tks_range_x,
        BufferVar<uint32_t> _tks_mark,
        BufferVar<AABB>     _tks_box,
        BufferVar<uint32_t> _flag,
        BufferVar<int32_t>  _tks_par) noexcept {
        
        set_block_size(256u, 1u, 1u);
        
        $int idx = dispatch_x();
        $if(idx < size) {
            _lvs_lca.write(idx, -1);
            _depths.write(idx, 0);
            $int l = idx - 1;
            $int r = idx;
            $bool mark;
            $if(l >= 0) {
                mark = _lvs_metric.read(l) < _lvs_metric.read(r);
            } $else {
                mark = false;
            };
            $int cur = mark ? l : r;
            
            _lvs_par.write(idx, cur);
            
            $if(_flag.size() == 0) {
                // when we only have 1 external node
                // there is no internal node to build
                $return;
            };
            
            $if(mark) {
                _tks_rc.write(cur, idx);
                _tks_range_y.write(cur, idx);
                _tks_mark.atomic(cur).fetch_or(0x00000002);
                _lvs_mark.write(idx, 0x00000007);
            } $else {
                _tks_lc.write(cur, idx);
                _tks_range_x.write(cur, idx);
                _tks_mark.atomic(cur).fetch_or(0x00000001);
                _lvs_mark.write(idx, 0x00000003);
            };
            
            // Note: threadfence equivalent in luisa-compute
            // We need to use memory barriers or different algorithm
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(
        size,
        count.view(),
        ext_lca.view(),
        metric.view(),
        ext_par.view(),
        ext_mark.view(),
        ext_aabb.view(),
        int_rc.view(),
        int_lc.view(),
        int_range_y.view(),
        int_range_x.view(),
        int_mark.view(),
        int_aabb.view(),
        flags.view(),
        int_par.view()
    ).dispatch(size);
}

LUISA_INLINE void StacklessBVH::Impl::calcIntNodeOrders(int size)
{
    using namespace luisa::compute;

    Kernel1D kernel = [&](
        BufferVar<int32_t>  _tks_lc,
        BufferVar<int32_t>  _lcas,
        BufferVar<uint32_t> _depths,
        BufferVar<uint32_t> _offsets,
        BufferVar<int32_t>  _tkMap,
        $int size) noexcept {
        
        $int idx = dispatch_x();
        $if(idx < size) {
            $int node  = _lcas.read(idx);
            $int depth = _depths.read(idx);
            $int id    = _offsets.read(idx);
            
            $if(node != -1) {
                $for($i, depth) {
                    _tkMap.write(node, id);
                    id = id + 1;
                    node = _tks_lc.read(node);
                };
            };
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(
        int_lc.view(),
        ext_lca.view(),
        count.view(),
        offsetTable.view(),
        tkMap.view(),
        size
    ).dispatch(size);
}

LUISA_INLINE void StacklessBVH::Impl::updateBvhExtNodeLinks(int size)
{
    using namespace luisa::compute;

    $if(flags.size() == 0) {
        // no internal nodes, thus no need to update
        $return;
    };

    Kernel1D kernel = [&](
        BufferVar<int32_t>  _mapTable,
        BufferVar<int32_t>  _lcas,
        BufferVar<uint32_t> _pars,
        $int size) noexcept {
        
        $int idx = dispatch_x();
        $if(idx < size) {
            $int newPar = _mapTable.read(_pars.read(idx));
            _pars.write(idx, newPar);
            $int ori = _lcas.read(idx);
            $if(ori != -1) {
                _lcas.write(idx, _mapTable.read(ori) << 1);
            } $else {
                _lcas.write(idx, idx << 1 | 1);
            };
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(
        tkMap.view(),
        ext_lca.view(),
        ext_par.view(),
        size
    ).dispatch(size);
}

LUISA_INLINE void StacklessBVH::Impl::reorderNode(int intSize)
{
    using namespace culbvh;
    using namespace luisa::compute;

    Kernel1D kernel = [&](
        $int intSize,
        // leaf nodes
        BufferVar<int32_t>  _lvs_lca,
        BufferVar<AABB>     _lvs_box,
        // internal nodes
        BufferVar<int32_t>  _tkMap,
        BufferVar<int32_t>  _unorderedTks_lc,
        BufferVar<uint32_t> _unorderedTks_mark,
        BufferVar<int32_t>  _unorderedTks_rangey,
        BufferVar<AABB>     _unorderedTks_box,
        // total nodes
        BufferVar<Node>     _nodes) noexcept {
        
        $int idx = dispatch_x();
        $if(idx <= intSize) {
            stacklessnode Node;
            Node.lc = -1;
            $int escape = _lvs_lca.read(idx + 1);
            
            $if(escape == -1) {
                Node.escape = -1;
            } $else {
                $int bLeaf = escape & 1;
                escape = escape >> 1;
                Node.escape = escape + (bLeaf ? intSize : 0);
            };
            Node.bound = _lvs_box.read(idx);
            
            _nodes.write(idx + intSize, Node);
            
            $if(idx >= intSize) {
                $return;
            };
            
            stacklessnode internalNode;
            $int newId = _tkMap.read(idx);
            $uint mark = _unorderedTks_mark.read(idx);
            
            internalNode.lc = (mark & 1) ? _unorderedTks_lc.read(idx) + intSize :
                                           _tkMap.read(_unorderedTks_lc.read(idx));
            internalNode.bound = _unorderedTks_box.read(idx);
            
            $int internalEscape = _lvs_lca.read(_unorderedTks_rangey.read(idx) + 1);
            
            $if(internalEscape == -1) {
                internalNode.escape = -1;
            } $else {
                $int bLeaf = internalEscape & 1;
                internalEscape = internalEscape >> 1;
                internalNode.escape = internalEscape + (bLeaf ? intSize : 0);
            };
            _nodes.write(newId, internalNode);
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(
        intSize,
        ext_lca.view(),
        ext_aabb.view(),
        tkMap.view(),
        int_lc.view(),
        int_mark.view(),
        int_range_y.view(),
        int_aabb.view(),
        nodes.view()
    ).dispatch(intSize + 1);
}

inline void StacklessBVH::Impl::build(luisa::compute::BufferView<AABB> aabbs)
{
    using namespace luisa::compute;
    
    objs         = aabbs;
    auto numObjs = aabbs.size();

    if(aabbs.size() == 0)
        return;

    const unsigned int numInternalNodes = numObjs - 1;
    const unsigned int numNodes = numObjs * 2 - 1;

    // Resize buffers
    mtcode  = device.create_buffer<uint32_t>(numObjs);
    sorted_id = device.create_buffer<int32_t>(numObjs);
    primMap = device.create_buffer<int32_t>(numObjs);
    ext_aabb = device.create_buffer<AABB>(numObjs);
    ext_idx = device.create_buffer<int32_t>(numObjs);
    ext_lca = device.create_buffer<int32_t>(numObjs + 1);
    ext_par = device.create_buffer<uint32_t>(numObjs);
    ext_mark = device.create_buffer<uint32_t>(numObjs);

    metric = device.create_buffer<int32_t>(numObjs);
    tkMap = device.create_buffer<int32_t>(numObjs);
    offsetTable = device.create_buffer<uint32_t>(numObjs);
    count = device.create_buffer<uint32_t>(numObjs);

    flags = device.create_buffer<uint32_t>(numInternalNodes);
    int_lc = device.create_buffer<int32_t>(numInternalNodes);
    int_rc = device.create_buffer<int32_t>(numInternalNodes);
    int_par = device.create_buffer<int32_t>(numInternalNodes);
    int_range_x = device.create_buffer<int32_t>(numInternalNodes);
    int_range_y = device.create_buffer<int32_t>(numInternalNodes);
    int_mark = device.create_buffer<uint32_t>(numInternalNodes);
    int_aabb = device.create_buffer<AABB>(numInternalNodes);

    nodes = device.create_buffer<Node>(numNodes);

    // Initialize flags to 0 using kernel
    Kernel1D init_kernel = [&](BufferVar<uint32_t> buf, $uint value) noexcept {
        $int idx = dispatch_x();
        $if(idx < cast<int>(buf.size())) {
            buf.write(idx, value);
        };
    };
    
    auto init_shader = device.compile(init_kernel);
    stream << init_shader(flags.view(), 0u).dispatch(numInternalNodes)
           << init_shader(ext_mark.view(), 7u).dispatch(numObjs)
           << init_shader(ext_lca.view(), 0u).dispatch(numObjs + 1)
           << init_shader(ext_par.view(), 0u).dispatch(numObjs);

    calcMaxBVFromBox(aabbs, scene_box.view());

    calcMCsFromBox(aabbs, scene_box.view(), mtcode.view());

    // Sort by morton codes - use host-side sort
    // Copy to host, sort, copy back
    std::vector<uint32_t> h_mtcode(numObjs);
    std::vector<int32_t> h_sorted_id(numObjs);
    stream << mtcode.view().copy_to(h_mtcode.data()) << synchronize();
    
    std::iota(h_sorted_id.begin(), h_sorted_id.end(), 0);
    std::sort(h_sorted_id.begin(), h_sorted_id.end(), 
              [&](int32_t a, int32_t b) { return h_mtcode[a] < h_mtcode[b]; });
    
    stream << sorted_id.view().copy_from(h_sorted_id.data()) << synchronize();

    calcInverseMapping();

    buildPrimitivesFromBox(aabbs);

    calcExtNodeSplitMetrics();

    buildIntNodes(numObjs);

    // Exclusive scan on host
    std::vector<uint32_t> h_count(numObjs);
    stream << count.view().copy_to(h_count.data()) << synchronize();
    std::vector<uint32_t> h_offset(numObjs);
    std::exclusive_scan(h_count.begin(), h_count.end(), h_offset.begin(), 0u);
    stream << offsetTable.view().copy_from(h_offset.data()) << synchronize();

    calcIntNodeOrders(numObjs);

    // fill the last ext_lca to -1
    std::vector<int32_t> h_last_lca = {-1};
    stream << ext_lca.view(numObjs, 1).copy_from(h_last_lca.data()) << synchronize();

    updateBvhExtNodeLinks(numObjs);

    reorderNode(numInternalNodes);
}

template <typename Pred>
void StacklessBVH::Impl::StacklessCDSharedSelf(Pred               pred,
                                               luisa::compute::BufferView<int32_t>  cpNum,
                                               luisa::compute::BufferView<Vector2i> buffer)
{
    using namespace culbvh;
    using namespace luisa::compute;

    auto numQuery = static_cast<int>(ext_aabb.size());
    auto numObjs  = numQuery;

    Kernel1D kernel = [&](
        $int Size,
        BufferVar<AABB>     _box,
        $int intSize,
        $int numObjs,
        BufferVar<int32_t>  _lvs_idx,
        BufferVar<Node>     _nodes,
        BufferVar<int32_t>  resCounter,
        BufferVar<Vector2i> res,
        Pred                pred) noexcept {
        
        set_block_size(K_THREADS, 1u, 1u);
        
        $shared<int2> sharedRes{MAX_RES_PER_BLOCK};
        $shared<int> sharedCounter;
        $shared<int> sharedGlobalIdx;
        
        $if(thread_x() == 0) {
            sharedCounter = 0;
        };
        
        sync_block();
        
        $int tid = dispatch_x();
        $bool active = tid < Size;
        $int idx;
        AABB bv;
        $if(active) {
            idx = _lvs_idx.read(tid);
            bv = _box.read(idx);
        };
        
        $int st = 0;
        Node node;
        const $int MaxIter = numObjs * 2;
        
        $loop {
            sync_block();
            $if(active) {
                $int inner_I = 0;
                $for(inner_I, MaxIter) {
                    $if(st == -1) {
                        $break;
                    };
                    
                    auto node_data = _nodes.read(st);
                    node.lc = node_data.lc;
                    node.escape = node_data.escape;
                    node.bound = node_data.bound;
                    
                    $if(node.bound.intersects(bv)) {
                        $if(node.lc == -1) {
                            $if(tid < st - intSize) {
                                auto pair = make_ordered_pair(idx, _lvs_idx.read(st - intSize));
                                $if(pred(pair.x, pair.y)) {
                                    $int sIdx = sharedCounter.atomic(0).fetch_add(1);
                                    $if(sIdx >= MAX_RES_PER_BLOCK) {
                                        $break;
                                    };
                                    sharedRes[sIdx] = pair;
                                };
                            };
                            st = node.escape;
                        } $else {
                            st = node.lc;
                        };
                    } $else {
                        st = node.escape;
                    };
                };
            };
            
            sync_block();
            $int totalResInBlock = min(sharedCounter, MAX_RES_PER_BLOCK);
            
            $if(thread_x() == 0) {
                sharedGlobalIdx = resCounter.atomic(0).fetch_add(totalResInBlock);
            };
            
            sync_block();
            
            $int globalIdx = sharedGlobalIdx;
            
            $if(thread_x() == 0) {
                sharedCounter = 0;
            };
            
            $bool done = totalResInBlock < MAX_RES_PER_BLOCK;
            
            // SafeCopyTo equivalent
            $if(globalIdx < cast<int>(res.size()) && totalResInBlock > 0) {
                $int CopyCount = min(totalResInBlock, cast<int>(res.size()) - globalIdx);
                $for(i, CopyCount) {
                    $int offset = i;
                    auto pair = sharedRes[offset];
                    res.write(globalIdx + offset, Vector2i{pair.x, pair.y});
                };
            };
            
            $if(done) {
                $break;
            };
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(
        numQuery,
        objs,
        numObjs - 1,
        numObjs,
        ext_idx.view(),
        nodes.view(),
        cpNum,
        buffer,
        pred
    ).dispatch(numQuery);
}

template <typename Pred>
void StacklessBVH::Impl::StacklessCDSharedOther(Pred pred,
                                                luisa::compute::BufferView<AABB>     query_aabbs,
                                                luisa::compute::BufferView<int32_t>  query_sorted_id,
                                                luisa::compute::BufferView<int32_t>  cpNum,
                                                luisa::compute::BufferView<Vector2i> buffer)
{
    using namespace culbvh;
    using namespace luisa::compute;

    auto numQuery = static_cast<int>(query_aabbs.size());
    auto numObjs  = static_cast<int>(ext_aabb.size());

    Kernel1D kernel = [&](
        $int Size,
        BufferVar<AABB>     _box,
        BufferVar<int32_t>  sortedIdx,
        $int intSize,
        $int numObjs,
        BufferVar<int32_t>  _lvs_idx,
        BufferVar<Node>     _nodes,
        BufferVar<int32_t>  resCounter,
        BufferVar<Vector2i> res,
        Pred                pred) noexcept {
        
        set_block_size(K_THREADS, 1u, 1u);
        
        $shared<int2> sharedRes{MAX_RES_PER_BLOCK};
        $shared<int> sharedCounter;
        $shared<int> sharedGlobalIdx;
        
        $if(thread_x() == 0) {
            sharedCounter = 0;
        };
        
        sync_block();
        
        $int tid = dispatch_x();
        $bool active = tid < Size;
        $int idx;
        AABB bv;
        $if(active) {
            idx = sortedIdx.read(tid);
            bv = _box.read(idx);
        };
        
        $int st = 0;
        Node node;
        const $int MaxIter = numObjs * 2;
        
        $loop {
            sync_block();
            $if(active) {
                $int inner_I = 0;
                $for(inner_I, MaxIter) {
                    $if(st == -1) {
                        $break;
                    };
                    
                    auto node_data = _nodes.read(st);
                    node.lc = node_data.lc;
                    node.escape = node_data.escape;
                    node.bound = node_data.bound;
                    
                    $if(node.bound.intersects(bv)) {
                        $if(node.lc == -1) {
                            int2 pair = int2{idx, _lvs_idx.read(st - intSize)};
                            $if(pred(pair.x, pair.y)) {
                                $int sIdx = sharedCounter.atomic(0).fetch_add(1);
                                $if(sIdx >= MAX_RES_PER_BLOCK) {
                                    $break;
                                };
                                sharedRes[sIdx] = pair;
                            };
                            st = node.escape;
                        } $else {
                            st = node.lc;
                        };
                    } $else {
                        st = node.escape;
                    };
                };
            };
            
            sync_block();
            $int totalResInBlock = min(sharedCounter, MAX_RES_PER_BLOCK);
            
            $if(thread_x() == 0) {
                sharedGlobalIdx = resCounter.atomic(0).fetch_add(totalResInBlock);
            };
            
            sync_block();
            
            $int globalIdx = sharedGlobalIdx;
            
            $if(thread_x() == 0) {
                sharedCounter = 0;
            };
            
            sync_block();
            
            $bool done = totalResInBlock < MAX_RES_PER_BLOCK;
            
            // SafeCopyTo equivalent
            $if(globalIdx < cast<int>(res.size()) && totalResInBlock > 0) {
                $int CopyCount = min(totalResInBlock, cast<int>(res.size()) - globalIdx);
                $for(i, CopyCount) {
                    $int offset = i;
                    auto pair = sharedRes[offset];
                    res.write(globalIdx + offset, Vector2i{pair.x, pair.y});
                };
            };
            
            $if(done) {
                $break;
            };
        };
    };

    auto shader = device.compile(kernel);
    stream << shader(
        numQuery,
        query_aabbs,
        query_sorted_id,
        numObjs - 1,
        numObjs,
        ext_idx.view(),
        nodes.view(),
        cpNum,
        buffer,
        pred
    ).dispatch(numQuery);
}

inline void StacklessBVH::build(luisa::compute::BufferView<AABB> aabbs)
{
    m_impl.build(aabbs);
}

template <std::invocable<IndexT, IndexT> Pred>
void StacklessBVH::detect(Pred callback, QueryBuffer& qbuffer)
{
    using namespace luisa::compute;
    
    if(m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    auto do_query = [&]
    {
        // clear counter
        std::vector<int32_t> zero = {0};
        m_impl.stream << qbuffer.m_cpNum.view().copy_from(zero.data()) << synchronize();

        m_impl.StacklessCDSharedSelf(
            callback, qbuffer.m_cpNum.view(), qbuffer.m_pairs.view());
    };

    do_query();

    // get total number of pairs
    std::vector<int32_t> h_cp_num(1);
    m_impl.stream << qbuffer.m_cpNum.view().copy_to(h_cp_num.data()) << synchronize();
    int h_cp_num_val = h_cp_num[0];
    
    // if failed, resize and retry
    if(h_cp_num_val > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.reserve(h_cp_num_val * m_impl.config.reserve_ratio);
        qbuffer.m_pairs = m_impl.device.create_buffer<Vector2i>(h_cp_num_val * m_impl.config.reserve_ratio);
        do_query();
    }

    UIPC_ASSERT(h_cp_num_val >= 0, "fatal error");
    qbuffer.m_size = h_cp_num_val;
}

inline void StacklessBVH::QueryBuffer::build(luisa::compute::BufferView<AABB> aabbs)
{
    using namespace luisa::compute;
    
    auto size = aabbs.size();
    
    if(m_queryMtCode.size() < size) {
        m_queryMtCode = m_device.create_buffer<uint32_t>(size);
        m_querySortedId = m_device.create_buffer<int32_t>(size);
    }

    // Need to access Impl static methods - simplified approach
    // For now, do morton code calculation on host
    std::vector<AABB> h_aabbs(size);
    m_stream << aabbs.copy_to(h_aabbs.data()) << synchronize();
    
    // Calculate scene box
    AABB scene_box;
    for(const auto& box : h_aabbs) {
        scene_box.extend(box);
    }
    
    std::vector<uint32_t> h_mtcode(size);
    std::vector<int32_t> h_sorted_id(size);
    
    for(size_t i = 0; i < size; ++i) {
        auto center = h_aabbs[i].center();
        float3 c = make_float3(center.x(), center.y(), center.z());
        float3 sceneMin = make_float3(scene_box.min().x(), scene_box.min().y(), scene_box.min().z());
        float3 sceneSize = make_float3(scene_box.sizes().x(), scene_box.sizes().y(), scene_box.sizes().z());
        float3 offset = c - sceneMin;
        h_mtcode[i] = culbvh::morton3D(offset.x / sceneSize.x,
                                        offset.y / sceneSize.y,
                                        offset.z / sceneSize.z);
        h_sorted_id[i] = i;
    }
    
    std::sort(h_sorted_id.begin(), h_sorted_id.end(),
              [&](int32_t a, int32_t b) { return h_mtcode[a] < h_mtcode[b]; });
    
    m_stream << m_queryMtCode.view().copy_from(h_mtcode.data())
             << m_querySortedId.view().copy_from(h_sorted_id.data())
             << synchronize();
}

template <std::invocable<IndexT, IndexT> Pred>
void StacklessBVH::query(luisa::compute::BufferView<AABB> aabbs, Pred callback, QueryBuffer& qbuffer)
{
    using namespace luisa::compute;
    
    if(aabbs.size() == 0 || m_impl.objs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    qbuffer.build(aabbs);

    auto do_query = [&]
    {
        // clear counter
        std::vector<int32_t> zero = {0};
        m_impl.stream << qbuffer.m_cpNum.view().copy_from(zero.data()) << synchronize();

        m_impl.StacklessCDSharedOther(callback,
                                      aabbs,
                                      qbuffer.m_querySortedId.view(),
                                      qbuffer.m_cpNum.view(),
                                      qbuffer.m_pairs.view());
    };

    do_query();

    // get total number of pairs
    std::vector<int32_t> h_cp_num(1);
    m_impl.stream << qbuffer.m_cpNum.view().copy_to(h_cp_num.data()) << synchronize();
    int h_cp_num_val = h_cp_num[0];
    
    // if failed, resize and retry
    if(h_cp_num_val > qbuffer.m_pairs.size())
    {
        qbuffer.m_pairs.reserve(h_cp_num_val * m_impl.config.reserve_ratio);
        qbuffer.m_pairs = m_impl.device.create_buffer<Vector2i>(h_cp_num_val * m_impl.config.reserve_ratio);
        do_query();
    }

    UIPC_ASSERT(h_cp_num_val >= 0, "fatal error");
    qbuffer.m_size = h_cp_num_val;
}

// QueryBuffer implementation
inline StacklessBVH::QueryBuffer::QueryBuffer(luisa::compute::Device& device)
    : m_pairs(device.create_buffer<Vector2i>(50 * 1024))
    , m_queryMtCode(device.create_buffer<uint32_t>(1))
    , m_querySceneBox(device.create_buffer<AABB>(1))
    , m_querySortedId(device.create_buffer<int32_t>(1))
    , m_cpNum(device.create_buffer<int32_t>(1))
    , m_device(device)
    , m_stream(device.create_stream())
{
}

}  // namespace uipc::backend::luisa
