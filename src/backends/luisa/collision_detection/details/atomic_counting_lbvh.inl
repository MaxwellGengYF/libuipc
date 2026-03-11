namespace uipc::backend::luisa
{
inline void AtomicCountingLBVH::QueryBuffer::reserve(size_t size, luisa::compute::Device& device)
{
    if(size > m_pairs.size()) {
        m_pairs = device.create_buffer<Vector2i>(size);
    }
}

inline AtomicCountingLBVH::AtomicCountingLBVH(luisa::compute::Device& device, luisa::compute::Stream& stream) noexcept
    : m_device(device)
    , m_stream(stream)
    , m_cp_num(device.create_buffer<IndexT>(1))
    , m_lbvh(device, stream)
{
}

inline void AtomicCountingLBVH::build(luisa::compute::BufferView<LinearBVHAABB> aabbs)
{
    m_aabbs = aabbs;
    m_lbvh.build(aabbs);
}

template <typename Pred>
void AtomicCountingLBVH::detect(Pred p, QueryBuffer& qbuffer)
{
    using namespace luisa::compute;

    if(m_aabbs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    auto do_query = [&]
    {
        // Reset counter to 0
        IndexT zero = 0;
        m_stream << m_cp_num.copy_from(&zero) << commit();

        // Create kernel for LBVH detection
        Kernel1D detect_kernel = [&](BufferView<LinearBVHAABB> aabbs, 
                                      BufferView<LinearBVHNode> nodes,
                                      BufferView<LinearBVHAABB> node_aabbs,
                                      BufferView<IndexT> cp_num,
                                      BufferView<Vector2i> pairs) {
            auto i = dispatch_id().x;
            if(i >= aabbs.size()) return;
            
            LinearBVHViewer lbvh_viewer(nodes.size(), aabbs.size(), nodes, node_aabbs);
            auto aabb = aabbs.read(i);
            
            lbvh_viewer.query(aabb, [&](uint32_t id) {
                if(id > i && p(i, id))
                {
                    auto last = cp_num.atomic(0).fetch_add(1);
                    if(last < pairs.size())
                        pairs.write(last, Vector2i(i, id));
                }
            });
        };

        auto shader = m_device.compile(detect_kernel);
        
        m_stream << shader(m_aabbs, 
                           m_lbvh.nodes_view(), 
                           m_lbvh.aabbs_view(),
                           m_cp_num.view(),
                           qbuffer.m_pairs.view()).dispatch(m_aabbs.size())
                 << commit();
    };

    do_query();

    // get total number of pairs
    IndexT h_cp_num;
    m_stream << m_cp_num.copy_to(&h_cp_num) << synchronize();
    
    // if failed, resize and retry
    if(h_cp_num > (IndexT)qbuffer.m_pairs.size())
    {
        qbuffer.reserve(h_cp_num * m_reserve_ratio, m_device);
        do_query();
    }

    qbuffer.m_size = h_cp_num;
}

template <typename Pred>
void AtomicCountingLBVH::query(luisa::compute::BufferView<LinearBVHAABB> query_aabbs, Pred p, QueryBuffer& qbuffer)
{
    using namespace luisa::compute;

    if(m_aabbs.size() == 0 || query_aabbs.size() == 0)
    {
        qbuffer.m_size = 0;
        return;
    }

    auto do_query = [&]
    {
        // Reset counter to 0
        IndexT zero = 0;
        m_stream << m_cp_num.copy_from(&zero) << commit();

        // Create kernel for LBVH query
        Kernel1D query_kernel = [&](BufferView<LinearBVHAABB> query_aabbs,
                                     BufferView<LinearBVHNode> nodes,
                                     BufferView<LinearBVHAABB> node_aabbs,
                                     BufferView<IndexT> cp_num,
                                     BufferView<Vector2i> pairs) {
            auto i = dispatch_id().x;
            if(i >= query_aabbs.size()) return;
            
            LinearBVHViewer lbvh_viewer(nodes.size(), m_aabbs.size(), nodes, node_aabbs);
            auto aabb = query_aabbs.read(i);
            
            lbvh_viewer.query(aabb, [&](uint32_t id) {
                if(p(i, id))
                {
                    auto last = cp_num.atomic(0).fetch_add(1);
                    if(last < pairs.size())
                        pairs.write(last, Vector2i(i, id));
                }
            });
        };

        auto shader = m_device.compile(query_kernel);
        
        m_stream << shader(query_aabbs, 
                           m_lbvh.nodes_view(), 
                           m_lbvh.aabbs_view(),
                           m_cp_num.view(),
                           qbuffer.m_pairs.view()).dispatch(query_aabbs.size())
                 << commit();
    };

    do_query();

    // get total number of pairs
    IndexT h_cp_num;
    m_stream << m_cp_num.copy_to(&h_cp_num) << synchronize();
    
    // if failed, resize and retry
    if(h_cp_num > (IndexT)qbuffer.size())
    {
        qbuffer.reserve(h_cp_num * m_reserve_ratio, m_device);
        do_query();
    }

    qbuffer.m_size = h_cp_num;
}
}  // namespace uipc::backend::luisa
