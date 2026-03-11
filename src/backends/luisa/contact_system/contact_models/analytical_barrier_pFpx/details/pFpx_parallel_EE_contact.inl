namespace uipc::backend::luisa::analyticalBarrier
{
namespace details
{
    inline void pFpx_parallel_ee(Float   d,
                                 Float   x11,
                                 Float   x12,
                                 Float   x13,
                                 Float   x21,
                                 Float   x22,
                                 Float   x23,
                                 Float   x31,
                                 Float   x32,
                                 Float   x33,
                                 Float   x41,
                                 Float   x42,
                                 Float   x43,
                                 Float   result[12][9]) noexcept
    {
        Float t12;
        Float t13;
        Float t14;
        Float t15;
        Float t16;
        Float t17;
        Float t18;
        Float t19;
        Float t20;
        Float t33;
        Float t34;
        Float t35;
        Float t36;
        Float t37;
        Float t38;
        Float t45;
        Float t46;
        Float t47;
        Float t69;
        Float t70;
        Float t71;
        Float t72;
        Float t73;
        Float t74;
        Float t76;
        Float t77;
        Float t79;
        Float t80;
        Float t81;
        Float t82;
        Float t83;
        Float t84;
        Float t85;
        Float t86;
        Float t89;
        Float t90;

        t12  = -x21 + x11;
        t13  = -x22 + x12;
        t14  = -x23 + x13;
        t15  = -x31 + x11;
        t16  = -x32 + x12;
        t17  = -x33 + x13;
        t18  = -x41 + x31;
        t19  = -x42 + x32;
        t20  = -x43 + x33;
        t33  = t15 * t19;
        t34  = t16 * t18;
        t35  = t15 * t20;
        t36  = t17 * t18;
        t37  = t16 * t20;
        t38  = t17 * t19;
        t45  = t12 * t19 + -(t13 * t18);
        t46  = t12 * t20 + -(t14 * t18);
        t47  = t13 * t20 + -(t14 * t19);
        t69  = t13 * t45 * 2.0f + t14 * t46 * 2.0f;
        t70  = t12 * t46 * 2.0f + t13 * t47 * 2.0f;
        t71  = t19 * t45 * 2.0f + t20 * t46 * 2.0f;
        t72  = t18 * t46 * 2.0f + t19 * t47 * 2.0f;
        t19  = (t45 * t45 + t46 * t46) + t47 * t47;
        t73  = t12 * t45 * 2.0f + -(t14 * t47 * 2.0f);
        t74  = t18 * t45 * 2.0f + -(t20 * t47 * 2.0f);
        t76  = 1.0f / t19;
        t19  = 1.0f / sqrt(t19);
        t79  = (t17 * t45 + t15 * t47) + -(t16 * t46);
        t77  = t76 * t76;
        t80  = t79 * t79;
        t81  = t69 * t19 / 2.0f;
        t82  = t70 * t19 / 2.0f;
        t83  = t71 * t19 / 2.0f;
        t84  = t72 * t19 / 2.0f;
        t85  = t73 * t19 / 2.0f;
        t86  = t74 * t19 / 2.0f;
        t89  = t69 * t77 * t80;
        t90  = t70 * t77 * t80;
        t20  = t71 * t77 * t80;
        t18  = t72 * t77 * t80;
        t71  = t73 * t77 * t80;
        t19  = t74 * t77 * t80;

        result[0][0]  = 0.0f;
        result[0][1]  = 0.0f;
        result[0][2]  = 0.0f;
        result[0][3]  = 0.0f;
        result[0][4]  = t83;
        result[0][5]  = 0.0f;
        result[0][6]  = 0.0f;
        result[0][7]  = 0.0f;
        t70           = t76 * t79;
        t69           = 1.0f / d * (1.0f / sqrt(t76 * t80));
        result[0][8]  = t69 * (t20 - t70 * ((-t37 + t38) + t47) * 2.0f) * -0.5f;
        result[1][0]  = 0.0f;
        result[1][1]  = 0.0f;
        result[1][2]  = 0.0f;
        result[1][3]  = 0.0f;
        result[1][4]  = -t86;
        result[1][5]  = 0.0f;
        result[1][6]  = 0.0f;
        result[1][7]  = 0.0f;
        result[1][8]  = t69 * (t19 - t70 * ((-t35 + t36) + t46) * 2.0f) / 2.0f;
        result[2][0]  = 0.0f;
        result[2][1]  = 0.0f;
        result[2][2]  = 0.0f;
        result[2][3]  = 0.0f;
        result[2][4]  = -t84;
        result[2][5]  = 0.0f;
        result[2][6]  = 0.0f;
        result[2][7]  = 0.0f;
        result[2][8]  = t69 * (t18 + t70 * ((-t33 + t34) + t45) * 2.0f) / 2.0f;
        result[3][0]  = 0.0f;
        result[3][1]  = 0.0f;
        result[3][2]  = 0.0f;
        result[3][3]  = 0.0f;
        result[3][4]  = -t83;
        result[3][5]  = 0.0f;
        result[3][6]  = 0.0f;
        result[3][7]  = 0.0f;
        result[3][8]  = t69 * (t20 + t70 * (t37 - t38) * 2.0f) / 2.0f;
        result[4][0]  = 0.0f;
        result[4][1]  = 0.0f;
        result[4][2]  = 0.0f;
        result[4][3]  = 0.0f;
        result[4][4]  = t86;
        result[4][5]  = 0.0f;
        result[4][6]  = 0.0f;
        result[4][7]  = 0.0f;
        result[4][8]  = t69 * (t19 + t70 * (t35 - t36) * 2.0f) * -0.5f;
        result[5][0]  = 0.0f;
        result[5][1]  = 0.0f;
        result[5][2]  = 0.0f;
        result[5][3]  = 0.0f;
        result[5][4]  = t84;
        result[5][5]  = 0.0f;
        result[5][6]  = 0.0f;
        result[5][7]  = 0.0f;
        result[5][8]  = t69 * (t18 - t70 * (t33 - t34) * 2.0f) * -0.5f;
        result[6][0]  = 0.0f;
        result[6][1]  = 0.0f;
        result[6][2]  = 0.0f;
        result[6][3]  = 0.0f;
        result[6][4]  = -t81;
        result[6][5]  = 0.0f;
        result[6][6]  = 0.0f;
        result[6][7]  = 0.0f;
        t20           = t13 * t17 + -(t14 * t16);
        result[6][8]  = t69 * (t89 - t70 * (t20 + t47) * 2.0f) / 2.0f;
        result[7][0]  = 0.0f;
        result[7][1]  = 0.0f;
        result[7][2]  = 0.0f;
        result[7][3]  = 0.0f;
        result[7][4]  = t85;
        result[7][5]  = 0.0f;
        result[7][6]  = 0.0f;
        result[7][7]  = 0.0f;
        t18           = t12 * t17 + -(t14 * t15);
        result[7][8]  = t69 * (t71 - t70 * (t18 + t46) * 2.0f) * -0.5f;
        result[8][0]  = 0.0f;
        result[8][1]  = 0.0f;
        result[8][2]  = 0.0f;
        result[8][3]  = 0.0f;
        result[8][4]  = t82;
        result[8][5]  = 0.0f;
        result[8][6]  = 0.0f;
        result[8][7]  = 0.0f;
        t19           = t12 * t16 + -(t13 * t15);
        result[8][8]  = t69 * (t90 + t70 * (t19 + t45) * 2.0f) * -0.5f;
        result[9][0]  = 0.0f;
        result[9][1]  = 0.0f;
        result[9][2]  = 0.0f;
        result[9][3]  = 0.0f;
        result[9][4]  = t81;
        result[9][5]  = 0.0f;
        result[9][6]  = 0.0f;
        result[9][7]  = 0.0f;
        result[9][8]  = t69 * (t89 - t70 * t20 * 2.0f) * -0.5f;
        result[10][0] = 0.0f;
        result[10][1] = 0.0f;
        result[10][2] = 0.0f;
        result[10][3] = 0.0f;
        result[10][4] = -t85;
        result[10][5] = 0.0f;
        result[10][6] = 0.0f;
        result[10][7] = 0.0f;
        result[10][8] = t69 * (t71 - t70 * t18 * 2.0f) / 2.0f;
        result[11][0] = 0.0f;
        result[11][1] = 0.0f;
        result[11][2] = 0.0f;
        result[11][3] = 0.0f;
        result[11][4] = -t82;
        result[11][5] = 0.0f;
        result[11][6] = 0.0f;
        result[11][7] = 0.0f;
        result[11][8] = t69 * (t90 + t70 * t19 * 2.0f) / 2.0f;
    }

}  // namespace details

inline void analytical_parallel_edge_edge_pFpx(Expr<float3> e0,
                                               Expr<float3> e1,
                                               Expr<float3> e2,
                                               Expr<float3> e3,
                                               Expr<float>  d_hatSqrt,
                                               Float        result[12][9]) noexcept
{
    details::pFpx_parallel_ee(d_hatSqrt,
                              e0.x,
                              e0.y,
                              e0.z,
                              e1.x,
                              e1.y,
                              e1.z,
                              e2.x,
                              e2.y,
                              e2.z,
                              e3.x,
                              e3.y,
                              e3.z,
                              result);
}

}  // namespace uipc::backend::luisa::analyticalBarrier
