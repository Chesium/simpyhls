def lu_core(par_n):
    f32_1 = 0
    f32_2 = 0
    f32_3 = 0
    u8_i = 0
    u8_j = 0
    u8_k = 0
    u8_m = 0
    for u8_j in range(par_n):
        for u8_i in range(par_n):
            f32_1 = fetch_A(i=u8_i, j=u8_j)
            f32_1 = neg_comb(v=f32_1)
            u8_m = u8_j if u8_i > u8_j else u8_i
            for u8_k in range(u8_m):
                f32_2 = fetch_LU(i=u8_i, j=u8_k)
                f32_3 = fetch_LU(i=u8_k, j=u8_j)
                f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
            if u8_i > u8_j:
                f32_2 = fetch_LU(i=u8_j, j=u8_j)
                f32_1 = div(a=f32_1, b=f32_2)
            f32_1 = neg_comb(v=f32_1)
            store_LU(i=u8_i, j=u8_j, v=f32_1)
