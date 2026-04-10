def backward_sub_core(par_n):
    f32_1 = 0
    f32_2 = 0
    f32_3 = 0
    u8_i = 0
    u8_k = 0
    u8_i = par_n
    while u8_i > 0:
        u8_i = u8_i - 1
        f32_1 = fetch_Y(i=u8_i)
        u8_k = u8_i + 1
        while u8_k < par_n:
            f32_2 = fetch_LU(i=u8_i, j=u8_k)
            f32_3 = fetch_X(i=u8_k)
            f32_3 = neg_comb(v=f32_3)
            f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
            u8_k = u8_k + 1
        f32_2 = fetch_LU(i=u8_i, j=u8_i)
        f32_1 = div(a=f32_1, b=f32_2)
        store_X(i=u8_i, v=f32_1)
