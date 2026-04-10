def forward_sub_core(par_n):
    f32_1 = 0
    f32_2 = 0
    f32_3 = 0
    u8_i = 0
    u8_k = 0
    for u8_i in range(par_n):
        f32_1 = fetch_J(i=u8_i)
        for u8_k in range(u8_i):
            f32_2 = fetch_LU(i=u8_i, j=u8_k)
            f32_3 = fetch_Y(i=u8_k)
            f32_3 = neg_comb(v=f32_3)
            f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
        store_Y(i=u8_i, v=f32_1)
