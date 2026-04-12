def stamping_core(par_elem_n):
    u16_e = 0
    u8_kind = 0
    u8_n0 = 0
    u8_n1 = 0
    u8_n2 = 0
    u8_n3 = 0
    u8_aux = 0
    f32_v0 = 0
    f32_v1 = 0
    f32_v2 = 0
    f32_neg = 0
    for u16_e in range(par_elem_n):
        u8_kind = fetchElemKind(idx=u16_e)
        u8_n0 = fetchElemN0(idx=u16_e)
        u8_n1 = fetchElemN1(idx=u16_e)
        u8_n2 = fetchElemN2(idx=u16_e)
        u8_n3 = fetchElemN3(idx=u16_e)
        u8_aux = fetchElemAux(idx=u16_e)
        f32_v0 = fetchElemVal0(idx=u16_e)
        f32_v1 = fetchElemVal1(idx=u16_e)
        f32_v2 = fetchElemVal2(idx=u16_e)

        if u8_kind == 1:
            if u8_n0 != 255:
                accumA(i=u8_n0, j=u8_n0, delta=f32_v0)
                if u8_n1 != 255:
                    f32_neg = neg_comb(v=f32_v0)
                    accumA(i=u8_n0, j=u8_n1, delta=f32_neg)
                    accumA(i=u8_n1, j=u8_n0, delta=f32_neg)
            if u8_n1 != 255:
                accumA(i=u8_n1, j=u8_n1, delta=f32_v0)

        if u8_kind == 2:
            if u8_n0 != 255:
                f32_neg = neg_comb(v=f32_v0)
                accumJ(i=u8_n0, delta=f32_neg)
            if u8_n1 != 255:
                accumJ(i=u8_n1, delta=f32_v0)

        if u8_kind == 3:
            if u8_aux != 255:
                if u8_n0 != 255:
                    accumA(i=u8_aux, j=u8_n0, delta=f32_v2)
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u8_n0, j=u8_aux, delta=f32_neg)
                if u8_n1 != 255:
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u8_aux, j=u8_n1, delta=f32_neg)
                    accumA(i=u8_n1, j=u8_aux, delta=f32_v2)
                accumJ(i=u8_aux, delta=f32_v0)

        if u8_kind == 4:
            if u8_aux != 255:
                if u8_n2 != 255:
                    accumA(i=u8_aux, j=u8_n2, delta=f32_v2)
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u8_n2, j=u8_aux, delta=f32_neg)
                if u8_n3 != 255:
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u8_aux, j=u8_n3, delta=f32_neg)
                    accumA(i=u8_n3, j=u8_aux, delta=f32_v2)
                if u8_n0 != 255:
                    f32_neg = neg_comb(v=f32_v0)
                    accumA(i=u8_aux, j=u8_n0, delta=f32_neg)
                if u8_n1 != 255:
                    accumA(i=u8_aux, j=u8_n1, delta=f32_v0)

        if u8_kind == 5:
            if u8_aux != 255:
                if u8_n0 != 255:
                    accumA(i=u8_aux, j=u8_n0, delta=f32_v2)
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u8_n0, j=u8_aux, delta=f32_neg)
                if u8_n2 != 255:
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u8_aux, j=u8_n2, delta=f32_neg)
                    accumA(i=u8_n2, j=u8_aux, delta=f32_v2)
                    accumA(i=u8_n2, j=u8_aux, delta=f32_v0)
                if u8_n1 != 255:
                    f32_neg = neg_comb(v=f32_v0)
                    accumA(i=u8_n1, j=u8_aux, delta=f32_neg)
                accumJ(i=u8_aux, delta=f32_v1)
