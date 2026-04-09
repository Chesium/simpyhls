def stamping_core(par_elem_n):
    u16_e = 0
    u8_kind = 0
    u16_n0 = 0
    u16_n1 = 0
    u16_n2 = 0
    u16_n3 = 0
    u16_aux = 0
    f32_v0 = 0
    f32_v1 = 0
    f32_v2 = 0
    f32_neg = 0
    for u16_e in range(par_elem_n):
        u8_kind = fetchElemKind(idx=u16_e)
        u16_n0 = fetchElemN0(idx=u16_e)
        u16_n1 = fetchElemN1(idx=u16_e)
        u16_n2 = fetchElemN2(idx=u16_e)
        u16_n3 = fetchElemN3(idx=u16_e)
        u16_aux = fetchElemAux(idx=u16_e)
        f32_v0 = fetchElemVal0(idx=u16_e)
        f32_v1 = fetchElemVal1(idx=u16_e)
        f32_v2 = fetchElemVal2(idx=u16_e)

        if u8_kind == 1:
            if u16_n0 != 65535:
                accumA(i=u16_n0, j=u16_n0, delta=f32_v0)
                if u16_n1 != 65535:
                    f32_neg = neg_comb(v=f32_v0)
                    accumA(i=u16_n0, j=u16_n1, delta=f32_neg)
                    accumA(i=u16_n1, j=u16_n0, delta=f32_neg)
            if u16_n1 != 65535:
                accumA(i=u16_n1, j=u16_n1, delta=f32_v0)

        if u8_kind == 2:
            if u16_n0 != 65535:
                f32_neg = neg_comb(v=f32_v0)
                accumJ(i=u16_n0, delta=f32_neg)
            if u16_n1 != 65535:
                accumJ(i=u16_n1, delta=f32_v0)

        if u8_kind == 3:
            if u16_aux != 65535:
                if u16_n0 != 65535:
                    accumA(i=u16_aux, j=u16_n0, delta=f32_v2)
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u16_n0, j=u16_aux, delta=f32_neg)
                if u16_n1 != 65535:
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u16_aux, j=u16_n1, delta=f32_neg)
                    accumA(i=u16_n1, j=u16_aux, delta=f32_v2)
                accumJ(i=u16_aux, delta=f32_v0)

        if u8_kind == 4:
            if u16_aux != 65535:
                if u16_n2 != 65535:
                    accumA(i=u16_aux, j=u16_n2, delta=f32_v2)
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u16_n2, j=u16_aux, delta=f32_neg)
                if u16_n3 != 65535:
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u16_aux, j=u16_n3, delta=f32_neg)
                    accumA(i=u16_n3, j=u16_aux, delta=f32_v2)
                if u16_n0 != 65535:
                    f32_neg = neg_comb(v=f32_v0)
                    accumA(i=u16_aux, j=u16_n0, delta=f32_neg)
                if u16_n1 != 65535:
                    accumA(i=u16_aux, j=u16_n1, delta=f32_v0)

        if u8_kind == 5:
            if u16_aux != 65535:
                if u16_n0 != 65535:
                    accumA(i=u16_aux, j=u16_n0, delta=f32_v2)
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u16_n0, j=u16_aux, delta=f32_neg)
                if u16_n2 != 65535:
                    f32_neg = neg_comb(v=f32_v2)
                    accumA(i=u16_aux, j=u16_n2, delta=f32_neg)
                    accumA(i=u16_n2, j=u16_aux, delta=f32_v2)
                    accumA(i=u16_n2, j=u16_aux, delta=f32_v0)
                if u16_n1 != 65535:
                    f32_neg = neg_comb(v=f32_v0)
                    accumA(i=u16_n1, j=u16_aux, delta=f32_neg)
                accumJ(i=u16_aux, delta=f32_v1)
