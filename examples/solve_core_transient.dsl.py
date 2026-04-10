#
# solve_core_transient
# --------------------
# One transient-analysis timestep for simple
# R / I / V / C / L / VSIN / ISIN / SWPWM / VPWM
# circuits using dense MNA and backward-Euler companion models.
#
# This is the transient solver variant kept alongside `solve_core_dc` so the
# frontend can choose between the smaller DC-only flow and the richer transient
# flow without changing the surrounding backend structure.
#
# The kernel performs one solve at time `par_time` with timestep `par_dt`.
# The previous-step solution is read from `prevX`, and the newly solved state is
# written both to `X` and back into `prevX` so the next invocation can reuse it.
#
# Supported element kinds:
#   1 = resistor           R(node0, node1, resistance)
#   2 = current source     I(node0, node1, current_dc)
#   3 = voltage source     V(node0, node1, voltage_dc)
#   4 = capacitor          C(node0, node1, capacitance)
#   5 = inductor           L(node0, node1, inductance)
#   6 = sinusoidal V src   VSIN(node0, node1, offset, amp, omega, phase)
#   7 = sinusoidal I src   ISIN(node0, node1, offset, amp, omega, phase)
#   8 = PWM switch         SWPWM(node0, node1, ron, roff, period, duty)
#   9 = PWM V source       VPWM(node0, node1, low, high, period, duty)
#
# The raw netlist is exposed through:
#   fetchElemKind(idx)
#   fetchElemN0(idx)
#   fetchElemN1(idx)
#   fetchElemVal0(idx)
#   fetchElemVal1(idx)
#   fetchElemVal2(idx)
#   fetchElemVal3(idx)
#
# Value field usage:
#   R / I / V / C / L : val0 only
#   VSIN / ISIN       : val0=offset, val1=amplitude, val2=omega, val3=phase
#   SWPWM             : val0=ron,    val1=roff,      val2=period, val3=duty
#   VPWM              : val0=low,    val1=high,      val2=period, val3=duty
#
# Unknown ordering:
# - node voltages first, for circuit nodes 1..par_node_n
# - extra branch-current unknowns after that, allocated internally for:
#   * DC voltage sources
#   * sinusoidal voltage sources
#   * inductors
#   * PWM voltage sources
#
# Numerical model:
# - dense real-valued MNA
# - partial row pivoting
# - backward Euler companion models for C and L
# - no explicit singularity / convergence status output
#
def solve_core_transient(par_elem_n, par_node_n, par_dt, par_time):
    f32_0 = 0
    f32_1 = 0
    f32_2 = 0
    f32_3 = 0
    f32_4 = 0
    f32_5 = 1
    f32_6 = 0
    f32_7 = 0
    u8_kind = 0
    u16_aux = 0
    u16_dim = 0
    u16_e = 0
    u16_i = 0
    u16_j = 0
    u16_k = 0
    u16_m = 0
    u16_n0 = 0
    u16_n1 = 0
    u16_next_aux = 0
    u16_pivot = 0
    u8_gate = 0

    # Count the final MNA dimension for this transient solve.
    u16_dim = par_node_n
    for u16_e in range(par_elem_n):
        u8_kind = fetchElemKind(idx=u16_e)
        if u8_kind == 3:
            u16_dim = u16_dim + 1
        if u8_kind == 5:
            u16_dim = u16_dim + 1
        if u8_kind == 6:
            u16_dim = u16_dim + 1
        if u8_kind == 9:
            u16_dim = u16_dim + 1

    # Clear the step-local working memories. prevX is intentionally preserved
    # because it carries state/history from the previous timestep.
    for u16_i in range(u16_dim):
        store_J(i=u16_i, v=f32_0)
        store_Y(i=u16_i, v=f32_0)
        store_X(i=u16_i, v=f32_0)
        for u16_j in range(u16_dim):
            store_A(i=u16_i, j=u16_j, v=f32_0)
            store_LU(i=u16_i, j=u16_j, v=f32_0)

    # Stamp the raw netlist into A and J using backward-Euler companion models.
    u16_next_aux = par_node_n
    for u16_e in range(par_elem_n):
        u8_kind = fetchElemKind(idx=u16_e)
        u16_n0 = fetchElemN0(idx=u16_e)
        u16_n1 = fetchElemN1(idx=u16_e)
        f32_1 = fetchElemVal0(idx=u16_e)
        f32_2 = fetchElemVal1(idx=u16_e)
        f32_3 = fetchElemVal2(idx=u16_e)
        f32_4 = fetchElemVal3(idx=u16_e)

        if u16_n0 != 0:
            u16_n0 = u16_n0 - 1
        else:
            u16_n0 = 65535

        if u16_n1 != 0:
            u16_n1 = u16_n1 - 1
        else:
            u16_n1 = 65535

        if u8_kind == 1:
            f32_6 = div(a=f32_5, b=f32_1)
            if u16_n0 != 65535:
                accumA(i=u16_n0, j=u16_n0, delta=f32_6)
                if u16_n1 != 65535:
                    f32_7 = neg_comb(v=f32_6)
                    accumA(i=u16_n0, j=u16_n1, delta=f32_7)
                    accumA(i=u16_n1, j=u16_n0, delta=f32_7)
            if u16_n1 != 65535:
                accumA(i=u16_n1, j=u16_n1, delta=f32_6)

        if u8_kind == 2:
            if u16_n0 != 65535:
                f32_6 = neg_comb(v=f32_1)
                accumJ(i=u16_n0, delta=f32_6)
            if u16_n1 != 65535:
                accumJ(i=u16_n1, delta=f32_1)

        if u8_kind == 3:
            u16_aux = u16_next_aux
            u16_next_aux = u16_next_aux + 1
            if u16_n0 != 65535:
                accumA(i=u16_aux, j=u16_n0, delta=f32_5)
                f32_6 = neg_comb(v=f32_5)
                accumA(i=u16_n0, j=u16_aux, delta=f32_6)
            if u16_n1 != 65535:
                f32_6 = neg_comb(v=f32_5)
                accumA(i=u16_aux, j=u16_n1, delta=f32_6)
                accumA(i=u16_n1, j=u16_aux, delta=f32_5)
            accumJ(i=u16_aux, delta=f32_1)

        # Capacitor backward-Euler companion:
        #   g = C / dt
        #   i_hist = g * v_prev
        # which becomes a resistor-like stamp plus an equivalent RHS term.
        if u8_kind == 4:
            f32_6 = div(a=f32_1, b=par_dt)
            f32_7 = f32_0
            if u16_n0 != 65535:
                f32_7 = fetch_prevX(i=u16_n0)
            if u16_n1 != 65535:
                f32_2 = fetch_prevX(i=u16_n1)
                f32_2 = neg_comb(v=f32_2)
                f32_7 = fma(a=f32_5, b=f32_2, c=f32_7)
            if u16_n0 != 65535:
                accumA(i=u16_n0, j=u16_n0, delta=f32_6)
                if u16_n1 != 65535:
                    f32_2 = neg_comb(v=f32_6)
                    accumA(i=u16_n0, j=u16_n1, delta=f32_2)
                    accumA(i=u16_n1, j=u16_n0, delta=f32_2)
            if u16_n1 != 65535:
                accumA(i=u16_n1, j=u16_n1, delta=f32_6)
            f32_2 = fma(a=f32_6, b=f32_7, c=f32_0)
            if u16_n0 != 65535:
                accumJ(i=u16_n0, delta=f32_2)
            if u16_n1 != 65535:
                f32_3 = neg_comb(v=f32_2)
                accumJ(i=u16_n1, delta=f32_3)

        # Inductor backward-Euler companion with a branch-current unknown.
        if u8_kind == 5:
            u16_aux = u16_next_aux
            u16_next_aux = u16_next_aux + 1
            f32_6 = div(a=f32_1, b=par_dt)
            f32_7 = fetch_prevX(i=u16_aux)
            if u16_n0 != 65535:
                accumA(i=u16_aux, j=u16_n0, delta=f32_5)
                f32_2 = neg_comb(v=f32_5)
                accumA(i=u16_n0, j=u16_aux, delta=f32_2)
            if u16_n1 != 65535:
                f32_2 = neg_comb(v=f32_5)
                accumA(i=u16_aux, j=u16_n1, delta=f32_2)
                accumA(i=u16_n1, j=u16_aux, delta=f32_5)
            f32_2 = neg_comb(v=f32_6)
            accumA(i=u16_aux, j=u16_aux, delta=f32_2)
            f32_7 = neg_comb(v=f32_7)
            f32_2 = fma(a=f32_6, b=f32_7, c=f32_0)
            accumJ(i=u16_aux, delta=f32_2)

        if u8_kind == 6:
            u16_aux = u16_next_aux
            u16_next_aux = u16_next_aux + 1
            f32_6 = fma(a=f32_3, b=par_time, c=f32_4)
            f32_6 = sin_comb(v=f32_6)
            f32_6 = fma(a=f32_2, b=f32_6, c=f32_1)
            if u16_n0 != 65535:
                accumA(i=u16_aux, j=u16_n0, delta=f32_5)
                f32_7 = neg_comb(v=f32_5)
                accumA(i=u16_n0, j=u16_aux, delta=f32_7)
            if u16_n1 != 65535:
                f32_7 = neg_comb(v=f32_5)
                accumA(i=u16_aux, j=u16_n1, delta=f32_7)
                accumA(i=u16_n1, j=u16_aux, delta=f32_5)
            accumJ(i=u16_aux, delta=f32_6)

        if u8_kind == 7:
            f32_6 = fma(a=f32_3, b=par_time, c=f32_4)
            f32_6 = sin_comb(v=f32_6)
            f32_6 = fma(a=f32_2, b=f32_6, c=f32_1)
            if u16_n0 != 65535:
                f32_7 = neg_comb(v=f32_6)
                accumJ(i=u16_n0, delta=f32_7)
            if u16_n1 != 65535:
                accumJ(i=u16_n1, delta=f32_6)

        # PWM-gated ideal switch. This is stamped as a resistor whose value
        # toggles between ron and roff according to the current PWM phase.
        if u8_kind == 8:
            u8_gate = pwm_gate_comb(time=par_time, period=f32_3, duty=f32_4)
            f32_6 = f32_2
            if u8_gate != 0:
                f32_6 = f32_1
            f32_6 = div(a=f32_5, b=f32_6)
            if u16_n0 != 65535:
                accumA(i=u16_n0, j=u16_n0, delta=f32_6)
                if u16_n1 != 65535:
                    f32_7 = neg_comb(v=f32_6)
                    accumA(i=u16_n0, j=u16_n1, delta=f32_7)
                    accumA(i=u16_n1, j=u16_n0, delta=f32_7)
            if u16_n1 != 65535:
                accumA(i=u16_n1, j=u16_n1, delta=f32_6)

        # PWM square-wave voltage source. The branch-variable structure matches
        # the ordinary V / VSIN source, but the source value is piecewise
        # constant over each PWM period.
        if u8_kind == 9:
            u8_gate = pwm_gate_comb(time=par_time, period=f32_3, duty=f32_4)
            f32_6 = f32_1
            if u8_gate != 0:
                f32_6 = f32_2
            u16_aux = u16_next_aux
            u16_next_aux = u16_next_aux + 1
            if u16_n0 != 65535:
                accumA(i=u16_aux, j=u16_n0, delta=f32_5)
                f32_7 = neg_comb(v=f32_5)
                accumA(i=u16_n0, j=u16_aux, delta=f32_7)
            if u16_n1 != 65535:
                f32_7 = neg_comb(v=f32_5)
                accumA(i=u16_aux, j=u16_n1, delta=f32_7)
                accumA(i=u16_n1, j=u16_aux, delta=f32_5)
            accumJ(i=u16_aux, delta=f32_6)

    # Dense LU factorization with partial row pivoting.
    for u16_j in range(u16_dim):
        u16_pivot = u16_j
        f32_1 = fetch_A(i=u16_j, j=u16_j)
        f32_1 = neg_comb(v=f32_1)
        for u16_k in range(u16_j):
            f32_2 = fetch_LU(i=u16_j, j=u16_k)
            f32_3 = fetch_LU(i=u16_k, j=u16_j)
            f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
        f32_1 = neg_comb(v=f32_1)
        f32_4 = abs_comb(v=f32_1)
        u16_i = u16_j + 1
        while u16_i < u16_dim:
            f32_1 = fetch_A(i=u16_i, j=u16_j)
            f32_1 = neg_comb(v=f32_1)
            for u16_k in range(u16_j):
                f32_2 = fetch_LU(i=u16_i, j=u16_k)
                f32_3 = fetch_LU(i=u16_k, j=u16_j)
                f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
            f32_1 = neg_comb(v=f32_1)
            f32_1 = abs_comb(v=f32_1)
            if gt_comb(a=f32_1, b=f32_4):
                f32_4 = f32_1
                u16_pivot = u16_i
            u16_i = u16_i + 1

        if u16_pivot != u16_j:
            for u16_k in range(u16_dim):
                f32_1 = fetch_A(i=u16_j, j=u16_k)
                f32_2 = fetch_A(i=u16_pivot, j=u16_k)
                store_A(i=u16_j, j=u16_k, v=f32_2)
                store_A(i=u16_pivot, j=u16_k, v=f32_1)
            for u16_k in range(u16_j):
                f32_1 = fetch_LU(i=u16_j, j=u16_k)
                f32_2 = fetch_LU(i=u16_pivot, j=u16_k)
                store_LU(i=u16_j, j=u16_k, v=f32_2)
                store_LU(i=u16_pivot, j=u16_k, v=f32_1)
            f32_1 = fetch_J(i=u16_j)
            f32_2 = fetch_J(i=u16_pivot)
            store_J(i=u16_j, v=f32_2)
            store_J(i=u16_pivot, v=f32_1)

        for u16_i in range(u16_dim):
            f32_1 = fetch_A(i=u16_i, j=u16_j)
            f32_1 = neg_comb(v=f32_1)
            u16_m = u16_j if u16_i > u16_j else u16_i
            for u16_k in range(u16_m):
                f32_2 = fetch_LU(i=u16_i, j=u16_k)
                f32_3 = fetch_LU(i=u16_k, j=u16_j)
                f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
            if u16_i > u16_j:
                f32_2 = fetch_LU(i=u16_j, j=u16_j)
                f32_1 = div(a=f32_1, b=f32_2)
            f32_1 = neg_comb(v=f32_1)
            store_LU(i=u16_i, j=u16_j, v=f32_1)

    # Forward substitution: solve L * Y = J.
    for u16_i in range(u16_dim):
        f32_1 = fetch_J(i=u16_i)
        for u16_k in range(u16_i):
            f32_2 = fetch_LU(i=u16_i, j=u16_k)
            f32_3 = fetch_Y(i=u16_k)
            f32_3 = neg_comb(v=f32_3)
            f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
        store_Y(i=u16_i, v=f32_1)

    # Backward substitution: solve U * X = Y.
    u16_i = u16_dim
    while u16_i > 0:
        u16_i = u16_i - 1
        f32_1 = fetch_Y(i=u16_i)
        u16_k = u16_i + 1
        while u16_k < u16_dim:
            f32_2 = fetch_LU(i=u16_i, j=u16_k)
            f32_3 = fetch_X(i=u16_k)
            f32_3 = neg_comb(v=f32_3)
            f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
            u16_k = u16_k + 1
        f32_2 = fetch_LU(i=u16_i, j=u16_i)
        f32_1 = div(a=f32_1, b=f32_2)
        store_X(i=u16_i, v=f32_1)

    # Commit the newly solved state as the history vector for the next step.
    for u16_i in range(u16_dim):
        f32_1 = fetch_X(i=u16_i)
        store_prevX(i=u16_i, v=f32_1)
