#
# solve_core
# ----------
# Single-kernel dense MNA solve flow for simple R / I / V netlists.
#
# This DSL function intentionally combines all stages into one HLS-friendly kernel:
# 1. inspect the raw netlist and derive the final matrix dimension
# 2. clear all matrix / vector backing stores
# 3. stamp the netlist into dense A and J memories
# 4. run dense LU decomposition with partial row pivoting
# 5. run forward substitution to solve L * Y = J
# 6. run backward substitution to solve U * X = Y
#
# Netlist contract
# ----------------
# The input netlist is a flat sequence of elements indexed by `idx` in
# [0, par_elem_n). Each element is exposed through blocking fetch primitives:
#
#   fetchElemKind(idx) -> element kind
#   fetchElemN0(idx)   -> node 0 (1-based node id, 0 means ground)
#   fetchElemN1(idx)   -> node 1 (1-based node id, 0 means ground)
#   fetchElemVal0(idx) -> element value
#
# Element kinds supported by this kernel:
#   1 = resistor R(node0, node1, resistance)
#   2 = current source I(node0, node1, current)
#   3 = voltage source V(node0, node1, voltage)
#
# Host-side preprocessing is intentionally minimal in this version:
# - nodes stay as raw circuit node numbers
# - only R / I / V are supported
# - branch-variable allocation for voltage sources happens inside this kernel
#
# Matrix / vector contract
# ------------------------
# A, LU, J, Y, X all live in external backing stores accessed through blocking
# primitives. The final solution is written to X through `store_X`.
#
# Indexing conventions
# --------------------
# - Circuit node 0 is ground and is not represented in the matrix.
# - Circuit nodes 1..par_node_n map to matrix rows/cols 0..par_node_n-1.
# - Ground is converted to the sentinel value 65535 inside the kernel so the
#   stamping code can skip matrix updates touching ground.
# - Additional MNA branch variables for voltage sources are allocated densely
#   after the node-voltage unknowns.
#
# Numerical / algorithmic assumptions
# -----------------------------------
# - This is a dense solver: sparse structure is not exploited.
# - Partial pivoting is row-pivoting only.
# - The kernel assumes the matrix is solvable; there is no explicit error output
#   for singular / ill-conditioned systems.
# - LU is stored in packed form in one matrix:
#     * U occupies the diagonal and upper triangle
#     * L occupies the strict lower triangle
#     * the diagonal of L is implicit 1 and is not stored
#
def solve_core(par_elem_n, par_node_n):
    # A small fixed bank of scalar temporaries is used because the current DSL
    # is scalar-only. Reusing a few named registers is much more HLS-friendly
    # than introducing Python containers or dynamic temporaries.
    f32_0 = 0
    f32_1 = 0
    f32_2 = 0
    f32_3 = 0
    f32_4 = 0
    f32_5 = 1
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

    # ------------------------------------------------------------------
    # Stage 0: derive the final MNA dimension.
    #
    # We start with one unknown per non-ground circuit node. Each independent
    # voltage source contributes one extra branch-current unknown, so we count
    # them here before clearing / stamping the matrix memories.
    # ------------------------------------------------------------------
    u16_dim = par_node_n
    for u16_e in range(par_elem_n):
        u8_kind = fetchElemKind(idx=u16_e)
        if u8_kind == 3:
            u16_dim = u16_dim + 1

    # ------------------------------------------------------------------
    # Stage 1: clear all working memories used by the solver.
    #
    # The generated hardware assumes explicit initialization through the store
    # interfaces rather than relying on simulator reset values or BRAM init.
    # ------------------------------------------------------------------
    for u16_i in range(u16_dim):
        store_J(i=u16_i, v=f32_0)
        store_Y(i=u16_i, v=f32_0)
        store_X(i=u16_i, v=f32_0)
        for u16_j in range(u16_dim):
            store_A(i=u16_i, j=u16_j, v=f32_0)
            store_LU(i=u16_i, j=u16_j, v=f32_0)

    # ------------------------------------------------------------------
    # Stage 2: netlist stamping.
    #
    # This stage performs the "preprocessing" that still remains in hardware:
    # - convert raw node ids to solver matrix indices
    # - allocate branch-variable indices for voltage sources
    # - emit dense MNA updates into A and J through accumulate primitives
    # ------------------------------------------------------------------
    u16_next_aux = par_node_n
    for u16_e in range(par_elem_n):
        u8_kind = fetchElemKind(idx=u16_e)
        u16_n0 = fetchElemN0(idx=u16_e)
        u16_n1 = fetchElemN1(idx=u16_e)
        f32_1 = fetchElemVal0(idx=u16_e)

        # Convert raw circuit node numbers to dense matrix indices.
        # Ground (node 0) is mapped to a sentinel so the stamp rules can simply
        # guard on `!= 65535` before touching the matrix or RHS vector.
        if u16_n0 != 0:
            u16_n0 = u16_n0 - 1
        else:
            u16_n0 = 65535

        if u16_n1 != 0:
            u16_n1 = u16_n1 - 1
        else:
            u16_n1 = 65535

        # Resistor stamp:
        #   g = 1 / R
        #   [ +g  -g ]
        #   [ -g  +g ]
        if u8_kind == 1:
            f32_2 = div(a=f32_5, b=f32_1)
            if u16_n0 != 65535:
                accumA(i=u16_n0, j=u16_n0, delta=f32_2)
                if u16_n1 != 65535:
                    f32_3 = neg_comb(v=f32_2)
                    accumA(i=u16_n0, j=u16_n1, delta=f32_3)
                    accumA(i=u16_n1, j=u16_n0, delta=f32_3)
            if u16_n1 != 65535:
                accumA(i=u16_n1, j=u16_n1, delta=f32_2)

        # Current source stamp:
        # current is defined from n0 -> n1, so it subtracts from the n0 entry
        # of J and adds to the n1 entry.
        if u8_kind == 2:
            if u16_n0 != 65535:
                f32_2 = neg_comb(v=f32_1)
                accumJ(i=u16_n0, delta=f32_2)
            if u16_n1 != 65535:
                accumJ(i=u16_n1, delta=f32_1)

        # Independent voltage source stamp:
        # allocate a fresh branch-variable row/column and emit the standard
        # MNA coupling terms plus the source value into J.
        if u8_kind == 3:
            u16_aux = u16_next_aux
            u16_next_aux = u16_next_aux + 1
            if u16_n0 != 65535:
                accumA(i=u16_aux, j=u16_n0, delta=f32_5)
                f32_2 = neg_comb(v=f32_5)
                accumA(i=u16_n0, j=u16_aux, delta=f32_2)
            if u16_n1 != 65535:
                f32_2 = neg_comb(v=f32_5)
                accumA(i=u16_aux, j=u16_n1, delta=f32_2)
                accumA(i=u16_n1, j=u16_aux, delta=f32_5)
            accumJ(i=u16_aux, delta=f32_1)

    # ------------------------------------------------------------------
    # Stage 3: dense LU decomposition with partial row pivoting.
    #
    # We build packed LU column by column. For each column j:
    # - find the largest-magnitude residual in column j
    # - swap rows in A, the already-computed strict-lower part of LU, and J
    # - compute the current LU column
    #
    # This follows the same packed-LU convention used by the standalone
    # `lu_core`, but with added pivoting so the combined solver is more robust.
    # ------------------------------------------------------------------
    for u16_j in range(u16_dim):
        # Search the best pivot row in the current column using the residual
        # values that would become U(*, j) before division.
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

        # Row swaps must keep A, the already materialized lower triangle in LU,
        # and the RHS J consistent with each other.
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

        # Compute column j of packed LU.
        # - when i <= j, we are producing U(i, j)
        # - when i >  j, we are producing L(i, j) = residual / U(j, j)
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

    # ------------------------------------------------------------------
    # Stage 4: forward substitution.
    #
    # Solve L * Y = J using the strict-lower part of packed LU. The diagonal of
    # L is implicit 1, so there is no division in this stage.
    # ------------------------------------------------------------------
    for u16_i in range(u16_dim):
        f32_1 = fetch_J(i=u16_i)
        for u16_k in range(u16_i):
            f32_2 = fetch_LU(i=u16_i, j=u16_k)
            f32_3 = fetch_Y(i=u16_k)
            f32_3 = neg_comb(v=f32_3)
            f32_1 = fma(a=f32_2, b=f32_3, c=f32_1)
        store_Y(i=u16_i, v=f32_1)

    # ------------------------------------------------------------------
    # Stage 5: backward substitution.
    #
    # Solve U * X = Y by walking upward from the last row to the first row.
    # ------------------------------------------------------------------
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
