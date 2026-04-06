def flooding_core(grid_height, grid_width):
    u8_i = 0
    u8_j = 0
    u4_p = 0
    u1_f = 0
    u8_node_idx = 0
    u16_t = 0
    u18_cur = 0
    u8_ni = 0
    u8_nj = 0
    u2_dir = 0
    for u8_j in range(grid_height):
        for u8_i in range(grid_width):
            u4_p = fetchP(i=u8_i, j=u8_j)
            u1_f = getVisited(i=u8_i, j=u8_j)
            if not u1_f and decode_iswire_comb(p=u4_p):
                u8_node_idx = u8_node_idx + 1
                setVisited(i=u8_i, j=u8_j)
                storeR(i=u8_i, j=u8_j, v=u8_node_idx)
                if getport_comb(p=u4_p, i=0):
                    addQueue(i=u8_i, j=u8_j, d=0)
                if getport_comb(p=u4_p, i=1):
                    addQueue(i=u8_i, j=u8_j, d=1)
                if getport_comb(p=u4_p, i=2):
                    addQueue(i=u8_i, j=u8_j, d=2)
                if getport_comb(p=u4_p, i=3):
                    addQueue(i=u8_i, j=u8_j, d=3)
                u16_t = getQueueLen()
                while u16_t > 0:
                    u18_cur = popQueue()
                    u8_ni = decode_i_comb(q_item=u18_cur)
                    u8_nj = decode_j_comb(q_item=u18_cur)
                    u2_dir = decode_d_comb(q_item=u18_cur)
                    u8_ni = get_nxt_i_comb(i=u8_ni, d=u2_dir)
                    u8_nj = get_nxt_j_comb(j=u8_nj, d=u2_dir)
                    u1_f = getVisited(i=u8_ni, j=u8_nj)
                    if (
                        not u1_f
                        and u8_ni >= 0
                        and u8_nj >= 0
                        and u8_ni < grid_width
                        and u8_nj < grid_height
                    ):
                        u4_p = fetchP(i=u8_ni, j=u8_nj)
                        if getport_comb(p=u4_p, i=get_opp_dir_comb(d=u2_dir)):
                            setVisited(i=u8_ni, j=u8_nj)
                            storeR(i=u8_ni, j=u8_nj, v=u8_node_idx)
                            if getport_comb(p=u4_p, i=0):
                                addQueue(i=u8_ni, j=u8_nj, d=0)
                            if getport_comb(p=u4_p, i=1):
                                addQueue(i=u8_ni, j=u8_nj, d=1)
                            if getport_comb(p=u4_p, i=2):
                                addQueue(i=u8_ni, j=u8_nj, d=2)
                            if getport_comb(p=u4_p, i=3):
                                addQueue(i=u8_ni, j=u8_nj, d=3)
                    u16_t = getQueueLen()