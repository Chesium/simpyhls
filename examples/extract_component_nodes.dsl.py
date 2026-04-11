#
# extract_component_nodes
# -----------------------
# Post-flood netlist annotation pass.
#
# Responsibilities:
# - discover whether a raw flooded region is connected to any ground symbol
# - compact sparse raw region ids into continuous node indices
# - write node0/node1 for every two-terminal component
#
# Interface notes:
# - This draft intentionally reuses the existing backend style:
#     fetchComponentType(idx)
#     fetchAnchorPositionX(idx)
#     fetchAnchorPositionY(idx)
#     fetchR(i, j)
#     storeNode0(idx, node_i)
# - It also assumes two small additions that the current RTL does not yet expose:
#     fetchComponentRotation(idx)
#     storeNode1(idx, node_i)
#
# Compaction policy:
# - every ground-connected raw region -> compact node 0
# - every other nonzero raw region is assigned in first-seen row-major order
# - because the current project does not yet have a dedicated region-remap RAM,
#   this draft computes groundedness and compact indices by scanning the flooded
#   result matrix and the ground components directly
#
def extract_component_nodes(par_elem_n, grid_height, grid_width):
    u16_idx = 0
    u16_ground_idx = 0
    u8_type = 0
    u8_ground_type = 0
    u8_anchor_x = 0
    u8_anchor_y = 0
    u8_ground_anchor_x = 0
    u8_ground_anchor_y = 0
    u2_rot = 0
    u2_dir0 = 0
    u2_ground_rot = 0
    u2_ground_dir0 = 0
    u8_term0_x = 0
    u8_term0_y = 0
    u8_term1_x = 0
    u8_term1_y = 0
    u8_ground_term_x = 0
    u8_ground_term_y = 0
    u8_raw0 = 0
    u8_raw1 = 0
    u8_ground_region = 0
    u8_region = 0
    u8_prev_region = 0
    u8_node0 = 0
    u8_node1 = 0
    u8_next_node = 0
    u8_scan_x = 0
    u8_scan_y = 0
    u8_prev_x = 0
    u8_prev_y = 0
    u1_need0 = 0
    u1_need1 = 0
    u1_seen = 0
    u1_raw0_is_ground = 0
    u1_raw1_is_ground = 0
    u1_region_is_ground = 0

    # Phase 2: extract node0/node1 for every two-terminal component and compact
    # raw flooded region ids into a continuous node numbering.
    for u16_idx in range(par_elem_n):
        u8_type = fetchComponentType(idx=u16_idx)
        u8_node0 = 0
        u8_node1 = 0
        u8_raw0 = 0
        u8_raw1 = 0
        u1_need0 = 0
        u1_need1 = 0
        u1_raw0_is_ground = 0
        u1_raw1_is_ground = 0

        if is_two_terminal_component_comb(t=u8_type):
            u8_anchor_x = fetchAnchorPositionX(idx=u16_idx)
            u8_anchor_y = fetchAnchorPositionY(idx=u16_idx)
            u2_rot = fetchComponentRotation(idx=u16_idx)

            # Terminal 0 is on the side opposite the component growth direction.
            u2_dir0 = get_opp_dir_comb(d=u2_rot)
            u8_term0_x = get_nxt_i_comb(i=u8_anchor_x, d=u2_dir0)
            u8_term0_y = get_nxt_j_comb(j=u8_anchor_y, d=u2_dir0)
            if u8_term0_x < grid_width and u8_term0_y < grid_height:
                u8_raw0 = fetchR(i=u8_term0_x, j=u8_term0_y)

            # Terminal 1 is beyond the far end of the two-cell component.
            u8_term1_x = get_nxt_i_comb(i=u8_anchor_x, d=u2_rot)
            u8_term1_y = get_nxt_j_comb(j=u8_anchor_y, d=u2_rot)
            u8_term1_x = get_nxt_i_comb(i=u8_term1_x, d=u2_rot)
            u8_term1_y = get_nxt_j_comb(j=u8_term1_y, d=u2_rot)
            if u8_term1_x < grid_width and u8_term1_y < grid_height:
                u8_raw1 = fetchR(i=u8_term1_x, j=u8_term1_y)

            if u8_raw0 != 0:
                for u16_ground_idx in range(par_elem_n):
                    u8_ground_type = fetchComponentType(idx=u16_ground_idx)
                    if is_ground_component_comb(t=u8_ground_type):
                        u8_ground_anchor_x = fetchAnchorPositionX(idx=u16_ground_idx)
                        u8_ground_anchor_y = fetchAnchorPositionY(idx=u16_ground_idx)
                        u2_ground_rot = fetchComponentRotation(idx=u16_ground_idx)
                        u2_ground_dir0 = get_opp_dir_comb(d=u2_ground_rot)
                        u8_ground_term_x = get_nxt_i_comb(
                            i=u8_ground_anchor_x, d=u2_ground_dir0
                        )
                        u8_ground_term_y = get_nxt_j_comb(
                            j=u8_ground_anchor_y, d=u2_ground_dir0
                        )
                        if (
                            u8_ground_term_x < grid_width
                            and u8_ground_term_y < grid_height
                        ):
                            u8_ground_region = fetchR(
                                i=u8_ground_term_x, j=u8_ground_term_y
                            )
                            if u8_ground_region == u8_raw0:
                                u1_raw0_is_ground = 1
                if u1_raw0_is_ground:
                    u8_node0 = 0
                else:
                    u1_need0 = 1

            if u8_raw1 != 0:
                for u16_ground_idx in range(par_elem_n):
                    u8_ground_type = fetchComponentType(idx=u16_ground_idx)
                    if is_ground_component_comb(t=u8_ground_type):
                        u8_ground_anchor_x = fetchAnchorPositionX(idx=u16_ground_idx)
                        u8_ground_anchor_y = fetchAnchorPositionY(idx=u16_ground_idx)
                        u2_ground_rot = fetchComponentRotation(idx=u16_ground_idx)
                        u2_ground_dir0 = get_opp_dir_comb(d=u2_ground_rot)
                        u8_ground_term_x = get_nxt_i_comb(
                            i=u8_ground_anchor_x, d=u2_ground_dir0
                        )
                        u8_ground_term_y = get_nxt_j_comb(
                            j=u8_ground_anchor_y, d=u2_ground_dir0
                        )
                        if (
                            u8_ground_term_x < grid_width
                            and u8_ground_term_y < grid_height
                        ):
                            u8_ground_region = fetchR(
                                i=u8_ground_term_x, j=u8_ground_term_y
                            )
                            if u8_ground_region == u8_raw1:
                                u1_raw1_is_ground = 1
                if u1_raw1_is_ground:
                    u8_node1 = 0
                else:
                    u1_need1 = 1

            if u1_need0 or u1_need1:
                u8_next_node = 1
                for u8_scan_y in range(grid_height):
                    for u8_scan_x in range(grid_width):
                        u8_region = fetchR(i=u8_scan_x, j=u8_scan_y)
                        u1_region_is_ground = 0
                        if u8_region != 0:
                            for u16_ground_idx in range(par_elem_n):
                                u8_ground_type = fetchComponentType(
                                    idx=u16_ground_idx
                                )
                                if is_ground_component_comb(t=u8_ground_type):
                                    u8_ground_anchor_x = fetchAnchorPositionX(
                                        idx=u16_ground_idx
                                    )
                                    u8_ground_anchor_y = fetchAnchorPositionY(
                                        idx=u16_ground_idx
                                    )
                                    u2_ground_rot = fetchComponentRotation(
                                        idx=u16_ground_idx
                                    )
                                    u2_ground_dir0 = get_opp_dir_comb(
                                        d=u2_ground_rot
                                    )
                                    u8_ground_term_x = get_nxt_i_comb(
                                        i=u8_ground_anchor_x, d=u2_ground_dir0
                                    )
                                    u8_ground_term_y = get_nxt_j_comb(
                                        j=u8_ground_anchor_y, d=u2_ground_dir0
                                    )
                                    if (
                                        u8_ground_term_x < grid_width
                                        and u8_ground_term_y < grid_height
                                    ):
                                        u8_ground_region = fetchR(
                                            i=u8_ground_term_x,
                                            j=u8_ground_term_y,
                                        )
                                        if u8_ground_region == u8_region:
                                            u1_region_is_ground = 1
                        if u8_region != 0 and not u1_region_is_ground:
                            u1_seen = 0
                            for u8_prev_y in range(grid_height):
                                for u8_prev_x in range(grid_width):
                                    if (
                                        u8_prev_y < u8_scan_y
                                        or (
                                            u8_prev_y == u8_scan_y
                                            and u8_prev_x < u8_scan_x
                                        )
                                    ):
                                        u8_prev_region = fetchR(
                                            i=u8_prev_x, j=u8_prev_y
                                        )
                                        if u8_prev_region == u8_region:
                                            u1_seen = 1
                            if not u1_seen:
                                if u1_need0 and u8_region == u8_raw0:
                                    u8_node0 = u8_next_node
                                    u1_need0 = 0
                                if u1_need1 and u8_region == u8_raw1:
                                    u8_node1 = u8_next_node
                                    u1_need1 = 0
                                u8_next_node = u8_next_node + 1

        storeNode0(idx=u16_idx, node_i=u8_node0)
        storeNode1(idx=u16_idx, node_i=u8_node1)
