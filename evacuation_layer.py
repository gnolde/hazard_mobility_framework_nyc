import numpy as np
import networkx as nx


def define_od_pairs():
    OD_pairs = [
        (42453472,42455026),(42429918,42455026),(42436340,42455026),
        (8840333846,42455026),(42448483,42457325),(4300940094,42455026),
        (42432002,42457325),(278608420,42457325),(42429844,42457325),
        (4207802391,42457325),(42440847,42457325)
    ]
    origin_nodes, dest_nodes = map(list, zip(*OD_pairs))
    return OD_pairs, origin_nodes, dest_nodes


def demand_func(t):
    if t < 120:
        return 15/12
    elif t < 360:
        return 25/12
    else:
        return 0.0


def freeflow_cost(u, v, data):
    return data["length"] / data["v_free"]


def initial_first_edges(G, OD_pairs, edge_index):
    first_edge = []
    for o, d in OD_pairs:
        path = nx.shortest_path(G, o, d, weight=freeflow_cost)
        first_edge.append(edge_index[(path[0], path[1])])
    return first_edge


def choo_speed_factor(h, FLOOD_CLOSURE_H):
    if h >= FLOOD_CLOSURE_H:
        return 0.0
    return np.exp(-9.0 * h)

def congestion_cost(e_idx, n_total, G, edge_list):
    u, v = edge_list[e_idx]
    length = G[u][v]["length"]
    v_free = G[u][v]["v_free"]
    cap    = G[u][v]["capacity"]

    t0 = length / v_free
    k_crit = cap / v_free
    k_e = n_total[e_idx] / length
    Q = max(0.0, (k_e - k_crit) * length)

    return t0 + Q / cap


def ctm_step(t, n, next_edge, flood_e, G, edge_index, origin_demand, first_edge, FLOOD_CLOSURE_H, DT):
    K, E = n.shape
    S = np.zeros((K, E))
    R = np.zeros(E)

    # sending (allow exit if flooded)
    for k in range(K):
        for (u, v), idx in edge_index.items():
            length = G[u][v]["length"]
            v0     = G[u][v]["v_free"]
            cap    = G[u][v]["capacity"]

            # flood slows vehicles but does NOT block exit
            sf = choo_speed_factor(flood_e[idx], FLOOD_CLOSURE_H)
            v_eff = v0 * sf

            # if completely flooded, v_eff = 0 → no movement, but no hard block
            S[k, idx] = min(
                (v_eff * DT / length) * n[k, idx],
                cap * DT,
                n[k, idx]
            )

    # receiving (block entry if flooded)
    for (u, v), idx in edge_index.items():
        length = G[u][v]["length"]
        k_jam  = G[u][v]["k_jam"]
        w      = G[u][v]["w"]
        cap    = G[u][v]["capacity"]

        sf = choo_speed_factor(flood_e[idx], FLOOD_CLOSURE_H)
        if sf <= 0:
            continue   # no entry allowed

        space = max(k_jam * length - n[:, idx].sum(), 0.0)
        R[idx] = min((w * DT / length) * space, cap * DT)

    # node flows
    f = np.zeros((K, E, E))
    for k in range(K):
        for i in next_edge[k]:
            j = next_edge[k][i]
            f[k, i, j] = S[k, i]

    for j in range(E):
        D = f[:, :, j].sum()
        if D > R[j] and D > 0:
            f[:, :, j] *= R[j] / D

    # external inflow 
    q_in = np.zeros((K, E))
    for k in range(K):
        e0 = first_edge[k]
        free = R[e0] - f[:, :, e0].sum()
        q_in[k, e0] = max(min(origin_demand[k](t), free), 0)

    # update state
    n_next = n.copy()
    for k in range(K):
        for e in range(E):
            n_next[k, e] += f[k, :, e].sum() - f[k, e, :].sum() + q_in[k, e]
            n_next[k, e] = max(n_next[k, e], 0)

    return n_next


def reroute_all(FLOOD_CLOSURE_H, G, edge_index, edge_list, OD_pairs, n, flood_e, prev_next_edge=None, p_reroute=1.0):
    K = len(OD_pairs)
    E = len(edge_list)

    # if first call, initialize
    if prev_next_edge is None:
        next_edge = [dict() for _ in range(K)]
    else:
        # start from previous routing
        next_edge = [dict(ne) for ne in prev_next_edge]

    n_total = n.sum(axis=0)

    # build weighted graph
    G_w = nx.DiGraph()
    for (u, v), idx in edge_index.items():
        if flood_e[idx] >= FLOOD_CLOSURE_H:
            continue
        cost = congestion_cost(idx, n_total, G, edge_list)
        G_w.add_edge(u, v, weight=cost)

    G_rev = G_w.reverse(copy=False)

    for k, (o, d) in enumerate(OD_pairs):

        if np.random.rand() > p_reroute:
            continue   # keep existing routing for OD pair k

        # otherwise, recompute shortest paths for OD pair k
        try:
            _, paths = nx.single_source_dijkstra(G_rev, d, weight="weight")
        except:
            continue

        next_edge[k].clear()

        for (u, v), i in edge_index.items():
            if v in paths and len(paths[v]) > 1:
                nxt = paths[v][-2]
                if (v, nxt) in edge_index:
                    next_edge[k][i] = edge_index[(v, nxt)]

    return next_edge