import networkx as nx
from evacuation_layer import congestion_cost
import numpy as np

def build_base_routing_graph(G, edge_index):
    G_ev = nx.DiGraph()
    for (u, v), _ in edge_index.items():
        if G.has_edge(u, v):
            G_ev.add_edge(u, v, weight=G[u][v]["length"])
        if G.has_edge(v, u):
            G_ev.add_edge(v, u, weight=G[v][u]["length"])
    return G_ev


def edge_flood_at_time(e_idx, flood, t, edge_cells):
    i, j = edge_cells[e_idx]
    return flood[t, i, j]

def run_ev_vehicle(
    origin,
    destination,
    G,
    edge_index,
    edge_list,
    history,        # history[t][e]
    flood,          # flood[t, i, j]
    edge_cells,
    DT,
    T_START,
    T_MAX,
    policy="baseline",
    H_BLOCK=0.20
):
    """
    EV routing with flood cost + congestion and persistent backtracking.

    - Hard block if h >= H_BLOCK
    - Flood slowdown: v_flood = v0 * exp(-9h)
    - Congestion speed from CTM
    - Motion uses min(congested, flooded) speed

    Baseline:
      - routing uses congestion only
      - blocks edges ONLY when encountered
      - blocked edges persist (backtracking works)

    Experimental:
      - routing uses effective speed
      - globally prunes flooded edges
    """

    def edge_flood(e_idx, t):
        i, j = edge_cells[e_idx]
        return flood[t, i, j]

    current_node = origin
    t = int(T_START)

    path_log = [(t, current_node)]

    blocked_edges = set()   

    metrics = {
        "policy": policy,
        "arrived": False,
        "t_start": T_START,
        "t_end": None,
        "travel_time_sec": 0.0,
        "edges_used": 0,
        "reroutes": 0,
        "blocked_edges": 0,
    }

    while t < T_MAX:

        if current_node == destination:
            metrics["arrived"] = True
            metrics["t_end"] = t
            break

        n_total = history[t]

        # ---------- build routing graph ----------
        G_ev = nx.DiGraph()

        for (u, v), e_idx in edge_index.items():

            # persistently removed edges
            if (u, v) in blocked_edges:
                continue

            h = edge_flood(e_idx, t)

            # experimental preemptive block
            if policy == "experimental" and h >= H_BLOCK:
                blocked_edges.add((u, v))
                metrics["blocked_edges"] += 1
                continue

            L  = G[u][v]["length"]
            v0 = G[u][v]["v_free"]

            # congestion
            tt_cong = congestion_cost(e_idx, n_total, G, edge_list)
            v_cong = L / max(tt_cong, 1e-6)

            # flood
            v_flood = v0 * np.exp(-9.0 * h)

            if policy == "baseline":
                weight = tt_cong
            else:
                v_eff = min(v_cong, v_flood)
                weight = L / max(v_eff, 1e-6)

            G_ev.add_edge(u, v, weight=weight)

        # ---------- routing ----------
        try:
            route = nx.shortest_path(
                G_ev, current_node, destination, weight="weight"
            )
            metrics["reroutes"] += 1
        except nx.NetworkXNoPath:
            metrics["t_end"] = t
            break

        if len(route) < 2:
            metrics["t_end"] = t
            break

        u, v = route[0], route[1]
        e_idx = edge_index.get((u, v))
        if e_idx is None:
            metrics["t_end"] = t
            break

        # ---------- reactive hard block ----------
        h = edge_flood(e_idx, t)
        if h >= H_BLOCK:
            blocked_edges.add((u, v))        # ← FIX
            metrics["blocked_edges"] += 1
            t += 1
            path_log.append((t, current_node))
            continue

        # ---------- movement ----------
        L  = G[u][v]["length"]
        v0 = G[u][v]["v_free"]

        tt_cong = congestion_cost(e_idx, n_total, G, edge_list)
        v_cong = L / max(tt_cong, 1e-6)
        v_flood = v0 * np.exp(-9.0 * h)

        v_eff = min(v_cong, v_flood)
        travel_time = L / max(v_eff, 1e-6)

        steps = max(1, int(np.ceil(travel_time / DT)))

        metrics["travel_time_sec"] += travel_time
        metrics["edges_used"] += 1

        for _ in range(steps):
            t += 1
            path_log.append((t, current_node))
            if t >= T_MAX:
                break

        current_node = v
        path_log.append((t, current_node))

    if metrics["t_end"] is None:
        metrics["t_end"] = t

    return path_log, metrics
