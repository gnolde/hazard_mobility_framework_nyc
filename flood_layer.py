import numpy as np
from shapely.geometry import LineString


def load_flood(path):
    flood = np.load(path)
    flood = np.array([(flood[t].T) for t in range(len(flood))])
    return flood


def flood_to_edges(flood, edge_cells, valid):
    T_flood = flood.shape[0]
    E = edge_cells.shape[0]
    flood_depth = np.zeros((T_flood, E), dtype=np.float32)

    I = edge_cells[:, 0]
    J = edge_cells[:, 1]

    for t in range(T_flood):
        frame = flood[t]
        fd = np.zeros(E, dtype=np.float32)
        fd[valid] = frame[I[valid], J[valid]]
        flood_depth[t] = fd

    return flood_depth


def map_edges_to_flood_grid(G, edge_list, bbox, flood_shape):
    x1, x2, y1, y2 = bbox
    nxg, nyg = flood_shape
    dx = (x2 - x1) / nyg
    dy = (y2 - y1) / nxg

    E = len(edge_list)
    edge_cells = np.zeros((E, 2), dtype=int)

    for e_idx, (u, v) in enumerate(edge_list):
        data = G[u][v]
        if "geometry" in data:
            p = data["geometry"].interpolate(0.5, normalized=True)
        else:
            p = LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"])
            ]).interpolate(0.5, normalized=True)

        j = int((p.x - x1) / dx)
        i = int((p.y - y1) / dy)
        edge_cells[e_idx] = (i, j)

    valid = (
        (edge_cells[:,0] >= 0) & (edge_cells[:,0] < nxg) &
        (edge_cells[:,1] >= 0) & (edge_cells[:,1] < nyg)
    )

    return edge_cells, valid