
import osmnx as ox
from shapely.geometry import LineString
import geopandas as gpd


def build_network(north, south, east, west):
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    return ox.utils_graph.get_digraph(G)


def add_edge_attributes(G):
    for u, v in G.edges():
        G[u][v]["length"] = G[u][v].get(
            "length",
            ox.distance.euclidean_dist_vec(
                G.nodes[u]["y"], G.nodes[u]["x"],
                G.nodes[v]["y"], G.nodes[v]["x"]
            )
        )

        sp = G[u][v].get("speed_kph", 25)
        G[u][v]["v_free"] = sp * 1000 / 3600

        G[u][v]["capacity"] = 1800 * 1.5 / 3600
        G[u][v]["k_jam"]    = 0.15 * 1.5
        G[u][v]["w"]        = 4.0


def build_edge_index(G):
    edges = list(G.edges())
    edge_index = {e: i for i, e in enumerate(edges)}
    return edge_index, edges


def build_edges_gdf(G, edge_index):
    geoms = []
    for (u, v), idx in edge_index.items():
        data = G[u][v]
        if "geometry" in data:
            geom = data["geometry"]
        else:
            geom = LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"])
            ])
        geoms.append({"u": u, "v": v, "idx": idx, "geometry": geom})

    return gpd.GeoDataFrame(geoms, crs="EPSG:4326")