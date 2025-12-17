import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.collections import LineCollection


def animate_flood_and_congestion(flood, X, Y, history,thr, origin_nodes,dest_nodes, G,edges_gdf, interval=200):
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['animation.embed_limit'] = 100

    fig, ax = plt.subplots(figsize=(10, 10))

    # flood map
    pcm = ax.pcolormesh(
        X[0], Y[0], flood[0],
        cmap="Reds",
        shading="auto",
        zorder=0,
        alpha=0.6
    )

    cbar_flood = fig.colorbar(pcm, ax=ax, fraction=0.03, pad=0.01)
    cbar_flood.set_label("Water depth (m)")

    # static network bakcground
    edges_gdf.plot(color="#cccccc", linewidth=1, ax=ax, zorder=1)

    # Origins
    for k, o in enumerate(origin_nodes):
        x, y = G.nodes[o]["x"], G.nodes[o]["y"]
        ax.scatter(x, y, s=90, c="red", edgecolors="black", zorder=5)
        ax.text(x, y, f"O{k+1}", color="red",
                fontsize=10, ha="right", va="bottom", zorder=6)

    # Destinations
    for k, d in enumerate(dest_nodes):
        x, y = G.nodes[d]["x"], G.nodes[d]["y"]
        ax.scatter(x, y, s=90, c="green", edgecolors="black", zorder=5)
        ax.text(x, y, f"D{k+1}", color="green",
                fontsize=10, ha="right", va="bottom", zorder=6)

    ax.set_axis_off()

    # congestion color bar
    global_max = max(n.max() for n in history)
    cong_norm = plt.Normalize(vmin=thr, vmax=global_max)
    cong_sm = plt.cm.ScalarMappable(cmap="viridis", norm=cong_norm)
    cong_sm.set_array([])

    cbar_cong = fig.colorbar(cong_sm, ax=ax, fraction=0.03, pad=0.06)
    cbar_cong.set_label(f"Vehicles on Link (>{thr})")

    # dynamic layers
    congestion_layer = None
    zero_flow_layer = None

    def update(frame):
        nonlocal congestion_layer

        # update flood values
        pcm.set_array(flood[frame].ravel())

        # remove previous congestion layer
        if congestion_layer is not None:
            congestion_layer.remove()
            congestion_layer = None

        # current link vehicle counts
        n_t = history[frame]

        # select links with ANY flow (> 0)
        gdf = edges_gdf.copy()
        gdf["vehicles"] = n_t
        used = gdf[gdf["vehicles"] > 0]

        # plot only links with flow
        if len(used) > 0:
            segs = [np.array(geom.coords) for geom in used.geometry]
            colors = used["vehicles"].values

            congestion_layer = LineCollection(
                segs,
                cmap="viridis",
                norm=cong_norm,
                linewidths=3,
                zorder=3
            )
            congestion_layer.set_array(colors)
            ax.add_collection(congestion_layer)

        ax.set_title(f"Flood + CTM Congestion — t={frame}")
        return ax,


    anim = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval,
        blit=False,   
        repeat=True
    )

    plt.close(fig)
    return anim


def animate_flood_congestion_with_EVs(
    flood, X, Y, history,
    path_base, path_exp,
    G, edges_gdf,
    origin_nodes, dest_nodes,
    ev_origin, ev_destination,
    thr=5,
    interval=150
):
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['animation.embed_limit'] = 100

    fig, ax = plt.subplots(figsize=(10, 10))

    # flood
    pcm = ax.pcolormesh(
        X[0], Y[0], flood[0],
        cmap="Reds",
        shading="auto",
        zorder=0,
        alpha=0.6
    )

    cbar_flood = fig.colorbar(pcm, ax=ax, fraction=0.03, pad=0.01)
    cbar_flood.set_label("Water depth (m)")

    # static background network
    edges_gdf.plot(color="#cccccc", linewidth=1, ax=ax, zorder=1)

    # evacuation origins
    for k, o in enumerate(origin_nodes):
        x, y = G.nodes[o]["x"], G.nodes[o]["y"]
        ax.scatter(x, y, s=80, c="red", edgecolors="black", zorder=5)
        ax.text(x, y, f"O{k+1}", color="red",
                fontsize=9, ha="right", va="bottom", zorder=6)

    # evacuation destinations
    for k, d in enumerate(dest_nodes):
        x, y = G.nodes[d]["x"], G.nodes[d]["y"]
        ax.scatter(x, y, s=80, c="green", edgecolors="black", zorder=5)
        ax.text(x, y, f"D{k+1}", color="green",
                fontsize=9, ha="right", va="bottom", zorder=6)

    # emergency vehicle origin
    xo, yo = G.nodes[ev_origin]["x"], G.nodes[ev_origin]["y"]
    ax.scatter(xo, yo, s=160, c="cyan", marker="s",
               edgecolors="black", zorder=7)
    ax.text(xo, yo, "EV start", color="cyan",
            fontsize=10, ha="left", va="bottom", zorder=8)

    # emergency vehicle destination
    xd, yd = G.nodes[ev_destination]["x"], G.nodes[ev_destination]["y"]
    ax.scatter(xd, yd, s=180, c="cyan", marker="*",
               edgecolors="black", zorder=7)
    ax.text(xd, yd, "EV dest", color="cyan",
            fontsize=10, ha="left", va="bottom", zorder=8)

    ax.set_axis_off()

    # congestion colorbar
    global_max = max(n.max() for n in history)
    cong_norm = plt.Normalize(vmin=thr, vmax=global_max)
    cong_sm = plt.cm.ScalarMappable(cmap="viridis", norm=cong_norm)
    cong_sm.set_array([])

    fig.colorbar(
        cong_sm, ax=ax, fraction=0.03, pad=0.06,
        label=f"Vehicles on Link (>{thr})"
    )

    # emergency vehicle data
    def path_to_xy(path):
        xs, ys, ts = [], [], []
        for t, n in path:
            xs.append(G.nodes[n]["x"])
            ys.append(G.nodes[n]["y"])
            ts.append(t)
        return np.array(xs), np.array(ys), np.array(ts)

    xb, yb, tb = path_to_xy(path_base)
    xe, ye, te = path_to_xy(path_exp)

    # emergency vehicle plotting
    base_line, = ax.plot([], [], color="cyan", lw=3, zorder=6, label="EV baseline")
    exp_line,  = ax.plot([], [], color="magenta", lw=3, zorder=6, label="EV experimental")

    base_dot = ax.scatter([], [], s=100, c="cyan", edgecolors="black", zorder=7)
    exp_dot  = ax.scatter([], [], s=100, c="magenta", edgecolors="black", zorder=7)

    ax.legend(loc="upper right")

    congestion_layer = None

    def update(frame):
        nonlocal congestion_layer

        # ---- flood ----
        pcm.set_array(flood[frame].ravel())

        # ---- congestion ----
        if congestion_layer is not None:
            congestion_layer.remove()

        n_t = history[frame]
        gdf = edges_gdf.copy()
        gdf["vehicles"] = n_t
        used = gdf[gdf["vehicles"] > thr]

        if len(used) > 0:
            segs = [np.array(g.coords) for g in used.geometry]
            colors = used["vehicles"].values

            congestion_layer = LineCollection(
                segs,
                cmap="viridis",
                norm=cong_norm,
                linewidths=3,
                zorder=3
            )
            congestion_layer.set_array(colors)
            ax.add_collection(congestion_layer)

        # baseline EV 
        mask_b = tb <= frame
        if mask_b.any():
            base_line.set_data(xb[mask_b], yb[mask_b])
            base_dot.set_offsets([xb[mask_b][-1], yb[mask_b][-1]])

        # experimental EV 
        mask_e = te <= frame
        if mask_e.any():
            exp_line.set_data(xe[mask_e], ye[mask_e])
            exp_dot.set_offsets([xe[mask_e][-1], ye[mask_e][-1]])

        ax.set_title(f"Flood + CTM Congestion + EVs — t={frame}")
        return ax,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval,
        blit=False,
        repeat=True
    )

    plt.close(fig)
    return anim
