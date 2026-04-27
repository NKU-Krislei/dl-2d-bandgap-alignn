"""
Step 6: Create report-ready concept figures and a compact project summary.
"""

from __future__ import annotations

import argparse

import numpy as np

from utils import FIGURES_DIR, RESULTS_DIR, ensure_directories, read_json, setup_matplotlib, write_json


def create_overview_figure() -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.set_title("ALIGNN Workflow for Band Gap Prediction", fontsize=16, fontweight="bold", pad=20)

    boxes = [
        (0.4, 2.3, 2.3, 1.5, "Crystal\nStructure", "#dbeafe", "#1d4ed8"),
        (3.3, 2.3, 2.5, 1.5, "Crystal Graph\n(two-body)", "#dcfce7", "#15803d"),
        (6.4, 2.3, 2.5, 1.5, "Line Graph\n(angles)", "#fef3c7", "#b45309"),
        (9.5, 2.3, 2.5, 1.5, "ALIGNN\nmessage passing", "#fae8ff", "#7e22ce"),
        (12.5, 2.3, 1.1, 1.5, "Band Gap", "#fee2e2", "#b91c1c"),
    ]

    for x, y, w, h, label, face, edge in boxes:
        patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=face, edgecolor=edge, linewidth=2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, fontweight="bold", color=edge)

    for start, end in ((2.7, 3.3), (5.8, 6.4), (8.9, 9.5), (12.0, 12.5)):
        ax.annotate("", xy=(end, 3.05), xytext=(start, 3.05), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#334155"})

    ax.text(1.55, 1.5, "Atomic species,\ncoordinates", ha="center", fontsize=9)
    ax.text(4.55, 1.5, "Atoms as nodes,\nbonds as edges", ha="center", fontsize=9)
    ax.text(7.65, 1.5, "Bond-bond graph\nencodes angles", ha="center", fontsize=9)
    ax.text(10.75, 1.5, "Alternating updates\nacross both graphs", ha="center", fontsize=9)
    ax.text(13.05, 1.5, "Regression\noutput (eV)", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "alignn_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_crystal_graph_demo() -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    import networkx as nx

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    a = 1.0
    boron = np.array([[0, 0], [1.0, 0], [0.5, np.sqrt(3) / 2]])
    nitrogen = np.array([[0.0, np.sqrt(3) / 3], [1.0, np.sqrt(3) / 3], [0.5, -np.sqrt(3) / 6]])

    axes[0].scatter(boron[:, 0], boron[:, 1], s=220, color="#fb7185", edgecolors="#9f1239", label="B")
    axes[0].scatter(nitrogen[:, 0], nitrogen[:, 1], s=220, color="#22d3ee", edgecolors="#155e75", label="N")
    for bx, by in boron:
        for nx_, ny_ in nitrogen:
            if np.hypot(bx - nx_, by - ny_) < 0.8:
                axes[0].plot([bx, nx_], [by, ny_], color="#475569", lw=2)
    axes[0].set_title("(a) h-BN Unit Cell")
    axes[0].axis("equal")
    axes[0].axis("off")
    axes[0].legend()

    graph = nx.Graph()
    for idx, (x, y) in enumerate(boron):
        graph.add_node(f"B{idx}", pos=(x, y), color="#fb7185")
    for idx, (x, y) in enumerate(nitrogen):
        graph.add_node(f"N{idx}", pos=(x, y), color="#22d3ee")
    for i, (bx, by) in enumerate(boron):
        for j, (nx_, ny_) in enumerate(nitrogen):
            if np.hypot(bx - nx_, by - ny_) < 0.8:
                graph.add_edge(f"B{i}", f"N{j}")

    pos = nx.get_node_attributes(graph, "pos")
    colors = [graph.nodes[node]["color"] for node in graph.nodes]
    nx.draw(graph, pos=pos, ax=axes[1], node_color=colors, node_size=450, edge_color="#64748b", with_labels=True)
    axes[1].set_title("(b) Crystal Graph")

    axes[2].axis("off")
    axes[2].set_title("(c) Line Graph Concept")
    axes[2].text(0.5, 0.72, "Nodes = bonds", ha="center", fontsize=13, fontweight="bold")
    axes[2].text(0.5, 0.52, "Edges = shared atoms", ha="center", fontsize=12)
    axes[2].text(0.5, 0.32, "This carries\nbond-angle information", ha="center", fontsize=12, color="#b91c1c")
    axes[2].annotate("", xy=(0.3, 0.2), xytext=(0.5, 0.45), arrowprops={"arrowstyle": "->", "lw": 2})
    axes[2].annotate("", xy=(0.7, 0.2), xytext=(0.5, 0.45), arrowprops={"arrowstyle": "->", "lw": 2})

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "crystal_graph_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_physics_context() -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    k = np.linspace(0, 4, 200)
    conduction = 1.8 + 0.35 * np.cos(1.4 * k)
    valence = -0.4 - 0.55 * np.cos(1.4 * k)
    axes[0].plot(k, conduction, color="#2563eb", lw=2, label="Conduction band")
    axes[0].plot(k, valence, color="#dc2626", lw=2, label="Valence band")
    axes[0].annotate("", xy=(1.9, valence[95]), xytext=(1.9, conduction[95]), arrowprops={"arrowstyle": "<->", "lw": 2, "color": "#16a34a"})
    axes[0].text(2.05, 0.5, "$E_g$", color="#16a34a", fontsize=15)
    axes[0].set_title("(a) Band Structure")
    axes[0].set_xlabel("Wave vector")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].legend()

    materials = ["Graphene", "BP", "MoS$_2$", "MoSe$_2$", "h-BN"]
    gaps = [0.0, 0.3, 1.89, 1.55, 5.97]
    axes[1].barh(materials, gaps, color=["#64748b", "#f59e0b", "#14b8a6", "#3b82f6", "#8b5cf6"])
    axes[1].set_title("(b) Typical 2D Band Gaps")
    axes[1].set_xlabel("Band gap (eV)")

    axes[2].axis("off")
    axes[2].set_title("(c) Data-to-Discovery Loop")
    steps = ["DFT database", "Graph encoding", "ALIGNN model", "Fast screening", "Candidate materials"]
    for idx, label in enumerate(steps):
        x = 0.08 + idx * 0.18
        patch = patches.FancyBboxPatch((x, 0.38), 0.14, 0.22, boxstyle="round,pad=0.03", facecolor="#e2e8f0", edgecolor="#475569", transform=axes[2].transAxes)
        axes[2].add_patch(patch)
        axes[2].text(x + 0.07, 0.49, label, ha="center", va="center", fontsize=9, transform=axes[2].transAxes)
        if idx < len(steps) - 1:
            axes[2].annotate("", xy=(x + 0.16, 0.49), xytext=(x + 0.14, 0.49), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.8})

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "physics_context.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_summary() -> dict[str, object]:
    summary = {
        "dataset_stats": read_json(RESULTS_DIR / "dataset_stats.json", default={}) or {},
        "benchmark": read_json(RESULTS_DIR / "pretrain_benchmark.json", default={}) or {},
        "evaluation": read_json(RESULTS_DIR / "evaluation_report.json", default={}) or {},
        "training": read_json(RESULTS_DIR / "training_history.json", default={}) or {},
    }
    write_json(RESULTS_DIR / "final_summary.json", summary)
    return summary


def generate_all_figures() -> dict[str, object]:
    ensure_directories()
    create_overview_figure()
    create_crystal_graph_demo()
    create_physics_context()
    return generate_summary()


def run() -> dict[str, object]:
    summary = generate_all_figures()
    print(f"💾 Saved summary bundle to {RESULTS_DIR / 'final_summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate concept figures and summary output")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
