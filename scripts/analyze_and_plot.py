#!/usr/bin/env python3
"""Analyze SAXPY occupancy & block-size sweep CSVs and generate Phase 4 plots."""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_baseline(baseline_rows: list[dict], out_path: Path) -> None:
    block_sizes = [int(r["block_size"]) for r in baseline_rows]
    avg_times = [float(r["avg_time_ms"]) for r in baseline_rows]
    occ_pcts = [float(r["theoretical_occupancy_pct"]) for r in baseline_rows]

    x_pos = list(range(len(block_sizes)))
    fastest_idx = min(range(len(avg_times)), key=lambda i: avg_times[i])

    fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=300)

    color_time = "#1f77b4"
    color_occ = "#d62728"

    l1 = ax1.plot(
        x_pos, avg_times, marker="o", linewidth=2.2, markersize=7,
        color=color_time, label="Avg Kernel Time (ms)"
    )
    ax1.scatter(
        [x_pos[fastest_idx]], [avg_times[fastest_idx]],
        s=140, zorder=5, color="#2ca02c", edgecolor="black", linewidth=1.2,
        label=f"Fastest Measured (bs={block_sizes[fastest_idx]}, {avg_times[fastest_idx]:.4f} ms)"
    )
    ax1.annotate(
        f"Fastest: bs={block_sizes[fastest_idx]}\n{avg_times[fastest_idx]:.4f} ms (50% occ)",
        xy=(x_pos[fastest_idx], avg_times[fastest_idx]),
        xytext=(x_pos[fastest_idx] + 0.35, avg_times[fastest_idx] - 0.03),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        fontsize=9, fontweight="bold", color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2ca02c", lw=0.8)
    )

    ax1.set_xlabel("Block Size (threads/block)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Kernel Time (ms)", fontsize=11, fontweight="bold", color=color_time)
    ax1.tick_params(axis="y", labelcolor=color_time)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(b) for b in block_sizes])
    ax1.set_ylim(2.15, 2.65)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    l2 = ax2.plot(
        x_pos, occ_pcts, marker="s", linestyle="--", linewidth=2.0, markersize=6,
        color=color_occ, label="Theoretical Occupancy (%)"
    )
    ax2.set_ylabel("Theoretical Occupancy (%)", fontsize=11, fontweight="bold", color=color_occ)
    ax2.tick_params(axis="y", labelcolor=color_occ)
    ax2.set_ylim(0, 115)

    # Mark cudaOccupancyMaxPotentialBlockSize = 768 (between 512 at x=4 and 1024 at x=5)
    api_x = 4.5
    api_line = ax2.axvline(
        x=api_x, color="#9467bd", linestyle=":", linewidth=2.0,
        label="Occupancy API Optimal (bs=768, 100% occ)"
    )
    ax2.annotate(
        "cudaOccupancyMaxPotentialBlockSize\nbs = 768 (100% theoretical occ)",
        xy=(api_x, 100.0),
        xytext=(2.6, 104.0),
        arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.2),
        fontsize=8.5, color="#4a148c",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f3e5f5", ec="#9467bd", lw=0.8)
    )

    lines = l1 + [ax1.collections[0]] + l2 + [api_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=8.5, framealpha=0.92)

    plt.title(
        "Baseline SAXPY (0 B Dyn Shared Mem): Kernel Time vs. Theoretical Occupancy\n"
        "RTX 4050 Laptop GPU (sm_89), N = 33,554,432",
        fontsize=12, fontweight="bold", pad=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dyn_shared(dyn_rows: list[dict], out_time: Path, out_occ: Path) -> None:
    tiers = sorted({int(r["dyn_shared_bytes"]) for r in dyn_rows})
    block_sizes = sorted({int(r["block_size"]) for r in dyn_rows})
    x_pos = list(range(len(block_sizes)))

    palette = {
        0: ("#1f77b4", "o", "0 B (Baseline)"),
        16384: ("#ff7f0e", "s", "16,384 B (16 KB)"),
        32768: ("#d62728", "^", "32,768 B (32 KB)"),
    }

    # Plot 2A: Execution Time across Dynamic Shared Memory tiers
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    for tier in tiers:
        sub = {int(r["block_size"]): float(r["avg_time_ms"])
               for r in dyn_rows if int(r["dyn_shared_bytes"]) == tier}
        ys = [sub[b] for b in block_sizes]
        color, marker, label = palette.get(tier, ("#333333", "o", f"{tier} B"))
        ax.plot(x_pos, ys, marker=marker, linewidth=2.2, markersize=7,
                color=color, label=label)
        best_i = min(range(len(ys)), key=lambda i: ys[i])
        ax.scatter([x_pos[best_i]], [ys[best_i]], s=120, zorder=5,
                   color=color, edgecolor="black", linewidth=1.2)
        ax.annotate(
            f"best: bs={block_sizes[best_i]}\n({ys[best_i]:.2f} ms)",
            xy=(x_pos[best_i], ys[best_i]),
            xytext=(x_pos[best_i] + 0.15, ys[best_i] + (0.45 if tier == 0 else -0.65)),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            fontsize=8.5, fontweight="bold", color=color
        )

    ax.set_xlabel("Block Size (threads/block)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Kernel Time (ms)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in block_sizes])
    ax.set_ylim(1.2, 9.6)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Dynamic Shared Memory / Block", fontsize=9.5, title_fontsize=9.5)
    ax.set_title(
        "SAXPY Execution Time Across Dynamic Shared-Memory Tiers\n"
        "RTX 4050 Laptop GPU (sm_89), N = 33,554,432",
        fontsize=12, fontweight="bold", pad=12
    )
    fig.tight_layout()
    fig.savefig(out_time, dpi=300)
    plt.close(fig)

    # Plot 2B: Companion Theoretical Occupancy across Dynamic Shared Memory tiers
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    for tier in tiers:
        sub = {int(r["block_size"]): float(r["theoretical_occupancy_pct"])
               for r in dyn_rows if int(r["dyn_shared_bytes"]) == tier}
        ys = [sub[b] for b in block_sizes]
        color, marker, label = palette.get(tier, ("#333333", "o", f"{tier} B"))
        ax.plot(x_pos, ys, marker=marker, linewidth=2.2, markersize=7,
                color=color, label=label)
        for xi, yi in zip(x_pos, ys):
            offset_y = 3.0 if tier != 16384 or xi != 4 else -6.5
            ax.text(xi, yi + offset_y, f"{yi:.1f}%", ha="center",
                    fontsize=7.5, color=color, fontweight="bold")

    ax.set_xlabel("Block Size (threads/block)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Theoretical Occupancy (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in block_sizes])
    ax.set_ylim(0, 114)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Dynamic Shared Memory / Block", fontsize=9.5, title_fontsize=9.5,
              loc="lower right")
    ax.set_title(
        "Theoretical Occupancy Across Dynamic Shared-Memory Tiers\n"
        "RTX 4050 Laptop GPU (sm_89), 100 KB Shared Mem / SM",
        fontsize=12, fontweight="bold", pad=12
    )
    fig.tight_layout()
    fig.savefig(out_occ, dpi=300)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    baseline_csv = root / "results" / "saxpy_block_size_sweep.csv"
    dyn_csv = root / "results" / "saxpy_dyn_shared_sweep.csv"
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = load_csv(baseline_csv)
    dyn_rows = load_csv(dyn_csv)

    out_p1 = plots_dir / "baseline_time_and_occupancy.png"
    out_p2_time = plots_dir / "dyn_shared_time_comparison.png"
    out_p2_occ = plots_dir / "dyn_shared_occupancy_comparison.png"

    plot_baseline(baseline_rows, out_p1)
    plot_dyn_shared(dyn_rows, out_p2_time, out_p2_occ)

    print(f"Generated: {out_p1}")
    print(f"Generated: {out_p2_time}")
    print(f"Generated: {out_p2_occ}")


if __name__ == "__main__":
    main()
