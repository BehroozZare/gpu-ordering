import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread


def _sparsity_ratio_percent(sparse_mat) -> float:
    n = sparse_mat.shape[0] * sparse_mat.shape[1]
    if n == 0:
        return 0.0
    return (1.0 - (sparse_mat.count_nonzero() / n)) * 100.0


def _save_spy_plot(mat, out_path: Path, title: str, markersize: float = 0.5) -> None:
    plt.figure(figsize=(8, 8))
    plt.spy(mat, markersize=markersize)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def _read_perm_vector(txt_path: Path) -> np.ndarray:
    # File is written by RXMESH_SOLVER::save_vector_to_file(perm, ...).
    # We accept whitespace or newline-separated ints.
    data = np.loadtxt(txt_path, dtype=np.int64)
    if data.ndim == 0:
        data = np.array([int(data)], dtype=np.int64)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize a single original matrix and its permuted version (and permutation vector) saved in output/render_data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "output" / "render_data")),
        help="Directory containing <mesh>_orig.mtx and <mesh>_<ORDERING>_perm.mtx files.",
    )
    parser.add_argument("--mesh", type=str, default="beetle", help="Mesh stem, e.g. 'beetle'.")
    parser.add_argument(
        "--ordering",
        type=str,
        default="PARTH",
        help="Ordering tag used in filenames, e.g. PARTH or PATCH_ORDERING.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "Figures"),
        help="Where to write PNG outputs (default: ./Figures).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = args.mesh
    ordering = args.ordering

    orig_path = data_dir / f"{mesh}_orig.mtx"
    perm_mtx_path = data_dir / f"{mesh}_{ordering}_perm.mtx"
    perm_vec_path = data_dir / f"{mesh}_{ordering}_perm.txt"

    if not orig_path.exists():
        raise FileNotFoundError(f"Missing original matrix: {orig_path}")
    if not perm_mtx_path.exists():
        raise FileNotFoundError(f"Missing permuted matrix: {perm_mtx_path}")

    print(f"Reading orig: {orig_path}")
    A = mmread(str(orig_path)).tocsc()
    print(f"A shape: {A.shape}, nnz: {A.count_nonzero()}, sparsity: {_sparsity_ratio_percent(A):.4f}%")

    print(f"Reading permuted: {perm_mtx_path}")
    A_perm = mmread(str(perm_mtx_path)).tocsc()
    print(
        f"A_perm shape: {A_perm.shape}, nnz: {A_perm.count_nonzero()}, sparsity: {_sparsity_ratio_percent(A_perm):.4f}%"
    )

    _save_spy_plot(A, out_dir / f"{mesh}_orig_spy.png", title=f"{mesh}: orig")
    _save_spy_plot(A_perm, out_dir / f"{mesh}_{ordering}_perm_spy.png", title=f"{mesh}: {ordering} perm")

    if perm_vec_path.exists():
        print(f"Reading perm vector: {perm_vec_path}")
        p = _read_perm_vector(perm_vec_path)
        print(f"perm size: {p.size}, min: {p.min()}, max: {p.max()}")

        # Visualize permutation as a point plot: (i, p[i]).
        plt.figure(figsize=(8, 8))
        plt.plot(np.arange(p.size), p, ".", markersize=0.5)
        plt.xlabel("i")
        plt.ylabel("p[i]")
        plt.tight_layout()
        perm_plot_path = out_dir / f"{mesh}_{ordering}_perm_vector.png"
        plt.savefig(perm_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved permutation plot to {perm_plot_path}")
    else:
        print(f"Permutation vector not found (optional): {perm_vec_path}")

    print(f"Saved spy plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())