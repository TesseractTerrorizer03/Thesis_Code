"""Execute the published notebooks with this interpreter, from the project root."""
from pathlib import Path
import argparse
import os
import sys


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = {
    "gmm": ROOT / "GMM" / "Size_Direction_Comparison.ipynb",
    "umap": ROOT / "Spherical_UMAP_Analysis.ipynb",
}


def execute(path):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"{path.name}: executed cell {cell_index}", flush=True)

    client = NotebookClient(
        notebook, km=manager, timeout=None, allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_executed=progress,
    )
    try:
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(notebook, path)
    nbformat.validate(notebook)
    print(f"Saved {path.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=["all", *NOTEBOOKS], default="all")
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    selected = NOTEBOOKS.values() if args.notebook == "all" else [NOTEBOOKS[args.notebook]]
    for path in selected:
        execute(path)


if __name__ == "__main__":
    main()
