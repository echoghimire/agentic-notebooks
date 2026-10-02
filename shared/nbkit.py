"""Build-time helpers for the notebooks that use the shared studio_http kit.

Only build_notebook.py scripts import this. The generated notebooks never do: writefile() pastes
each source file (including shared/studio_http.py, kit.css and kit.js) into a notebook cell, so
every .ipynb stays self-contained.
"""
import json
import os

SHARED = os.path.dirname(os.path.abspath(__file__))
KIT_FILES = [("studio_http.py", "Shared web kit: password, MCP endpoint, proxy, process helpers"),
             ("kit.css", "Shared page styles"),
             ("kit.js", "Shared page scripts")]


class Notebook:
    def __init__(self, app_dir):
        self.app_dir = app_dir
        self.cells = []

    def md(self, src):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})

    def code(self, src):
        self.cells.append({"cell_type": "code", "metadata": {"trusted": True}, "source": src.strip("\n"),
                           "outputs": [], "execution_count": None})

    def writefile(self, path, title=None):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            body = f.read()
        assert "'''" not in body and not body.endswith("\\"), "%s cannot be embedded in r'''...'''" % path
        self.code("# %s\n# Writes %s into APP_DIR. Safe to re-run.\nimport os\nAPP_DIR = %r\n"
                  "os.makedirs(APP_DIR, exist_ok=True)\n"
                  "with open(os.path.join(APP_DIR, %r), \"w\", encoding=\"utf-8\") as f:\n"
                  "    f.write(r'''%s''')\nprint(\"wrote\", os.path.join(APP_DIR, %r))"
                  % (title or name, name, self.app_dir, name, body, name))

    def kit_files(self):
        for name, title in KIT_FILES:
            self.writefile(os.path.join(SHARED, name), title)

    def save(self, path, gpu=True):
        path = os.path.normpath(path)
        nb = {"metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                           "language_info": {"name": "python", "version": "3.11"},
                           "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "dataSources": [],
                                      "isInternetEnabled": True, "language": "python", "sourceType": "notebook",
                                      "isGpuEnabled": gpu}},
              "nbformat": 4, "nbformat_minor": 4, "cells": self.cells}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print("wrote", path, len(self.cells), "cells")


def setup_cell(app_dir, notebook_title):
    """Code for the cell that makes the written app files importable and checks they exist."""
    return """import os, sys
APP_DIR = %r
_missing = [f for f in ("studio_http.py", "kit.css", "kit.js") if not os.path.exists(os.path.join(APP_DIR, f))]
if _missing:
    raise RuntimeError("Run the 'Writes ... into APP_DIR' cells above first (missing: %%s)" %% ", ".join(_missing))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
import studio_http as K   # %s helpers: kaggle_secret, spawn, wait_http, tail, start_tunnel, keep_alive""" % (
        app_dir, notebook_title)


PIP_KEEP_CORE = r'''
def pip_install(*args, keep_core=True):
    """pip install, quietly. keep_core pins Kaggle's own torch / numpy builds so nothing replaces them."""
    cons = "/tmp/keep_core.txt"
    lines = []
    if keep_core:
        import importlib.metadata as md
        for pkg in ("torch", "torchvision", "torchaudio", "numpy"):
            try:
                lines.append("%s==%s" % (pkg, md.version(pkg)))
            except md.PackageNotFoundError:
                pass
    open(cons, "w").write("\n".join(lines) + "\n")
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-c", cons, *args]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:])
        raise RuntimeError("pip install failed: " + " ".join(args))
'''.strip("\n")
