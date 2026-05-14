"""
Pruebas del layout de imports del dashboard (sin levantar Streamlit).

Verifica que la raíz del repositorio se inyecta en ``sys.path`` de forma
que ``steiner`` sea importable cuando el proceso no parte desde la raíz.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class TestDashboardImports(unittest.TestCase):
    def test_steiner_importable_with_repo_root_on_path(self) -> None:
        """Simula ejecución tipo Streamlit: cwd en dashboard, path con raíz."""
        code = """
import os, sys
repo = os.environ["STEINER_REPO_ROOT"]
sys.path.insert(0, repo)
import steiner  # noqa: F401
print("ok")
"""
        env = os.environ.copy()
        env["STEINER_REPO_ROOT"] = str(REPO)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO / "dashboard"),
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)

    def test_app_py_declares_repo_root_injection(self) -> None:
        text = (REPO / "dashboard" / "app.py").read_text(encoding="utf-8")
        self.assertIn("sys.path.insert", text)
        self.assertIn("parents[1]", text)


if __name__ == "__main__":
    unittest.main()
