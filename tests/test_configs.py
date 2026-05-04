"""Integration tests: run every real YAML config through the full pipeline.

These tests call main.run() end-to-end (load model → explain → render) and
assert the output contains the expected header tokens.

Mark: configs that are slow (e.g. global_samples=500) are tagged @pytest.mark.slow
so they can be skipped during fast runs with:

    uv run pytest -m "not slow"
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pulsetrace.main import run

# Root of the project (two levels up from this file)
ROOT = Path(__file__).parent.parent


def _cfg(name: str) -> Path:
    return ROOT / "configs" / name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_and_capture(cfg_name: str, capsys) -> str:
    run(_cfg(cfg_name))
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# sklearn + LIME  (global)
# ---------------------------------------------------------------------------

class TestSklearnLimeGlobal:
    @pytest.mark.slow
    def test_iris_global(self, capsys):
        out = _run_and_capture("sklearn_iris_lime.yaml", capsys)
        assert "LIME" in out
        assert "mode=global" in out
        assert "task=classification" in out
        assert "Samples:" in out
        assert "Class:" in out

    def test_house_global(self, capsys):
        out = _run_and_capture("sklearn_house_lime.yaml", capsys)
        assert "LIME" in out
        assert "mode=global" in out
        assert "task=regression" in out
        assert "Base value:" in out

    def test_alzheimer_global(self, capsys):
        out = _run_and_capture("sklearn_alzheimer_lime.yaml", capsys)
        assert "LIME" in out
        assert "mode=global" in out
        assert "task=classification" in out
        assert "Samples:" in out
        assert "Class:" in out


# ---------------------------------------------------------------------------
# sklearn + LIME  (local)
# ---------------------------------------------------------------------------

class TestSklearnLimeLocal:
    def test_iris_local(self, capsys):
        out = _run_and_capture("sklearn_iris_lime_local.yaml", capsys)
        assert "LIME" in out
        assert "mode=local" in out
        assert "task=classification" in out
        assert "Class:" in out

    def test_house_local(self, capsys):
        out = _run_and_capture("sklearn_house_lime_local.yaml", capsys)
        assert "LIME" in out
        assert "mode=local" in out
        assert "task=regression" in out
        assert "Base value:" in out

    def test_alzheimer_local(self, capsys):
        out = _run_and_capture("sklearn_alzheimer_lime_local.yaml", capsys)
        assert "LIME" in out
        assert "mode=local" in out
        assert "task=classification" in out
        assert "Class:" in out


# ---------------------------------------------------------------------------
# Keras + LIME  (global)
# ---------------------------------------------------------------------------

class TestKerasLimeGlobal:
    def test_house_global(self, capsys):
        out = _run_and_capture("keras_house_lime.yaml", capsys)
        assert "LIME" in out
        assert "mode=global" in out
        assert "task=regression" in out
        assert "Base value:" in out


# ---------------------------------------------------------------------------
# Keras + LIME  (local)
# ---------------------------------------------------------------------------

class TestKerasLimeLocal:
    def test_house_local(self, capsys):
        out = _run_and_capture("keras_house_lime_local.yaml", capsys)
        assert "LIME" in out
        assert "mode=local" in out
        assert "task=regression" in out
        assert "Base value:" in out


# ---------------------------------------------------------------------------
# sklearn + SHAP  (global)
# ---------------------------------------------------------------------------

class TestSklearnShapGlobal:
    def test_iris_global(self, capsys):
        out = _run_and_capture("sklearn_iris_shap.yaml", capsys)
        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=classification" in out
        assert "Class:" in out

    def test_house_global(self, capsys):
        out = _run_and_capture("sklearn_house_shap.yaml", capsys)
        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=regression" in out
        assert "Base value:" in out

    def test_alzheimer_global(self, capsys):
        out = _run_and_capture("sklearn_alzheimer_shap.yaml", capsys)
        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=classification" in out
        assert "Class:" in out


# ---------------------------------------------------------------------------
# sklearn + SHAP  (local)
# ---------------------------------------------------------------------------

class TestSklearnShapLocal:
    def test_iris_local(self, capsys):
        out = _run_and_capture("sklearn_iris_shap_local.yaml", capsys)
        assert "SHAP" in out
        assert "mode=local" in out
        assert "task=classification" in out
        assert "Class:" in out

    def test_house_local(self, capsys):
        out = _run_and_capture("sklearn_house_shap_local.yaml", capsys)
        assert "SHAP" in out
        assert "mode=local" in out
        assert "task=regression" in out
        assert "Base value:" in out

    def test_alzheimer_local(self, capsys):
        out = _run_and_capture("sklearn_alzheimer_shap_local.yaml", capsys)
        assert "SHAP" in out
        assert "mode=local" in out
        assert "task=classification" in out
        assert "Class:" in out


# ---------------------------------------------------------------------------
# Keras + SHAP  (global)
# ---------------------------------------------------------------------------

class TestKerasShapGlobal:
    def test_house_global(self, capsys):
        out = _run_and_capture("keras_house_shap.yaml", capsys)
        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=regression" in out
        assert "Base value:" in out


# ---------------------------------------------------------------------------
# Keras + SHAP  (local)
# ---------------------------------------------------------------------------

class TestKerasShapLocal:
    def test_house_local(self, capsys):
        out = _run_and_capture("keras_house_shap_local.yaml", capsys)
        assert "SHAP" in out
        assert "mode=local" in out
        assert "task=regression" in out
        assert "Base value:" in out


# ---------------------------------------------------------------------------
# Keras + LIME  (image — mnist / cifar10 / fashion_mnist)
# ---------------------------------------------------------------------------

class TestKerasImageLime:
    def test_mnist_global(self, capsys):
        out = _run_and_capture("keras_mnist_lime.yaml", capsys)
        assert "LIME" in out
        assert "mode=global" in out
        assert "task=classification" in out

    def test_mnist_local(self, capsys):
        out = _run_and_capture("keras_mnist_lime_local.yaml", capsys)
        assert "LIME" in out
        assert "mode=local" in out
        assert "task=classification" in out

    def test_cifar10_global(self, capsys):
        out = _run_and_capture("keras_cifar10_lime.yaml", capsys)
        assert "LIME" in out
        assert "mode=global" in out
        assert "task=classification" in out

    def test_cifar10_local(self, capsys):
        out = _run_and_capture("keras_cifar10_lime_local.yaml", capsys)
        assert "LIME" in out
        assert "mode=local" in out
        assert "task=classification" in out

    def test_fashion_mnist_global(self, capsys):
        out = _run_and_capture("keras_fashion_mnist_lime.yaml", capsys)
        assert "LIME" in out
        assert "mode=global" in out
        assert "task=classification" in out

    def test_fashion_mnist_local(self, capsys):
        out = _run_and_capture("keras_fashion_mnist_lime_local.yaml", capsys)
        assert "LIME" in out
        assert "mode=local" in out
        assert "task=classification" in out


# ---------------------------------------------------------------------------
# Keras + SHAP  (image — mnist / cifar10 / fashion_mnist)
# ---------------------------------------------------------------------------

class TestKerasImageShap:
    def test_mnist_global(self, capsys):
        out = _run_and_capture("keras_mnist_shap.yaml", capsys)
        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=classification" in out

    def test_mnist_local(self, capsys):
        out = _run_and_capture("keras_mnist_shap_local.yaml", capsys)
        assert "SHAP" in out
        assert "mode=local" in out
        assert "task=classification" in out

    def test_cifar10_global(self, capsys):
        out = _run_and_capture("keras_cifar10_shap.yaml", capsys)
        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=classification" in out

    def test_cifar10_local(self, capsys):
        out = _run_and_capture("keras_cifar10_shap_local.yaml", capsys)
        assert "SHAP" in out
        assert "mode=local" in out
        assert "task=classification" in out

    def test_fashion_mnist_global(self, capsys):
        out = _run_and_capture("keras_fashion_mnist_shap.yaml", capsys)
        assert "SHAP" in out
        assert "mode=global" in out
        assert "task=classification" in out

    def test_fashion_mnist_local(self, capsys):
        out = _run_and_capture("keras_fashion_mnist_shap_local.yaml", capsys)
        assert "SHAP" in out
        assert "mode=local" in out
        assert "task=classification" in out


# ---------------------------------------------------------------------------
# JSON output format
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_json_classification(self, capsys):
        """JSON output must be parseable and contain expected top-level keys."""
        import json
        from pulsetrace.config import load_config
        from pulsetrace.adapters import build_adapter
        from pulsetrace.data import load_dataset
        from pulsetrace.explainers import build_explainer
        from pulsetrace.output import render

        cfg = load_config(_cfg("sklearn_alzheimer_shap_local.yaml"))
        adapter = build_adapter(cfg.model)
        dataset = load_dataset(cfg.dataset)
        instance = load_dataset(cfg.local.dataset)  # type: ignore[union-attr]
        result = build_explainer(cfg.explainer).explain_local(adapter, instance, dataset)

        capsys.readouterr()  # clear previous output
        render(result, output_format="json")
        out = capsys.readouterr().out

        data = json.loads(out)
        assert data["method"] == "shap"
        assert data["mode"] == "local"
        assert data["task"] == "classification"
        assert isinstance(data["contributions"], list)
        assert len(data["contributions"]) > 0

    def test_json_regression(self, capsys):
        import json
        from pulsetrace.config import load_config
        from pulsetrace.adapters import build_adapter
        from pulsetrace.data import load_dataset
        from pulsetrace.explainers import build_explainer
        from pulsetrace.output import render

        cfg = load_config(_cfg("sklearn_house_shap_local.yaml"))
        adapter = build_adapter(cfg.model)
        dataset = load_dataset(cfg.dataset)
        instance = load_dataset(cfg.local.dataset)  # type: ignore[union-attr]
        result = build_explainer(cfg.explainer).explain_local(adapter, instance, dataset)

        capsys.readouterr()
        render(result, output_format="json")
        out = capsys.readouterr().out

        data = json.loads(out)
        assert data["method"] == "shap"
        assert data["task"] == "regression"
        assert "base_values" in data


# ---------------------------------------------------------------------------
# HTML output format
# ---------------------------------------------------------------------------

class TestHtmlOutput:
    def test_html_classification(self, capsys):
        """HTML output must be written to a self-contained file with expected content."""
        from pathlib import Path

        from pulsetrace.config import load_config
        from pulsetrace.adapters import build_adapter
        from pulsetrace.data import load_dataset
        from pulsetrace.explainers import build_explainer
        from pulsetrace.output import render

        cfg = load_config(_cfg("sklearn_alzheimer_shap_local.yaml"))
        adapter = build_adapter(cfg.model)
        dataset = load_dataset(cfg.dataset)
        instance = load_dataset(cfg.local.dataset)  # type: ignore[union-attr]
        result = build_explainer(cfg.explainer).explain_local(adapter, instance, dataset)

        capsys.readouterr()
        render(result, output_format="html")
        out = capsys.readouterr().out
        assert "HTML saved ->" in out

        filepath = Path(out.strip().split("HTML saved -> ")[-1])
        html = filepath.read_text(encoding="utf-8")

        assert "<!DOCTYPE html>" in html
        assert "shap" in html.lower()
        assert "classification" in html
        assert "bar-fill" in html
        assert len(html) > 500

    def test_html_regression(self, capsys):
        from pathlib import Path

        from pulsetrace.config import load_config
        from pulsetrace.adapters import build_adapter
        from pulsetrace.data import load_dataset
        from pulsetrace.explainers import build_explainer
        from pulsetrace.output import render

        cfg = load_config(_cfg("sklearn_house_shap_local.yaml"))
        adapter = build_adapter(cfg.model)
        dataset = load_dataset(cfg.dataset)
        instance = load_dataset(cfg.local.dataset)  # type: ignore[union-attr]
        result = build_explainer(cfg.explainer).explain_local(adapter, instance, dataset)

        capsys.readouterr()
        render(result, output_format="html")
        out = capsys.readouterr().out
        assert "HTML saved ->" in out

        filepath = Path(out.strip().split("HTML saved -> ")[-1])
        html = filepath.read_text(encoding="utf-8")

        assert "<!DOCTYPE html>" in html
        assert "regression" in html
        assert "bar-fill" in html
        assert "Base value:" in html
