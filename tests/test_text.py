"""Tests for text data support — config, loader, dataset, adapter, explainers."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Task 1 tests: Config
# ---------------------------------------------------------------------------

class TestTextDatasetConfig:
    def test_valid(self, tmp_path: Path) -> None:
        from pulsetrace.config.schema import TextDatasetConfig
        cfg = TextDatasetConfig(type="text", path=tmp_path)
        assert cfg.type == "text"
        assert cfg.only_x is False

    def test_only_x(self, tmp_path: Path) -> None:
        from pulsetrace.config.schema import TextDatasetConfig
        cfg = TextDatasetConfig(type="text", path=tmp_path, only_x=True)
        assert cfg.only_x is True

    def test_rejects_unknown_fields(self, tmp_path: Path) -> None:
        from pydantic import ValidationError
        from pulsetrace.config.schema import TextDatasetConfig
        with pytest.raises(ValidationError):
            TextDatasetConfig(type="text", path=tmp_path, unknown_field="x")  # type: ignore[call-arg]

    def test_in_dataset_union(self, tmp_path: Path) -> None:
        from pulsetrace.config.schema import DatasetConfig
        from pydantic import TypeAdapter
        ta: TypeAdapter[DatasetConfig] = TypeAdapter(DatasetConfig)
        cfg = ta.validate_python({"type": "text", "path": str(tmp_path)})
        from pulsetrace.config.schema import TextDatasetConfig
        assert isinstance(cfg, TextDatasetConfig)


class TestHfModelConfig:
    def test_valid(self) -> None:
        from pulsetrace.config.schema import HfModelConfig
        cfg = HfModelConfig(
            type="hf",
            path_or_name="distilbert-base-uncased-finetuned-sst-2-english",
            labels=["negative", "positive"],
        )
        assert cfg.type == "hf"
        assert cfg.labels == ["negative", "positive"]

    def test_in_model_union(self) -> None:
        from pulsetrace.config.schema import ModelConfig
        from pydantic import TypeAdapter
        ta: TypeAdapter[ModelConfig] = TypeAdapter(ModelConfig)
        cfg = ta.validate_python({
            "type": "hf",
            "path_or_name": "some/model",
            "labels": ["a", "b"],
        })
        from pulsetrace.config.schema import HfModelConfig
        assert isinstance(cfg, HfModelConfig)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def text_dir(tmp_path: Path) -> Path:
    """Two-class text dataset in subdirectory layout."""
    pos = tmp_path / "positive"
    neg = tmp_path / "negative"
    pos.mkdir()
    neg.mkdir()
    (pos / "s1.txt").write_text("This movie is great and wonderful", encoding="utf-8")
    (pos / "s2.txt").write_text("Fantastic film loved every minute", encoding="utf-8")
    (neg / "s1.txt").write_text("Terrible film hated it completely", encoding="utf-8")
    (neg / "s2.txt").write_text("Boring and awful waste of time", encoding="utf-8")
    return tmp_path


@pytest.fixture
def local_dir(tmp_path: Path) -> Path:
    """Flat directory for local (only_x) mode."""
    flat = tmp_path / "local"
    flat.mkdir()
    (flat / "sample.txt").write_text("What an interesting experience", encoding="utf-8")
    return flat


# ---------------------------------------------------------------------------
# Task 2 tests: TextDataset + loader
# ---------------------------------------------------------------------------

class TestTextDataset:
    def test_subclass_of_dataset(self, text_dir: Path) -> None:
        from pulsetrace.data.dataset import Dataset
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        assert isinstance(ds, Dataset)

    def test_data_type(self, text_dir: Path) -> None:
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        assert ds.data_type == "text"

    def test_texts_field(self, text_dir: Path) -> None:
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        assert len(ds.texts) == 4
        assert all(isinstance(t, str) for t in ds.texts)

    def test_classes_sorted(self, text_dir: Path) -> None:
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        assert list(ds.classes) == ["negative", "positive"]

    def test_X_shape(self, text_dir: Path) -> None:
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        assert ds.X.shape == (4, 1)

    def test_y_labels(self, text_dir: Path) -> None:
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        assert set(ds.y) == {"positive", "negative"}

    def test_only_x_mode(self, local_dir: Path) -> None:
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        ds = load_text_dataset(TextDatasetConfig(type="text", path=local_dir, only_x=True))
        assert ds.classes is None
        assert len(ds.texts) == 1
        assert ds.y.size == 0

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        with pytest.raises(FileNotFoundError):
            load_text_dataset(TextDatasetConfig(type="text", path=tmp_path / "nonexistent"))

    def test_load_dataset_routes_text(self, text_dir: Path) -> None:
        from pulsetrace.data import load_dataset
        from pulsetrace.config.schema import TextDatasetConfig
        from pulsetrace.data.text_dataset import TextDataset
        ds = load_dataset(TextDatasetConfig(type="text", path=text_dir))
        assert isinstance(ds, TextDataset)


# ---------------------------------------------------------------------------
# Task 3 tests: HfAdapter (slow — requires network + model download)
# ---------------------------------------------------------------------------

class TestHfAdapter:
    @pytest.mark.slow
    def test_predict_proba_shape(self) -> None:
        from pulsetrace.adapters.huggingface import HfAdapter
        from pulsetrace.config.schema import HfModelConfig
        cfg = HfModelConfig(
            type="hf",
            path_or_name="distilbert-base-uncased-finetuned-sst-2-english",
            labels=["negative", "positive"],
        )
        adapter = HfAdapter(cfg)
        texts = ["Great movie!", "Terrible film."]
        probs = adapter.predict_proba(texts)
        assert probs.shape == (2, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.slow
    def test_predict_shape(self) -> None:
        from pulsetrace.adapters.huggingface import HfAdapter
        from pulsetrace.config.schema import HfModelConfig
        cfg = HfModelConfig(
            type="hf",
            path_or_name="distilbert-base-uncased-finetuned-sst-2-english",
            labels=["negative", "positive"],
        )
        adapter = HfAdapter(cfg)
        preds = adapter.predict(["Great!", "Terrible."])
        assert preds.shape == (2,)

    @pytest.mark.slow
    def test_task_is_classification(self) -> None:
        from pulsetrace.adapters.huggingface import HfAdapter
        from pulsetrace.config.schema import HfModelConfig
        cfg = HfModelConfig(
            type="hf",
            path_or_name="distilbert-base-uncased-finetuned-sst-2-english",
            labels=["negative", "positive"],
        )
        adapter = HfAdapter(cfg)
        assert adapter.task == "classification"

    @pytest.mark.slow
    def test_exposes_tokenizer(self) -> None:
        from pulsetrace.adapters.huggingface import HfAdapter
        from pulsetrace.config.schema import HfModelConfig
        cfg = HfModelConfig(
            type="hf",
            path_or_name="distilbert-base-uncased-finetuned-sst-2-english",
            labels=["negative", "positive"],
        )
        adapter = HfAdapter(cfg)
        assert hasattr(adapter, "tokenizer")
        assert callable(adapter.tokenizer)


# ---------------------------------------------------------------------------
# Task 4 tests: text_panel
# ---------------------------------------------------------------------------

class TestTextPanel:
    def test_returns_image_panel(self) -> None:
        from pulsetrace.explainers.text_utils import text_panel
        from pulsetrace.explainers.result import ImagePanel
        panel = text_panel(
            "This movie is great",
            [("great", 0.6), ("movie", -0.3)],
            title="Test panel",
        )
        assert isinstance(panel, ImagePanel)

    def test_original_b64_nonempty(self) -> None:
        from pulsetrace.explainers.text_utils import text_panel
        panel = text_panel("Good film", [("Good", 0.5)], title="T")
        assert len(panel.original_b64) > 100

    def test_explanation_b64_nonempty(self) -> None:
        from pulsetrace.explainers.text_utils import text_panel
        panel = text_panel("Good film", [("Good", 0.5)], title="T")
        assert len(panel.explanation_b64) > 100

    def test_confidence_stored(self) -> None:
        from pulsetrace.explainers.text_utils import text_panel
        panel = text_panel("Good film", [("Good", 0.5)], title="T", confidence=0.87)
        assert panel.confidence == pytest.approx(0.87)

    def test_no_confidence_is_none(self) -> None:
        from pulsetrace.explainers.text_utils import text_panel
        panel = text_panel("Good film", [("Good", 0.5)], title="T")
        assert panel.confidence is None

    def test_empty_weights_still_renders(self) -> None:
        from pulsetrace.explainers.text_utils import text_panel
        panel = text_panel("Some text here", [], title="T")
        assert panel.original_b64
        assert panel.explanation_b64


# ---------------------------------------------------------------------------
# Shared mock adapter (used by Task 5 and Task 6)
# ---------------------------------------------------------------------------

class _MockHfAdapter:
    """Minimal HfAdapter stand-in for fast unit tests — no real model loaded."""

    task = "classification"
    labels = ["negative", "positive"]

    class _Tok:
        mask_token = "[MASK]"
        mask_token_id = 0

        def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
            return {"input_ids": [[1, 2, 3] for _ in texts]}

        def convert_ids_to_tokens(self, ids):  # type: ignore[no-untyped-def]
            return [f"tok{i}" for i in ids]

        def decode(self, ids, **kwargs):  # type: ignore[no-untyped-def]
            return " ".join(f"tok{i}" for i in ids)

    tokenizer = _Tok()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.3
        probs[:, 1] = 0.7
        return probs

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1).astype(float)


# ---------------------------------------------------------------------------
# Task 5 tests: LIME text explainer
# ---------------------------------------------------------------------------

class TestLimeText:
    def test_global_returns_image_panels(self, text_dir: Path) -> None:
        from pulsetrace.config.schema import ExplainerConfig, TextDatasetConfig
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.explainers.lime import LimeExplainer
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        cfg = ExplainerConfig(type="lime", global_samples=2, num_features=3, num_samples=50)
        exp = LimeExplainer(cfg)
        result = exp.explain_global(_MockHfAdapter(), ds)
        assert result.image_panels is not None
        assert len(result.image_panels) == 2

    def test_global_base_values_keys_are_strings(self, text_dir: Path) -> None:
        from pulsetrace.config.schema import ExplainerConfig, TextDatasetConfig
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.explainers.lime import LimeExplainer
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        cfg = ExplainerConfig(type="lime", global_samples=2, num_features=3, num_samples=50)
        result = LimeExplainer(cfg).explain_global(_MockHfAdapter(), ds)
        assert result.base_values is not None
        for k in result.base_values:
            assert isinstance(k, str)

    def test_global_contributions_nonempty(self, text_dir: Path) -> None:
        from pulsetrace.config.schema import ExplainerConfig, TextDatasetConfig
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.explainers.lime import LimeExplainer
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        cfg = ExplainerConfig(type="lime", global_samples=2, num_features=3, num_samples=50)
        result = LimeExplainer(cfg).explain_global(_MockHfAdapter(), ds)
        assert len(result.contributions) > 0

    def test_local_returns_one_panel(self, text_dir: Path, local_dir: Path) -> None:
        from pulsetrace.config.schema import ExplainerConfig, TextDatasetConfig
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.explainers.lime import LimeExplainer
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        inst = load_text_dataset(TextDatasetConfig(type="text", path=local_dir, only_x=True))
        cfg = ExplainerConfig(type="lime", num_features=3, num_samples=50)
        result = LimeExplainer(cfg).explain_local(_MockHfAdapter(), inst, ds)
        assert result.image_panels is not None
        assert len(result.image_panels) == 1

    def test_local_mode_string(self, text_dir: Path, local_dir: Path) -> None:
        from pulsetrace.config.schema import ExplainerConfig, TextDatasetConfig
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.explainers.lime import LimeExplainer
        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        inst = load_text_dataset(TextDatasetConfig(type="text", path=local_dir, only_x=True))
        cfg = ExplainerConfig(type="lime", num_features=3, num_samples=50)
        result = LimeExplainer(cfg).explain_local(_MockHfAdapter(), inst, ds)
        assert result.mode == "local"
        assert result.method == "lime"


# ---------------------------------------------------------------------------
# Task 6 tests: SHAP text explainer (uses monkeypatching for speed)
# ---------------------------------------------------------------------------

class TestShapText:
    @pytest.fixture
    def mock_shap_explanation(self) -> object:
        """Mock shap.Explanation returned by shap.Explainer(texts)."""
        class _Exp:
            # shape: (n_texts=2, n_tokens=5, n_classes=2)
            values = np.random.default_rng(0).standard_normal((2, 5, 2))
            data = [
                np.array(["This", "movie", "is", "great", "wonderful"]),
                np.array(["Terrible", "film", "hated", "it", "completely"]),
            ]

        return _Exp()

    @pytest.fixture
    def mock_shap_explainer_cls(self, mock_shap_explanation: object):
        """Factory that returns a mock shap.Explainer instance."""
        class _Explainer:
            expected_value = np.array([0.3, 0.7])

            def __call__(self, texts, silent=True):  # type: ignore[no-untyped-def]
                return mock_shap_explanation

        def _factory(*args, **kwargs):  # type: ignore[no-untyped-def]
            return _Explainer()

        return _factory

    def test_global_returns_image_panels(
        self, text_dir: Path, mock_shap_explainer_cls, monkeypatch
    ) -> None:
        import shap
        monkeypatch.setattr(shap, "Explainer", mock_shap_explainer_cls)
        monkeypatch.setattr(shap.maskers, "Text", lambda *a, **kw: None)

        from pulsetrace.config.schema import ExplainerConfig, TextDatasetConfig
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.explainers.shap import ShapExplainer

        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        cfg = ExplainerConfig(type="shap", global_samples=2, num_features=3)
        result = ShapExplainer(cfg).explain_global(_MockHfAdapter(), ds)
        assert result.image_panels is not None
        assert len(result.image_panels) == 2

    def test_global_base_values_present(
        self, text_dir: Path, mock_shap_explainer_cls, monkeypatch
    ) -> None:
        import shap
        monkeypatch.setattr(shap, "Explainer", mock_shap_explainer_cls)
        monkeypatch.setattr(shap.maskers, "Text", lambda *a, **kw: None)

        from pulsetrace.config.schema import ExplainerConfig, TextDatasetConfig
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.explainers.shap import ShapExplainer

        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        cfg = ExplainerConfig(type="shap", global_samples=2, num_features=3)
        result = ShapExplainer(cfg).explain_global(_MockHfAdapter(), ds)
        assert result.base_values is not None
        assert len(result.base_values) == 2

    def test_local_returns_one_panel(
        self, text_dir: Path, local_dir: Path, monkeypatch
    ) -> None:
        import shap

        class _SingleExp:
            values = np.random.default_rng(1).standard_normal((1, 5, 2))
            data = [np.array(["What", "an", "interesting", "experience", "here"])]

        class _Explainer:
            expected_value = np.array([0.4, 0.6])
            def __call__(self, texts, silent=True):  # type: ignore[no-untyped-def]
                return _SingleExp()

        monkeypatch.setattr(shap, "Explainer", lambda *a, **kw: _Explainer())
        monkeypatch.setattr(shap.maskers, "Text", lambda *a, **kw: None)

        from pulsetrace.config.schema import ExplainerConfig, TextDatasetConfig
        from pulsetrace.data.text_loader import load_text_dataset
        from pulsetrace.explainers.shap import ShapExplainer

        ds = load_text_dataset(TextDatasetConfig(type="text", path=text_dir))
        inst = load_text_dataset(TextDatasetConfig(type="text", path=local_dir, only_x=True))
        cfg = ExplainerConfig(type="shap", num_features=3)
        result = ShapExplainer(cfg).explain_local(_MockHfAdapter(), inst, ds)
        assert result.image_panels is not None
        assert len(result.image_panels) == 1
        assert result.mode == "local"
