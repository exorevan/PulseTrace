from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from pulsetrace.adapters.sklearn import SklearnAdapter
from pulsetrace.config.schema import ExplainerConfig
from pulsetrace.data.dataset import Dataset
from pulsetrace.explainers.lime import LimeExplainer
from pulsetrace.explainers.result import ExplanationResult
from pulsetrace.explainers.shap import ShapExplainer


# ---- Fixtures ----

@pytest.fixture(scope="module")
def iris_adapter_and_dataset():
    X, y = load_iris(return_X_y=True)
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    classes = np.unique(y)
    dataset = Dataset(
        X=X.astype(float),
        y=y.astype(object),
        feature_names=feature_names,
        target_name="species",
        classes=classes,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    adapter = SklearnAdapter(model)
    return adapter, dataset


@pytest.fixture(scope="module")
def regression_adapter_and_dataset():
    rng = np.random.RandomState(42)
    X = rng.rand(60, 4)
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.randn(60) * 0.05
    feature_names = ["f1", "f2", "f3", "f4"]
    dataset = Dataset(
        X=X,
        y=y.astype(object),
        feature_names=feature_names,
        target_name="value",
        classes=None,
    )
    model = LinearRegression().fit(X, y)
    adapter = SklearnAdapter(model)
    return adapter, dataset


_LIME_CFG = ExplainerConfig(type="lime", num_features=4, num_samples=100)


# ---- LimeExplainer ----

class TestLimeGlobal:
    def test_returns_explanation_result(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        result = LimeExplainer(_LIME_CFG).explain_global(adapter, dataset)
        assert isinstance(result, ExplanationResult)

    def test_mode_is_global(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        result = LimeExplainer(_LIME_CFG).explain_global(adapter, dataset)
        assert result.mode == "global"

    def test_method_is_lime(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        result = LimeExplainer(_LIME_CFG).explain_global(adapter, dataset)
        assert result.method == "lime"

    def test_classification_contributions_have_labels(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        result = LimeExplainer(_LIME_CFG).explain_global(adapter, dataset)
        assert all(c.label is not None for c in result.contributions)

    def test_regression_contributions_have_no_labels(self, regression_adapter_and_dataset):
        adapter, dataset = regression_adapter_and_dataset
        result = LimeExplainer(_LIME_CFG).explain_global(adapter, dataset)
        assert result.task == "regression"
        assert all(c.label is None for c in result.contributions)

    def test_has_contributions(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        result = LimeExplainer(_LIME_CFG).explain_global(adapter, dataset)
        assert len(result.contributions) > 0


class TestLimeLocal:
    def test_returns_explanation_result(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        instance = Dataset(
            X=dataset.X[:1],
            y=np.array([]),
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            classes=dataset.classes,
        )
        result = LimeExplainer(_LIME_CFG).explain_local(adapter, instance, dataset)
        assert isinstance(result, ExplanationResult)
        assert result.mode == "local"

    def test_classification_local_has_label(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        instance = Dataset(
            X=dataset.X[:1],
            y=np.array([]),
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            classes=dataset.classes,
        )
        result = LimeExplainer(_LIME_CFG).explain_local(adapter, instance, dataset)
        assert all(c.label is not None for c in result.contributions)

    def test_regression_local_no_label(self, regression_adapter_and_dataset):
        adapter, dataset = regression_adapter_and_dataset
        instance = Dataset(
            X=dataset.X[:1],
            y=np.array([]),
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            classes=None,
        )
        result = LimeExplainer(_LIME_CFG).explain_local(adapter, instance, dataset)
        assert all(c.label is None for c in result.contributions)


# ---- ShapExplainer ----

_SHAP_CFG = ExplainerConfig(type="shap", num_features=4)


class TestShapGlobal:
    def test_returns_explanation_result(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        result = ShapExplainer(_SHAP_CFG).explain_global(adapter, dataset)
        assert isinstance(result, ExplanationResult)

    def test_method_is_shap(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        result = ShapExplainer(_SHAP_CFG).explain_global(adapter, dataset)
        assert result.method == "shap"

    def test_classification_contributions_have_labels(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        result = ShapExplainer(_SHAP_CFG).explain_global(adapter, dataset)
        assert all(c.label is not None for c in result.contributions)

    def test_regression_contributions_no_labels(self, regression_adapter_and_dataset):
        adapter, dataset = regression_adapter_and_dataset
        result = ShapExplainer(_SHAP_CFG).explain_global(adapter, dataset)
        assert result.task == "regression"
        assert all(c.label is None for c in result.contributions)

    def test_regression_weights_are_non_negative(self, regression_adapter_and_dataset):
        """Global SHAP uses mean absolute values — all weights should be >= 0."""
        adapter, dataset = regression_adapter_and_dataset
        result = ShapExplainer(_SHAP_CFG).explain_global(adapter, dataset)
        assert all(c.weight >= 0 for c in result.contributions)


class TestShapLocal:
    def test_returns_explanation_result(self, iris_adapter_and_dataset):
        adapter, dataset = iris_adapter_and_dataset
        instance = Dataset(
            X=dataset.X[:1],
            y=np.array([]),
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            classes=dataset.classes,
        )
        result = ShapExplainer(_SHAP_CFG).explain_local(adapter, instance, dataset)
        assert isinstance(result, ExplanationResult)
        assert result.mode == "local"

    def test_correct_number_of_contributions(self, regression_adapter_and_dataset):
        adapter, dataset = regression_adapter_and_dataset
        instance = Dataset(
            X=dataset.X[:1],
            y=np.array([]),
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            classes=None,
        )
        result = ShapExplainer(_SHAP_CFG).explain_local(adapter, instance, dataset)
        assert len(result.contributions) <= _SHAP_CFG.num_features


class TestShapFallback:
    """SHAP sometimes returns 2D (n_samples, n_features) for classifiers instead of the
    expected 3D (n_samples, n_features, n_classes). Both paths must produce valid output."""

    def _mock_shap(self, values_array: np.ndarray):
        """Return a mock shap.Explainer that always yields values_array."""
        fake_shap_values = MagicMock()
        fake_shap_values.values = values_array
        mock_explainer_instance = MagicMock(return_value=fake_shap_values)
        return mock_explainer_instance

    def test_global_2d_classification_no_crash(self, iris_adapter_and_dataset):
        """Global explain: 2D SHAP output for a classifier must not crash and
        must produce contributions with label=None (per-class info unavailable)."""
        adapter, dataset = iris_adapter_and_dataset
        n_features = len(dataset.feature_names)
        fake = self._mock_shap(np.random.rand(10, n_features))

        with patch("pulsetrace.explainers.shap.shap.Explainer", return_value=fake):
            result = ShapExplainer(_SHAP_CFG).explain_global(adapter, dataset)

        assert isinstance(result, ExplanationResult)
        assert result.task == "classification"
        assert len(result.contributions) > 0
        assert all(c.label is None for c in result.contributions)

    def test_global_2d_classification_respects_num_features(self, iris_adapter_and_dataset):
        """Global 2D fallback must still honour num_features cap."""
        adapter, dataset = iris_adapter_and_dataset
        n_features = len(dataset.feature_names)
        fake = self._mock_shap(np.random.rand(10, n_features))

        with patch("pulsetrace.explainers.shap.shap.Explainer", return_value=fake):
            result = ShapExplainer(_SHAP_CFG).explain_global(adapter, dataset)

        assert len(result.contributions) <= _SHAP_CFG.num_features

    def test_local_1d_classification_no_crash(self, iris_adapter_and_dataset):
        """Local explain: 1D SHAP output per instance for a classifier must not crash
        and must produce contributions with label=None."""
        adapter, dataset = iris_adapter_and_dataset
        n_features = len(dataset.feature_names)
        instance = Dataset(
            X=dataset.X[:1],
            y=np.array([]),
            feature_names=dataset.feature_names,
            target_name=dataset.target_name,
            classes=dataset.classes,
        )
        # After values[0] slicing: (1, n_features)[0] → (n_features,) — 1D
        fake = self._mock_shap(np.random.rand(1, n_features))

        with patch("pulsetrace.explainers.shap.shap.Explainer", return_value=fake):
            result = ShapExplainer(_SHAP_CFG).explain_local(adapter, instance, dataset)

        assert isinstance(result, ExplanationResult)
        assert result.task == "classification"
        assert len(result.contributions) > 0
        assert all(c.label is None for c in result.contributions)
