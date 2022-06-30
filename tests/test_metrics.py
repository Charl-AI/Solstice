import solstice
import sklearn.metrics as skmetrics
import jax
import jax.numpy as jnp


def test_classification_metrics():
    """Test `ClassificationMetrics`. Includes the full use case of initialising, merging
    over batches, then computing final metrics. We test the results against
    `sklearn.metrics.classification_report`."""

    num_batches = 2
    keys = jax.random.split(jax.random.PRNGKey(0), num_batches)
    metrics = None
    total_preds = None
    total_labels = None
    for key in keys:
        preds_key, labels_key = jax.random.split(key, 2)
        batch_preds = jax.random.bernoulli(key=preds_key, p=0.7, shape=(1000,)).astype(
            jnp.float32
        )
        batch_labels = jax.random.bernoulli(
            key=labels_key, p=0.7, shape=(1000,)
        ).astype(jnp.float32)
        batch_metrics = solstice.ClassificationMetrics(
            batch_preds, batch_labels, jnp.nan, 2
        )
        metrics = batch_metrics.merge(metrics) if metrics else batch_metrics
        total_preds = (
            batch_preds
            if total_preds is None
            else jnp.concatenate([total_preds, batch_preds])
        )
        total_labels = (
            batch_labels
            if total_labels is None
            else jnp.concatenate([total_labels, batch_labels])
        )
    assert metrics is not None
    metrics_dict = metrics.compute()
    skmetrics_dict = skmetrics.classification_report(
        total_labels, total_preds, output_dict=True
    )

    # check metrics to 4 decimal places
    process_metrics = lambda x: jax.device_get(jnp.round(x, 4))

    metrics_dict = jax.tree_util.tree_map(process_metrics, metrics_dict)
    skmetrics_dict = jax.tree_util.tree_map(process_metrics, skmetrics_dict)

    # only bother checking the averaged ones, no point checking all the individual ones
    assert metrics_dict["accuracy"] == skmetrics_dict["accuracy"]
    assert metrics_dict["f1_macro"] == skmetrics_dict["macro avg"]["f1-score"]
    assert metrics_dict["tpr_macro"] == skmetrics_dict["macro avg"]["recall"]
    assert metrics_dict["ppv_macro"] == skmetrics_dict["macro avg"]["precision"]
    assert metrics_dict["f1_weighted"] == skmetrics_dict["weighted avg"]["f1-score"]
    assert metrics_dict["tpr_weighted"] == skmetrics_dict["weighted avg"]["recall"]
    assert metrics_dict["ppv_weighted"] == skmetrics_dict["weighted avg"]["precision"]


test_classification_metrics()
