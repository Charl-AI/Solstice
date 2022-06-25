# Metrics

Our Metrics API is similar to the one in [CLU](https://github.com/google/CommonLoopUtils), although more sexy because we use equinox :) We favour defining one single object for handling all metrics for an experiment instead of composing multiple objects into a collection. This is more efficient because often we can calculate a battery of metrics from the same intermediate results. It is also simpler and easier to reason about.

::: solstice.Metrics

---

::: solstice.ClassificationMetrics
