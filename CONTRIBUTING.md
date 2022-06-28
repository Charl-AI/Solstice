# Contributing

> This page looks nicer in the [docs](https://charl-ai.github.io/Solstice/contributing/)

All contributions are welcome, including questions, issues, feature suggestions, and pull requests. There are two main parts to this codebase: abstract interfaces (such as `solstice.Experiment` and `solstice.Metrics`) and concrete implementations (such as `solstice.ClassificationExperiment` and `solstice.ClassificationMetrics`). Contributing concrete implementations is a good idea because it directly adds functionality for users without changing APIs. Changing the abstractions will require some discussion.

My research focusses on computer vision, so this codebase will likely be biased towards things useful for that. Contributions helpful to other fields, such as NLP, are encouraged :)

!!! info
    The supported method for installing the Solstice development environment is [VSCode devcontainers](). All you need is to have Docker and the VSCode 'Remote - Containers' extension installed. You can then install this project by running `Remote containers: Clone Repository in Container Volume` from the command palette (alternatively, you could clone the repository normally and run `Remote Containers: Open folder in Container`.

    The default container is CPU only, but you can use GPU (CUDA 11.3) by following the instructions in `devcontainer.json` and rebuilding the container.

When making PRs, please ensure any changes are tested and documented. You can run tests by running `pytest`, and you can check the documentation locally by running `mkdocs serve`.

## Style Guide

To match the style of the rest of the project, please follow these basic guidelines when making PRs. In general, if you follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), you should be fine.

### Code

!!! tip
    We use [Poetry](https://python-poetry.org/) for dependency management. You can add development dependencies with `poetry add -D <package>`.

- Black formatted (this will be automatic if using the devcontainer).
- Use type annotations for all parameters and return types.
- Try to avoid adding dependencies. If you need to use an import for type annotations, use the `TYPE_CHECKING` flag so that the dependency is only needed as a development one.

!!! example
    Here, tensorflow is not imported at runtime, so it does not need to be a solstice dependency. It should be included as a development dependency for type checking.
    ```python
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        import tensorflow as tf

    def data_pipeline(ds: tf.data.dataset) -> tf.data.dataset:
        return ds.batch(32).prefetch(1)

    ```



### Docs

!!! tip
    Use boxes like this one to make docs look nicer. These are supported by the markdown [Admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#usage) extension.

- We use mkdocstrings to allow documentation to be generated from docstrings. Favour documenting the API this way, as opposed to 'hard-coding' the documentation in the docs directory.
- Document all public API with Google-style docstrings (the devcontainer comes with the 'autodocstring' extension to make this easier).
- All methods in an abstract class (e.g. `solstice.Experiment`) should be documented, including examples of usage and implementation tips.
- The methods for concrete implementation classes (e.g. `solstice.ClassificationExperiment`) often don't need docstrings, as the abstract class already documents the interface.

!!! example


### Tests
