# Config Management

Solstice does not include config management functions, nor does it offer an opinion on how to do it. However, we believe that it is an important area of experiment design, so we offer a quick (opinionated) guide for managing configs with Solstice and Hydra.

## The problem

Ideally, all experiments would be fully configurable through command line or config files, without changing much (if any) code. This can be tricky to acheieve - you know you have failed when your `main.py` or `train.py` file looks something like this:

```python

def main(config):
    if config.dataset_name == "mnist":
        train_ds, val_ds, test_ds = get_mnist_data(config.dataset_batch_size)

        if config.model_name == "resnet":
            raise ValueError(f"resnet implementation does not work with mnist")
        elif config.model_name == "mlp":
            model = get_mlp(28*28, config.mlp_model_depth, config.mlp_model_width)

    elif config.dataset_name == "cifar":
        train_ds, val_ds, test_ds = get_cifar_data(config.dataset_batch_size, config.dataset_split)

        if config.model_name == "resnet":
            model = get_resnet(32*32, config.resnet_type)
        elif config.model_name == "mlp":
            model = get_mlp(32*32, config.mlp_model_depth, config.mlp_model_width)

    else:
        raise ValueError(f"Dataset not supported {config.dataset_name=}")

    optimizer = optax.adam(config.lr)

    ...


```

Yuck! You can see that there are loads of components that would be fine on their own, but end up getting glued together by some horrible convoluted logic in the main file. Each time you want to add functionality you will have to add a bunch more if statements - the whole thing is a combinatoric nightmare! You also have the problem that different config options want to be configured differently, for example the resnet needs to know different things to the mlp. This means you end up passing everything into your program and ignoring half of it, which is incredible error prone. We show how Solstice+Hydra fixes this.
