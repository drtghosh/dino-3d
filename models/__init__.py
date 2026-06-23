from models.dino import DINOPointEncoder


def get_model(config, name):
    if name == "dinov1":
        return DINOPointEncoder(config)
    elif name == "dinov2":
        return DINOPointEncoder(config)
    else:
        raise NotImplementedError("Got name '{}'".format(name))


def set_requires_grad(models, requires_grad=False):
    """
        Parameters:
            models (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(models, list):
        models = [models]
    for model in models:
        if models is not None:
            for param in model.parameters():
                param.requires_grad = requires_grad
                