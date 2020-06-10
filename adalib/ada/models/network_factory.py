import torch
import ada.models.modules as amm


def get_module_fn(module_name):
    if module_name == "feedforward":
        return amm.FeatureExtractFF
    elif module_name == "digits":
        return amm.FeatureExtractorDigits
    elif module_name == "AlexNet":
        return amm.ResNet50AlexNetFeatureFeature
    elif module_name == "ResNet18":
        return amm.ResNet18Feature
    elif module_name == "ResNet34":
        return amm.ResNet34Feature
    elif module_name == "ResNet50":
        return amm.ResNet50Feature
    elif module_name == "ResNet101":
        return amm.ResNet101Feature
    elif module_name == "ResNet152":
        return amm.ResNet152Feature
    else:
        raise NotImplementedError(
            "Unknown module {module_name}. Please define the function in modules.py and register its use here."
        )


class NetworkFactory:
    """
    This class takes a network configuration dictionary and
    creates the corresponding modules. 
    """

    def __init__(self, network_config):
        self._params = network_config

    def get_feature_extractor(self, input_dim, *args):
        cfg = self._params["feature"]
        net = get_module_fn(cfg["name"])
        if cfg["name"] == "feedforward":
            return net(
                input_dim,
                cfg["hidden_size"],
                activation_fn=torch.nn.LeakyReLU,
                inplace=True,
            )
        else:
            return net(*args)

    def get_task_classifier(self, input_dim, n_classes):
        cfg = self._params["task"]
        if cfg["name"] == "feedforward":
            return amm.FFSoftmaxClassifier(
                input_dim,
                n_classes=n_classes,
                name="h",
                hidden=cfg.get("hidden_size", ()),
            )
        if cfg["name"] == "digits":
            return amm.DataClassifierDigits(input_size=input_dim, n_class=n_classes)
        raise NotImplementedError()

    def get_critic_network(self, input_dim):
        cfg = self._params["critic"]
        if cfg["name"] == "feedforward":
            return amm.FFSoftmaxClassifier(
                input_dim, n_classes=2, name="g", hidden=cfg.get("hidden_size", ())
            )
        elif cfg["name"] == "digits":
            return amm.DomainClassifierDigits(input_size=input_dim)
        raise NotImplementedError()
