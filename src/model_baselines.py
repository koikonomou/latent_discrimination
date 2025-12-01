# models_zoo.py
import torch
import torch.nn as nn
import torchvision as tv


# -------------------------
# Small ConvNet (Conv / Conv-NN)
# -------------------------

class ConvNet(nn.Module):
    """
    Small ConvNet similar to DD/DC baselines:
    Conv -> (Norm) -> ReLU -> AvgPool blocks.
    norm_type: 'in' (InstanceNorm2d), 'bn' (BatchNorm2d), 'none'
    """
    def __init__(self, num_classes=10, norm_type='in'):
        super().__init__()

        def block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)]
            if norm_type == 'in':
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            elif norm_type == 'bn':
                layers.append(nn.BatchNorm2d(out_ch))
            # 'none' -> no norm
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.AvgPool2d(kernel_size=2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# -------------------------
# Normalization helpers
# -------------------------

def _insert_in_after_conv(module: nn.Module):
    """
    For AlexNet.features: insert InstanceNorm2d after each Conv2d.
    """
    new_layers = []
    for layer in module:
        new_layers.append(layer)
        if isinstance(layer, nn.Conv2d):
            new_layers.append(nn.InstanceNorm2d(layer.out_channels, affine=True))
    return nn.Sequential(*new_layers)


def _replace_bn_with_in(module: nn.Module):
    """
    Recursively replace BatchNorm2d with InstanceNorm2d(affine=True, no running stats).
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            new = nn.InstanceNorm2d(child.num_features, affine=True, track_running_stats=False)
            setattr(module, name, new)
        else:
            _replace_bn_with_in(child)


def _strip_norm_layers(module: nn.Module):
    """
    Replace BatchNorm2d / InstanceNorm2d with Identity
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            setattr(module, name, nn.Identity())
        else:
            _strip_norm_layers(child)


# -------------------------
# ResNet helper
# -------------------------

def _make_resnet(depth: int, norm: str, num_classes: int, pretrained: bool):
    if depth == 18:
        WeightsEnum = tv.models.ResNet18_Weights
        weights = WeightsEnum.IMAGENET1K_V1 if pretrained else None
        base = tv.models.resnet18
    elif depth == 34:
        WeightsEnum = tv.models.ResNet34_Weights
        weights = WeightsEnum.IMAGENET1K_V1 if pretrained else None
        base = tv.models.resnet34
    elif depth == 50:
        WeightsEnum = tv.models.ResNet50_Weights
        weights = WeightsEnum.IMAGENET1K_V2 if pretrained else None
        base = tv.models.resnet50
    elif depth == 101:
        WeightsEnum = tv.models.ResNet101_Weights
        weights = WeightsEnum.IMAGENET1K_V2 if pretrained else None
        base = tv.models.resnet101
    elif depth == 152:
        WeightsEnum = tv.models.ResNet152_Weights
        weights = WeightsEnum.IMAGENET1K_V2 if pretrained else None
        base = tv.models.resnet152
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    model = base(weights=weights)

    if norm == "in":
        _replace_bn_with_in(model)
    elif norm == "nn":
        _strip_norm_layers(model)
    # norm == "bn": keep default BN

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# -------------------------
# Main factory
# -------------------------

def create_model(arch: str, num_classes: int = 10, pretrained: bool = False):
    """
    Supported arch (case-insensitive):
      Conv, Conv-NN,
      AlexNet-NN, AlexNet-IN,
      VGG11-IN, VGG11-BN,
      ResNet18-BN, ResNet18-IN,
      ResNet34-IN, ResNet50-IN, ResNet101-IN, ResNet152-IN
    """
    key = arch.lower()

    # ---- Conv / Conv-NN ----
    if key == "conv":
        model = ConvNet(num_classes=num_classes, norm_type='in')
        return model, "conv"

    if key == "conv-nn":
        model = ConvNet(num_classes=num_classes, norm_type='none')
        return model, "conv-nn"

    # ---- AlexNet ----
    if key == "alexnet-nn":
        Weights = tv.models.AlexNet_Weights
        weights = Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.models.alexnet(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, "alexnet-nn"

    if key == "alexnet-in":
        Weights = tv.models.AlexNet_Weights
        weights = Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.models.alexnet(weights=weights)
        model.features = _insert_in_after_conv(model.features)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, "alexnet-in"

    # ---- VGG11 ----
    if key == "vgg11-bn":
        Weights = tv.models.VGG11_BN_Weights
        weights = Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.models.vgg11_bn(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, "vgg11-bn"

    if key == "vgg11-in":
        Weights = tv.models.VGG11_BN_Weights
        weights = Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.models.vgg11_bn(weights=weights)
        _replace_bn_with_in(model.features)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, "vgg11-in"

    # ---- Densenet169 ----
    if key == "densenet169":
        Weights = tv.models.DenseNet169_Weights
        weights = Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.models.densenet169(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model, "densenet169"

    # ---- ResNets ----
    if key == "resnet18-bn":
        model = _make_resnet(18, norm="bn", num_classes=num_classes, pretrained=pretrained)
        return model, "resnet18-bn"

    if key == "resnet18-in":
        model = _make_resnet(18, norm="in", num_classes=num_classes, pretrained=pretrained)
        return model, "resnet18-in"

    if key == "resnet34-in":
        model = _make_resnet(34, norm="in", num_classes=num_classes, pretrained=pretrained)
        return model, "resnet34-in"

    if key == "resnet50-in":
        model = _make_resnet(50, norm="in", num_classes=num_classes, pretrained=pretrained)
        return model, "resnet50-in"

    if key == "resnet101-in":
        model = _make_resnet(101, norm="in", num_classes=num_classes, pretrained=pretrained)
        return model, "resnet101-in"

    if key == "resnet152-in":
        model = _make_resnet(152, norm="in", num_classes=num_classes, pretrained=pretrained)
        return model, "resnet152-in"

    raise ValueError(f"Unsupported arch: {arch}")
