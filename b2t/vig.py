from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import transforms as T

def transform_celeba() -> T.Compose:
    return T.Compose(
        [
            T.CenterCrop(178),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def transform_waterbirds() -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def dataset_transform(dataset_name: str) -> Tuple[T.Compose, int]:
    if dataset_name == "waterbirds":
        transform = transform_waterbirds()
        num_classes = 2
    elif dataset_name == "celeba":
        transform = transform_celeba()
        num_classes = 2
    else:
        raise ValueError()
    return transform, num_classes


def configure_device(device_id: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        if device_id is not None and isinstance(device_id, int):
            device = f"cuda:{device_id}"
        else:
            device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


class ViG:
    def __init__(
        self,
        model: Union[str, nn.Module],
        device_or_id: Union[torch.device, int] = None,
        dataset_name: str = "waterbirds",
        mask_threshold: float = 0.5,
    ):
        if device_or_id is None or not isinstance(device_or_id, torch.device):
            self.device = configure_device(device_id=device_or_id)
        else:
            self.device = device_or_id
        self.preprocess_fnc, num_classes = dataset_transform(dataset_name)
        model, target_layers = self.configure_model(
            model=model, num_classes=num_classes, device=self.device
        )

        self.cam = GradCAM(model=model, target_layers=target_layers)
        self.targets = [ClassifierOutputTarget(0)]
        self.mask_threshold = mask_threshold

    def configure_model(
        self,
        model: Union[str, nn.Module],
        num_classes: int,
        device: torch.device,
    ) -> Tuple[nn.Module, List[nn.Module]]:
        if isinstance(model, str):
            model = torch.load(model, map_location=device)
        else:
            raise ValueError()
        model.to(device=device)
        model.eval()
        if hasattr(model, "layer4"):
            target_layers = [model.layer4[-1]]
        else:
            target_layers = [model.featurizer.layer4[-1]]
        return model, target_layers

    def __call__(
        self,
        image,
    ) -> torch.Tensor:
        tr_image = self.preprocess_fnc(image).unsqueeze(0)
        grayscale_cam = self.cam(input_tensor=tr_image, targets=self.targets)
        mask = np.where(
            grayscale_cam
            < np.quantile(grayscale_cam[0, :], self.mask_threshold),
            0.0,
            1.0,
        )
        return (image * mask).to(device=self.device)
