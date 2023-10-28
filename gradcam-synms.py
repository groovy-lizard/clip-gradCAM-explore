import numpy as np
import torch
import torch.nn.functional as F
# import cv2
import clip
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from torch import nn


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad

# Reference: https://arxiv.org/abs/1610.02391


def gradCAM(
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor,
        layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel dimension
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence
        # so we clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(gradcam,
                            input.shape[2:],
                            mode='bicubic',
                            align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    return x / 0.6

# Modified from:
# https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb


def getAttMap(img, attn_map, blur=True):
    img = img[:, :, :1]
    if blur:
        attn_map = gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('plasma')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
        (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map


def viz_attn(img, attn_map, title, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.title(title)
    plt.show()


def save_attn(img, attn_map, title, fname, blur=True):
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(getAttMap(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')


def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


def main():
    print('\nLoading model...')
    available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16']
    layers = ['layer4', 'layer3', 'layer2', 'layer1']

    clip_model = available_models[0]
    saliency_layer = layers[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=False)

    print('\nStarting main loop')
    while True:
        fname = input("Enter filename: ")
        label = input('Enter target label: ')
        image_input = preprocess(Image.open(fname)).unsqueeze(0).to(device)
        image_np = load_image(fname, model.visual.input_resolution)
        text_input = clip.tokenize([f'a photo of a {label}']).to(device)
        attn_map = gradCAM(
            model.visual,
            image_input,
            model.encode_text(text_input).float(),
            getattr(model.visual, saliency_layer)
        )

        save_attn(image_np, attn_map, label, fname, blur=True)
        loop = input("Run it again? [y/N]: ")
        if len(loop) == 0:
            break
        elif loop == 'n' or loop == 'N':
            break


if __name__ == "__main__":
    main()
