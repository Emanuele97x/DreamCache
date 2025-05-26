from typing import Union, Optional, Tuple
import numpy as np
import torch
from PIL import Image


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class ToRGB(object):
    """Converts PIL Image to RGB"""

    def __call__(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError(
                "Image should be a Pillow Image. Got {}.".format(type(image))
            )

        if not image.mode == "RBG":
            image = image.convert("RGB")
        return image


class ColorBackgroundRGBA(object):
    """Color alpha = 0 pixels to user specified color in RGBA image."""

    def __init__(
        self,
        color: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (255, 255, 255),
    ):
        if len(color) not in [3, 4]:
            raise ValueError(
                "Given color must have at least 3 values (RGB) or 4 values (RGBA). Got {}".format(
                    len(color)
                )
            )
        if not all([0 <= c < 256 for c in color]):
            raise ValueError(
                "Not all color values are 0 <= color < 256. Got {}".format(color)
            )

        self.color = np.array(color, dtype=np.uint8)

    def __call__(
        self, image: Union[np.ndarray, Image.Image]
    ) -> Union[np.ndarray, Image.Image]:
        if not (isinstance(image, Image.Image) or isinstance(image, np.ndarray)):
            raise TypeError(
                "Image should be Pillow Image or NumPy array. Got {}.".format(
                    type(image)
                )
            )

        image_np = image
        if isinstance(image, Image.Image):
            image_np = np.array(image)

        if image_np.ndim != 3:
            raise ValueError(
                "Expected 3 dimensional input. Got {}".format(image_np.ndim)
            )
        if image_np.shape[-1] == 3:
            return Image.fromarray(image_np)
        if image_np.shape[-1] != 4:
            raise ValueError(
                "Input image does not have 4 channels (RGBA). Got {}".format(
                    image_np.shape
                )
            )

        # Extract the alpha channel
        alpha_channel = image_np[:, :, 3]

        # Create a binary mask from the alpha channel for all transparent pixels
        background_mask = (alpha_channel == 0).astype(np.uint8)

        # Set masked pixels to specified color
        if self.color.size == 4:
            image_np[background_mask] = self.color
        else:
            # Alpha is unchanged
            # Apply the mask to the RGB channels
            image_np_rgb = (
                image_np[..., :3] * (1 - background_mask[..., np.newaxis])
                + self.color * background_mask[..., np.newaxis]
            )

            # Combine the new RGB channels with the original Alpha channel
            image_np = np.concatenate((image_np_rgb, image_np[..., 3:]), axis=-1)

        if isinstance(image, Image.Image):
            return Image.fromarray(image_np)
        return image


class CropForegroundRGBA(object):
    """Crop foreground from RGBA image using alpha channel (alpha > 0).

    Returns foreground cropped image
    """

    def __call__(
        self, image: Union[np.ndarray, Image.Image]
    ) -> Union[np.ndarray, Image.Image]:
        if not (isinstance(image, Image.Image) or isinstance(image, np.ndarray)):
            raise TypeError(
                "Image should be Pillow Image or NumPy array. Got {}.".format(
                    type(image)
                )
            )

        image_np = image
        if isinstance(image, Image.Image):
            image_np = np.array(image)

        # Extract the alpha channel
        alpha_channel = image_np[:, :, 3]

        # Create a binary mask from the alpha channel
        foreground_mask = alpha_channel > 0

        # Find the bounding box of the foreground
        rows = np.any(foreground_mask, axis=1)
        cols = np.any(foreground_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the foreground based on the bounding box
        foreground_cropped = image_np[rmin : rmax + 1, cmin : cmax + 1]
        if isinstance(image, Image.Image):
            return Image.fromarray(foreground_cropped)
        return foreground_cropped


class RandomResizeForeground(object):
    """Randomly resizes foreground image into background (optional)
    If background is not given, will resize foreground onto white background
    Note: Will resize background if background is given and shape != self.size

    Args:
        size (sequence or int): Output size. If size is integer, returns square image
        min_pct_area (float, optional) : Minimum pct of area for resized foreground in output image. Defaults to 0.25
    """

    def __init__(
        self,
        size: Union[Tuple[int, int], int],
        min_pct_area: float = 0.4,
    ):
        if isinstance(size, int):
            size = (size, size)
        if not all([0 < s for s in size]):
            raise ValueError("Expected positive values for size. Got {}".format(size))
        if min_pct_area < 0 or min_pct_area > 1:
            raise ValueError(
                "Expected 0 <= min_pct_area <= 1. Got {}".format(min_pct_area)
            )

        self.size = size
        self.W, self.H = self.size
        self.min_pct_area = min_pct_area

    def __call__(
        self,
        foreground: Union[np.ndarray, Image.Image],
        background: Optional[Union[np.ndarray, Image.Image]] = None,
    ) -> Union[np.ndarray, Image.Image]:
        """
        Returns:
            NumPy array if: background is given as NumPy array or background is None and foreground is NumPy array
            O.w. returns PIL.Image

        """
        pil_foreground = foreground
        if isinstance(pil_foreground, np.ndarray):
            pil_foreground = Image.fromarray(foreground)

        pil_background = background
        if pil_background is None:
            pil_background = Image.new(pil_foreground.mode, self.size, (255, 255, 255))
        if isinstance(pil_background, np.ndarray):
            pil_background = Image.fromarray(background)

        if pil_background.size != self.size:
            pil_background = pil_background.resize(self.size)

        w, h = pil_foreground.width, pil_foreground.height
        aspect_ratio = w / h

        # Calculate the minimum allowable size for the resized foreground
        min_foreground_width = int(self.min_pct_area * self.W)
        min_foreground_height = int(self.min_pct_area * self.H)

        # Randomly sample the desired width and height for the resized foreground
        desired_width = np.random.randint(min_foreground_width, self.W + 1)
        desired_height = np.random.randint(min_foreground_height, self.H + 1)

        # Calculate the height based on the desired width and the aspect ratio
        desired_height = int(desired_width / aspect_ratio)

        # Ensure that the new dimensions are within the allowable range
        if desired_height > self.H:
            desired_height = self.H
            desired_width = int(desired_height * aspect_ratio)
        if desired_width > self.W:
            desired_width = self.W
            desired_height = int(desired_width / aspect_ratio)

        # Resize the cropped foreground to (desired_height, desired_width)
        desired_size = (desired_width, desired_height)
        resized_foreground = pil_foreground.resize(desired_size)

        # Generate random coordinates to place the resized foreground within the input image
        max_x = self.W - desired_width
        max_y = self.H - desired_height
        random_x = np.random.randint(0, max_x + 1)
        random_y = np.random.randint(0, max_y + 1)

        # Paste the resized foreground onto the canvas at the random coordinates
        pil_background.paste(resized_foreground, (random_x, random_y))
        if isinstance(background, np.ndarray) or (
            background is None and isinstance(foreground, np.ndarray)
        ):
            return np.array(pil_background)
        return pil_background


class BackgroundRemoval:
    def __init__(self, device="cuda"):
        from carvekit.api.high import HiInterface

        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        image = self.interface([image])[0]
        return image