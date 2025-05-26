import os
from pathlib import Path
from typing import Union, Callable, Optional
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL.ImageOps import exif_transpose
import random

from ldm.data.utils import (
    ToRGB,
    ColorBackgroundRGBA,
    CropForegroundRGBA,
    RandomResizeForeground,
)


class GenDataset(Dataset):
    def __init__(
        self,
        data_root: Union[str, os.PathLike],
        split: str = "train",
        splits = (0.999, 0.001, 0.000),
        size: int = 512,
        crop_res: int = 256,
        resolution: int = 512,
        empty_prompt_p=0.07,
        jitter_p=0.1,
        erase_p=0.4,
        min_pct_area=0.7,
    ):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data {self.data_root} images root doesn't exist.")

        objects = os.listdir(self.data_root)
        self.objects = objects

        # Split data into train, val, test
        total_samples = len(self.objects)
        self.total_samples = total_samples
        train_end = int(splits[0] * total_samples)
        val_end = train_end + int(splits[1] * total_samples)

        if split == "train":
            self.objects = self.objects[:train_end]
        elif split == "val":
            self.objects = self.objects[train_end:val_end]

        self.num_images = len(self.objects)

        self.empty_prompt_p = empty_prompt_p
        self.resolution = resolution

        self.jitter_transform = transforms.RandomApply(
            torch.nn.ModuleList([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)]),
            p=jitter_p,
        )
        self.erase = transforms.RandomErasing(p=erase_p, scale=(0.02, 0.25), value=1)

        self.image_transform = transforms.Resize(resolution)
        self.ref_image_transform = transforms.Compose(
            [
                ColorBackgroundRGBA(),
                CropForegroundRGBA(),
                ToRGB(),
                RandomResizeForeground(resolution, min_pct_area=min_pct_area),
                transforms.RandomHorizontalFlip(),
            ]
        )

        self.image_norm_transform =  transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        self.refimage_norm_transform = self.image_norm_transform

    @property
    def _length(self):
        return self.num_images

    def __len__(self):
        return self._length

    def _prepare_image(self, image_input, transform):
        if isinstance(image_input, Image.Image):
            return transform(image_input)
        elif isinstance(image_input, np.ndarray):
            return transform(Image.fromarray(image_input, mode="RGBA"))
        elif isinstance(image_input, (str, Path)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                image = Image.open(image_input)
                image = exif_transpose(image)
                if np.std(np.array(image)) == 0:
                    raise ValueError("Image has zero standard deviation.")
            return transform(image)
        else:
            raise TypeError("Unsupported image input type.")

    def __getitem__(self, index):
        try:
            example = {}
            object_name = self.objects[index]
            object_dir = Path(self.data_root, object_name)
            images = [
                img
                for img in list(object_dir.iterdir())
                if "_generation.png" in str(img)
            ]

            if not images:
                raise ValueError(f"No images found in {object_dir}")

            target_image_path = random.choice(images)
            prompt_fname = target_image_path.with_name(
                target_image_path.name.replace("_generation.png", "_caption.txt")
            )
            with open(prompt_fname, "r") as f:
                prompt = f.read().strip()

            refprompt = prompt.split(",")[0]

            if np.random.rand() < self.empty_prompt_p:
                prompt = ""

            example["caption"] = prompt
            example["refprompt"] = refprompt

            # Process instance image (current image)
            original_instance_image = self._prepare_image(
                target_image_path, self.image_transform
            )
            if self.image_norm_transform is not None:
                instance_image = self.image_norm_transform(original_instance_image)
            else:
                instance_image = original_instance_image

            example["image"] = instance_image  # Mapping to 'image'

           
            ref_image_path = target_image_path  

            # Load and process reference image
            ref_im = np.array(Image.open(ref_image_path))
            mask_path = ref_image_path.with_name(
                ref_image_path.name.replace("generation.png", "mask.png")
            )
            mask = np.array(Image.open(mask_path))
            ref_im_rgba = np.concatenate((ref_im, mask[:, :, np.newaxis]), axis=-1)

            original_reference_image = self._prepare_image(
                ref_im_rgba, self.ref_image_transform
            )
            if self.refimage_norm_transform is not None:
                reference_image = self.refimage_norm_transform(original_reference_image)
                # Apply jitter and erase
                reference_image = self.erase(
                    self.jitter_transform(reference_image)
                )
            else:
                reference_image = original_reference_image

            example["image_ref"] = reference_image 

            # Additional fields to match previous dataset
            example["label"] = object_name
            example["text_init"] = ""
            example["placeholder_pos"] = [0, 1]

            

        except Exception as e:
            print("Data Error:", e)
            return self.__getitem__(np.random.randint(self._length))
        return example



