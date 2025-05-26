import os
import random
import json
from typing import Any, Tuple
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class EditDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        splits: Tuple[float, float, float] = (0.99, 0.001, 0.009),
        size: int = 512,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        drop_text_prob: float = 0.07, 
        
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = data_root
        self.flip_prob = flip_prob
        self.interpolation = self.get_interpolation("bilinear")
        self.size = size
        self.text_prob = drop_text_prob
       

        # Collect all image and caption file pairs
        self.image_files = [f for f in os.listdir(data_root) if f.endswith('.jpg')]
        self.caption_files = [f.replace('.jpg', '.txt') for f in self.image_files]

        # Split data into train, val, test
        total_samples = len(self.image_files)
        self.total_samples = total_samples
        train_end = int(splits[0] * total_samples)
        val_end = train_end + int(splits[1] * total_samples)

        if split == "train":
            self.image_files = self.image_files[:train_end]
            self.caption_files = self.caption_files[:train_end]
        elif split == "val":
            self.image_files = self.image_files[train_end:val_end]
            self.caption_files = self.caption_files[train_end:val_end]
        else:
            self.image_files = self.image_files[val_end:]
            self.caption_files = self.caption_files[val_end:]

        self.flip = transforms.RandomHorizontalFlip(p=flip_prob)

    def __len__(self):
        return len(self.image_files)
    
    def get_interpolation(self, method: str):
        return {
            "linear": Image.BILINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[method]

    def __getitem__(self, i):
        example = {}
        while True:
            try:
                image_path = os.path.join(self.path, self.image_files[i % self.total_samples])
                caption_path = os.path.join(self.path, self.caption_files[i % self.total_samples])
                
                with open(caption_path, 'r') as fp:
                    prompt_0 = fp.read().strip()
                
                image_0 = Image.open(image_path)

                
                if image_0.mode != "RGB":
                    image_0 = image_0.convert("RGB")
                
                example["image"] = self.process_image(image_0)
                example["label"] = "ciao"

                break
            except:
                print(self.caption_files[i % self.total_samples])                
                i = i+1

        example["text_init"] = ""
        example["caption"] = prompt_0
        if random.random() <= self.text_prob:
            example["caption"] = ""
          
        # if random.random() <= 0.1:
        #     example["text_init"]= ""

        example["placeholder_pos"] = [0, 1]
      
        return example
    
    def process_image(self, image: Image.Image):
        img = np.array(image).astype(np.uint8)
        
        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        return (np.array(image).astype(np.uint8) / 127.5 - 1.0).astype(np.float32)
