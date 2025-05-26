import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from transformers import CLIPTokenizer

import random

imagenet_templates_smallest = [
    'a close-up photo of a {} {}  {} {} ',
    'a nice photo of a {} {}  {} {} ',
    'a photo of the {} {}  {} {} ',
    'a close-up photo of the {} {}  {} {} ',
    'the photo of a {} {}  {} {} ',
    'the close-up photo of a {} {}  {} {} ',
    'a picture of {} {}  {} {} ',
    'a good photo of a {} {}  {} {} ',
    'a good picture of the {} {}  {} {} ', 
    'a photo of my {} {}  {} {} ', 
    'a close-up photo of the {} {}  {} {} ',
    'a bright photo of the {} {}  {} {} ', 
    'a depiction of a {} {}  {} {} ',
    'a close-up {} {}  {} {} ',
    'a cropped photo of {} {}  {} {} ',
    'a cropped picture of {} {}  {} {} '


]

imagenet_templates_mem = [
    'a photo of a {} {}'
]
   

imagenet_templates_small = [
    'a photo of a {} {} ',
    #'a rendering of a {} {} ',
    'a cropped photo of the {} {} ',
    'the photo of a {} {} ',
    'a photo of a clean {} {} ',
    #'a photo of a dirty {} {} ',
    'a dark photo of the {} {} ',
    'a photo of my {} {} ',
    'a photo of the cool {} {} ',
    'a close-up photo of a {} {} ',
    'a bright photo of the {} {} ',
    'a cropped photo of a {} {} ',
    'a photo of the {} {} ',
    'a good photo of the {} {} ',
    'a photo of one {} {} ',
    'a close-up photo of the {} {} ',
    'a rendition of the {} {} ',
    'a photo of the clean {} {} ',
    'a rendition of a {} {} ',
    'a photo of a nice {} {} ',
    'a good photo of a {} {} ',
    'a photo of the nice {} {} ',
    'a photo of the small {} {} ',
   ## 'a photo of the weird {} {} ',
   # 'a photo of the large {} {} ',
   # 'a photo of a cool {} {} ',
    'a photo of a small {} {} ',
    'an illustration of a {} {} ',
    #'a rendering of a {} {} ',
    'a cropped photo of the {} {} ',
    'the photo of a {} {} ',
    #'an illustration of a clean {} {} ',
    #'an illustration of a dirty {} {} ',
    'a dark photo of the {} {} ',
    #'an illustration of my {} {} ',
    #'an illustration of the cool {} {} ',
    'a close-up photo of a {} {} ',
    'a bright photo of the {} {} ',
    'a cropped photo of a {} {} ',
    #'an illustration of the {} {} ',
    'a good photo of the {} {} ',
    #'an illustration of one {} {} ',
    'a close-up photo of the {} {} ',
    'a rendition of the {} {} ',
    #'an illustration of the clean {} {} ',
    #'a rendition of a {} {} ',
    'an illustration of a nice {} {} ',
    'a good photo of a {} {} ',
    'an illustration of the nice {} {} ',
    'an illustration of the small {} {} ',
    #'an illustration of the weird {} {} ',
    #'an illustration of the large {} {} ',
    #'an illustration of a cool {} {} ',
    #'an illustration of a small {} {} ',
    'a depiction of a {} {} ',
    #'a rendering of a {} {} ',
    'a cropped photo of the {} {} ',
    'the photo of a {} {} ',
    'a depiction of a clean {} {} ',
    'a depiction of a dirty {} {} ',
    'a dark photo of the {} {} ',
    'a depiction of my {} {} ',
    'a depiction of the cool {} {} ',
    'a close-up photo of a {} {} ',
    'a bright photo of the {} {} ',
    'a cropped photo of a {} {} ',
    'a depiction of the {} {} ',
    'a good photo of the {} {} ',
    'a depiction of one {} {} ',
    'a close-up photo of the {} {} ',
    'a rendition of the {} {} ',
    'a depiction of the clean {} {} ',
    'a rendition of a {} {} ',
    'a depiction of a nice {} {} ',
    'a good photo of a {} {} ',
    'a depiction of the nice {} {} ',
    #'a depiction of the small {} {} ',
    #'a depiction of the weird {} {} ',
    #'a depiction of the large {} {} ',
    #'a depiction of a cool {} {} ',
    #'a depiction of a small {} {} ',
]

imagenet_dual_templates_small = [
    'a photo of a {} {}  with {} {} ',
    'a rendering of a {} {}  with {} {} ',
    'a cropped photo of the {} {}  with {} {} ',
    'the photo of a {} {}  with {} {} ',
    'a photo of a clean {} {}  with {} {} ',
    'a photo of a dirty {} {}  with {} {} ',
    'a dark photo of the {} {}  with {} {} ',
    'a photo of my {} {}  with {} {} ',
    'a photo of the cool {} {}  with {} {} ',
    'a close-up photo of a {} {}  with {} {} ',
    'a bright photo of the {} {}  with {} {} ',
    'a cropped photo of a {} {}  with {} {} ',
    'a photo of the {} {}  with {} {} ',
    'a good photo of the {} {}  with {} {} ',
    'a photo of one {} {}  with {} {} ',
    'a close-up photo of the {} {}  with {} {} ',
    'a rendition of the {} {}  with {} {} ',
    'a photo of the clean {} {}  with {} {} ',
    'a rendition of a {} {}  with {} {} ',
    'a photo of a nice {} {}  with {} {} ',
    'a good photo of a {} {}  with {} {} ',
    'a photo of the nice {} {}  with {} {} ',
    'a photo of the small {} {}  with {} {} ',
    'a photo of the weird {} {}  with {} {} ',
    'a photo of the large {} {}  with {} {} ',
    'a photo of a cool {} {}  with {} {} ',
    'a photo of a small {} {}  with {} {} ',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

placeholder_tokens = {
    'backpack': '@',
    'backpack_dog': '#',
    'bear_plushie': '$',
    'berry_bowl': '%',
    'can': '^',
    'candle': '&',
    'cat': '*',
    'cat2': '!',
    'clock': '(',
    'colorful_sneaker': '=',
    'dog': '-',
    'dog2': '_',
    'dog3': '~',
    'dog5': '`',
    'dog6': '|',
    'dog7': '/',
    'dog8': ';', 
    'duck_toy':')',
    'fancy_boot': '<',
    'grey_sloth_plushie': '>',
    'monster_toy': '{',
    'pink_sunglasses': '}',
    'poop_emoji': '[',
    'rc_car': ']',
    'red_cartoon': '"',
    'robot_toy': ':',
    'shiny_sneaker': '+', 
    'teapot': ',',
    'vase': '©',
    'wolf_plushie': '¢'
}

class_attributes = {
    'backpack': 'backpack',
    'backpack_dog': 'backpack',
    'bear_plushie': 'bear plushie',
    'berry_bowl': 'berry bowl',
    'can': 'can',
    'candle': 'candle',
    'cat': 'cat',
    'cat2': 'cat',
    'clock': 'clock',
    'colorful_sneaker': 'sneaker',
    'dog': 'dog',
    'dog2': 'dog',
    'dog3': 'dog',
    'dog5': 'dog',
    'dog6': 'dog',
    'dog7': 'dog',
    'dog8': 'dog',
    'duck_toy': 'duck toy',
    'fancy_boot': 'boot',
    'grey_sloth_plushie': 'sloth plushie',
    'monster_toy': 'toy',
    'pink_sunglasses': 'sunglasses',
    'poop_emoji': 'emoji toy',
    'rc_car': 'toy car',
    'red_cartoon': 'cartoon',
    'robot_toy': 'toyrobot',
    'shiny_sneaker': 'sneaker',
    'teapot': 'teapot',
    'vase': 'vase',
    'wolf_plushie': 'wolf plushie'
}

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.0,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 init_text = None,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text
        self.init_text = init_text
        
        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.BILINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        numbers = list(range(self.num_images))
        if len(numbers) > 1:
            numbers.remove(i % self.num_images)
        sel = random.choice(numbers)
        image_ref = Image.open(self.image_paths[sel])
        # clip_vis = torch.load(self.image_paths[sel].replace("images", "clip_feature").replace(".jpeg", ".pt"))
        # example["clip_vis"] = clip_vis
        
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        if not image_ref.mode == "RGB":
            image_ref = image_ref.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_smallest).format(placeholder_string)
        
        example["caption"] = text
        text_tokens = get_clip_token_for_string(self.tokenizer, text)
        ph_tokens = get_clip_token_for_string(self.tokenizer, [placeholder_string])
        ph_tok = ph_tokens[0,1]
        placeholder_idx = torch.where(text_tokens == ph_tok)
        endoftext_idx = (torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1))

        example["placeholder_pos"] = [placeholder_idx, endoftext_idx]

        # example["placeholder_pos"] = text.strip().split().index(placeholder_string) + 1
        
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        img_ref = np.array(image_ref).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
            
        if self.center_crop:
            crop = min(img_ref.shape[0], img_ref.shape[1])
            h, w, = img_ref.shape[0], img_ref.shape[1]
            img_ref = img_ref[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image_ref = Image.fromarray(img_ref)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            image_ref = image_ref.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        
        image_ref = self.flip(image_ref)
        image_ref = np.array(image_ref).astype(np.uint8)
        example["image_ref"] = (image_ref / 127.5 - 1.0).astype(np.float32)
        
        example["text_init"] = text.replace("*", self.init_text)
        return example
    

class PersonalizedBase_segment(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.0,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 init_text = None,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text
        self.init_text = init_text
        
        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.BILINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {} 
        image = Image.open(self.image_paths[i % self.num_images])

        numbers = list(range(self.num_images))
        if len(numbers) > 1:
            numbers.remove(i % self.num_images)
        sel = random.choice(numbers)
        image_ref = Image.open(self.image_paths[sel])
         
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        if not image_ref.mode == "RGB":
            image_ref = image_ref.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)
        
        example["caption"] = text + ", set in a pure black background"
        text_tokens = get_clip_token_for_string(self.tokenizer, text)
        ph_tokens = get_clip_token_for_string(self.tokenizer, [placeholder_string])
        ph_tok = ph_tokens[0,1]
        placeholder_idx = torch.where(text_tokens == ph_tok)
        endoftext_idx = (torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1))

        example["placeholder_pos"] = [placeholder_idx, endoftext_idx]

        # example["placeholder_pos"] = text.strip().split().index(placeholder_string) + 1
        
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        img_ref = np.array(image_ref).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
            
        if self.center_crop:
            crop = min(img_ref.shape[0], img_ref.shape[1])
            h, w, = img_ref.shape[0], img_ref.shape[1]
            img_ref = img_ref[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        image_ref = Image.fromarray(img_ref)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            image_ref = image_ref.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        
        image_ref = self.flip(image_ref)
        image_ref = np.array(image_ref).astype(np.uint8)
        example["image_ref"] = (image_ref / 127.5 - 1.0).astype(np.float32)
        
        example["text_init"] = text.replace("*", self.init_text)
        return example


class MultiConceptPersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 mask_root=None,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.0,
                 set="train",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 init_text=None,
                 drop_text_prob = 0.0,
                 drop_mask_prob = 0.1):

        self.data_root = data_root
        self.placeholder_tokens = placeholder_tokens 
        self.class_attributes = class_attributes
        self.concept_paths = {concept: os.path.join(data_root, concept) for concept in os.listdir(data_root)}
        self.mask_root = mask_root
       
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        for concept, path in self.concept_paths.items():
            if mask_root is not None:
                mask_path = os.path.join(mask_root, concept)
            for img_file in os.listdir(path):
                self.image_paths.append(os.path.join(path, img_file))
                if mask_root is not None:
                    self.mask_paths.append(os.path.join(mask_path, img_file.replace(".jpg", ".png")))
                self.labels.append(concept)
               

        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats if set == "train" else self.num_images

        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.init_text = init_text
        self.interpolation = self.get_interpolation(interpolation)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.drop_prob = drop_text_prob
        self.drop_mask_p = drop_mask_prob

    def get_interpolation(self, method):
        return {
            "linear": Image.BILINEAR,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[method]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {} 
        index = i % self.num_images
        image = Image.open(self.image_paths[index])
        label = self.labels[index]
        if self.mask_root is not None:
            mask_image = Image.open(self.mask_paths[index])
            if not mask_image.mode == "RGB":
                mask_image = mask_image.convert("RGB")
            processed_mask = self.process_image(mask_image)
            # Generating the binary mask where object is present (non-white)
            mask = (processed_mask.sum(axis=2) != 3).astype(np.float32)
            if random.random() <= self.drop_mask_p:
                mask = np.ones_like(mask)
            example["mask"] = mask
        else: 
            example["mask"] = None

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_tokens.get(label)
        attribute = self.class_attributes.get(label)  
        text = random.choice(imagenet_templates_mem).format(placeholder_string, attribute)
        #if save_memory
        # text = text.replace(placeholder_string, '')
        # print(text)
        
        example["caption"] = text
        # if random.random() <= self.drop_prob:
        #     example["caption"] = ""
       
        example["label"] = label
        example["image"] = self.process_image(image)
        
        text_init = text.replace(placeholder_string, self.init_text if self.init_text else label)
        example["text_init"] = text_init

        example["placeholder_pos"] = [0,1]
        example["image_ref"] = example["image"]#for debugging
        #print(example["caption"])
        return example

    def process_image(self, image):
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        return (np.array(image).astype(np.uint8) / 127.5 - 1.0).astype(np.float32)
