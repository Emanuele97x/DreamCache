import argparse, os, sys, glob

sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.personalized import PersonalizedBase
from evaluation.clip_eval import ImageDirEvaluator, CLIPImageDataset
from torch.utils.data import DataLoader

def save_scores(file_name, image_similarity, text_similarity):
        with open(file_name, 'a') as f:
            f.write(f"Image Similarity: {image_similarity}\n")
            f.write(f"Text Similarity: {text_similarity}\n")


def load_images_from_folder(folder):
    image_files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
    print(image_files)
    return image_files

def save_image_batch(images, captions, file_name):
    """Save a batch of images with captions."""
    # Ensure the output directory exists
    os.makedirs("debug_images", exist_ok=True)
    
    # Create a grid of images
    grid = vutils.make_grid(images, normalize=True, scale_each=True)
    plt.figure(figsize=(16, 16))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    
    # Prepare caption text
    for i, caption in enumerate(captions):
        plt.text(10, i * 32 + 20, caption, color='white', backgroundcolor='black', fontsize=8)
    
    # Save the figure
    plt.savefig(os.path.join("debug_images", file_name))
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to directory with images used to train the embedding vectors"
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        help="Path to directory with images used to train the embedding vectors"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        help="Path to directory with images used to train the embedding vectors"
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Path to directory with images used to train the embedding vectors"
    )

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Assign a value to the device variable
    evaluator = ImageDirEvaluator(device)  # Instantiate the evaluator object

    
    captions = [
        "a * in the jungle",
        "a * in the snow",
        "a * on the beach",
        "a * on a cobblestone street",
        "a * on top of pink fabric",
        "a * on top of a wooden floor",
        "a * with a city in the background",
        "a * with a mountain in the background",
        "a * with a blue house in the background",
        "a * on top of a purple rug in a forest",
        "a * with a wheat field in the background",
        "a * with a tree and autumn leaves in the background",
        "a * with the Eiffel Tower in the background",
        "a * floating on top of water",
        "a * floating in an ocean of milk",
        "a * on top of green grass with sunflowers around it",
        "a * on top of a mirror",
        "a * on top of the sidewalk in a crowded street",
        "a * on top of a dirt road",
        "a * on top of a white rug",
        "a red *",
        "a purple *",
        "a shiny *",
        "a wet *",
        "a cube shaped *"
    ]

    
    class_attributes = {
        'backpack': 'backpack',
        'backpack_dog': 'backpack',
        'bear_plushie': 'bear plushie',
        'berry_bowl': 'bowl',
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
        'robot_toy': 'robot toy',
        'shiny_sneaker': 'sneaker',
        'teapot': 'teapot',
        'vase': 'vase',
        'wolf_plushie': 'wolf plushie'
    }

    
    src_images = load_images_from_folder(opt.data_dir)
    generated_images = load_images_from_folder(opt.generated_dir)

    src_images_dataset = CLIPImageDataset(src_images)
    generated_images_dataset = CLIPImageDataset(generated_images)

    src_loader = DataLoader(src_images_dataset, batch_size=len(src_images), shuffle=False)
    generated_loader = DataLoader(generated_images_dataset, batch_size=4, shuffle=False)

    total_sim_img = 0
    total_sim_text = 0
    num_captions = len(captions)

    src_batch = next(iter(src_loader)).to(device)

   
    for i, (caption, gen_batch) in enumerate(zip(captions, generated_loader)):
        target_text = caption.replace("*", class_attributes[opt.class_name])
        #src_batch = next(iter(src_loader)) 

        sim_img, sim_text = evaluator.evaluate(gen_batch.to(device), src_batch, target_text, opt.class_name)
       
        #save_image_batch(gen_batch, [target_text] * 4, f"caption_{i}_images.png")


        total_sim_img += sim_img
        total_sim_text += sim_text

        print(f"Caption: {caption}")
        print("Image similarity: ", sim_img)
        print("Text similarity: ", sim_text)

    avg_sim_img = total_sim_img / num_captions
    avg_sim_text = total_sim_text / num_captions

    save_scores(f"all_scores_{opt.tag}.txt", avg_sim_img, avg_sim_text)
    print("Average Image Similarity across all captions: ", avg_sim_img)
    print("Average Text Similarity across all captions: ", avg_sim_text)