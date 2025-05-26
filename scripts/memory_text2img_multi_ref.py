import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import PIL
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_multicond
from ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange



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
    'berry_bowl': 'bowl',
    'can': 'can',
    'candle': 'candle',
    'cat': 'cat',
    'cat2': 'cat',
    'clock': 'clock',
    'colorful_sneaker': 'sneakers',
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
    'robot_toy': 'toy robot',
    'shiny_sneaker': 'sneaker',
    'teapot': 'teapot',
    'vase': 'vase',
    'wolf_plushie': 'wolf plushie'
}


def image_process(img_path):
    image = Image.open(img_path)

    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image).astype(np.uint8)
    image = Image.fromarray(img)
    image = image.resize((512, 512), resample=PIL.Image.BILINEAR)

    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)
    return image

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    # assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens

def load_imageca(model, image_ca_path):  
    state_dict = torch.load(image_ca_path, map_location='cpu')
    state_dict = {"model." + k: v for k, v in state_dict.items()}
    print(f"KEYS_STATE = {state_dict.keys()}")
    model.load_state_dict(state_dict, strict=False)
    return model



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default="memories/dreambooth_all",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--image_scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--ft_path", 
        type=str, 
        help="Path to a fine-tuned checkpoint")
    
    parser.add_argument(
        "--load_step",
        type=int,
        default=299,
        help="Training step used to infer"
    )
    
    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")
    
    parser.add_argument(
        "--class_key", 
        type=str, 
        help="Path to the class we want to generate")
    
    parser.add_argument(
        "--class_key2", 
        type=str, 
        default = None,
        help="Path to the class we want to generate")

    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to a sample image, one image for now."
    )



    opt = parser.parse_args()
    seed_everything(opt.seed)
    
    image_ca_path = os.path.join(opt.ft_path, "checkpoints/cross_attention-{}.pt".format(opt.load_step))
    
    print("cross attention path: " + image_ca_path)
    #print("embedding path: " + embedding_path)
    

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt_path}")
    
    
     
    model = load_imageca(model, image_ca_path)
    model.model.freeze_imageca() 
    
    print("Model Loaded")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    print("Model moved to device")
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler_multicond(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    class_key = opt.class_key
    class_key2 = opt.class_key2
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
        
        class_attr = class_attributes[class_key]
        data = [prompt.replace('class', class_attr) for prompt in data]
        data = [prompt for prompt in data for _ in range(batch_size)]  
        data = list(chunk(data, batch_size))  

    print(f"CLASS KEY: {class_key}")
    labels = [class_key]
    if class_key2 is not None:
        labels.append(class_key2)

    print(f"CLASS KEY: {labels}")
    negative_prompt = "Open mouth, tongue out, teeth visible, bad photo, low quality, blurred, deformed"
    print(data)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                ph_pos = [0,1]
                image = image_process(opt.image_path).to(device)
                start_time = time.time()
                encoder_posterior = model.encode_first_stage(image)
                xr1 = model.get_first_stage_encoding(encoder_posterior).detach()
                
                xr1 = xr1.expand(opt.n_samples, -1, -1, -1)
                uc = model.get_learned_conditioning(batch_size * [negative_prompt] )
                t = torch.tensor([1,], dtype=torch.long).to(device)
                x_noisy = model.q_sample(x_start=xr1, t=t)
                h1 = model.apply_model_get_h(x_noisy, xr1, t, uc, uc, ph_pos, label = None)
                
            
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                      
           
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         image_cond = h1,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         image_scale = opt.image_scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code,
                                                         label = labels)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        end_time = time.time()
                        inference_time = end_time - start_time
                        print(f"Inference Time: {inference_time:.2f} seconds")

                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.jpg"))
                                base_count += 1

                       

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()