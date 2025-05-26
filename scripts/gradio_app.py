import gradio as gr
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import PIL
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler_multicond
from einops import rearrange
import os
import time
from typing import Union, Optional, Tuple

# --------------------------------------------------
# Background Removal with rembg
# --------------------------------------------------
from rembg import new_session, remove



# Create a rembg session using the "isnet-general-use" model.
rembg_session = new_session("isnet-general-use")


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
# --------------------------------------------------
# Model Loading and Processing Utilities
# --------------------------------------------------

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}", flush=True)
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}", flush=True)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("Missing keys:")
        print(m, flush=True)
    if len(u) > 0 and verbose:
        print("Unexpected keys:")
        print(u, flush=True)
    model.eval()
    return model

def load_imageca(model, image_ca_path):  
    state_dict = torch.load(image_ca_path, map_location='cpu')
    state_dict = {"model." + k: v for k, v in state_dict.items()}
    print(f"Loaded image cross-attention weights with keys: {list(state_dict.keys())}", flush=True)
    model.load_state_dict(state_dict, strict=False)
    return model

def process_input_image(image: Image.Image):
    """
    Removes the background using rembg, fills the transparent areas 
    with the specified color using ColorBackgroundRGBA, converts to RGB,
    resizes the image to 512x512, saves the processed image as 'no_bg_image.png'
    for a sanity check, and converts it into a tensor normalized in the range [-1, 1].
    """
    print("Starting background removal...", flush=True)
    image_no_bg = remove(image, session=rembg_session)
    print("Background removal complete.", flush=True)
    
    # Use the provided ColorBackgroundRGBA function to fill transparent pixels.
    # Here we fill with white; change the tuple as needed.
    color_bg = ColorBackgroundRGBA((255, 255, 255))
    image_no_bg_colored = color_bg(image_no_bg)
    
    # Convert to RGB (drop the alpha channel) since the background is now colored.
    image_no_bg_colored = image_no_bg_colored.convert("RGB")
    
    # Resize the image to 512x512.
    image_no_bg_colored = image_no_bg_colored.resize((512, 512), resample=PIL.Image.BILINEAR)
    
    # Save the processed image as a PNG for sanity check.
    image_no_bg_colored.save("no_bg_image.png", "PNG")
    print("Sanity check image saved as 'no_bg_image.png'.")
    
    # Convert image to numpy array, normalize to [-1, 1], and convert to tensor.
    np_image = np.array(image_no_bg_colored).astype(np.uint8)
    np_image = (np_image / 127.5 - 1.0).astype(np.float32)
    tensor_image = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
    print(tensor_image.shape)
    
    return tensor_image



# Global variables for the model, sampler, and device.
model = None
sampler = None
device = None

def init_model():
    
    global model, sampler, device
    seed_everything(22)
    load_step = 199999
    CKPT_PATH = "/v2-1_512-ema-pruned.ckpt"
    FT_PATH = "/your_finetuned_checkpoint_path"
    MEMORY_PATH_BASE = ""
    OUTDIR_BASE = "your_outdir"
    

    config = OmegaConf.load("configs/stable-diffusion/v2-inference.yaml")
    config.model.params.memory_path = ""
    model = load_model_from_config(config, CKPT_PATH)
    image_ca_path = os.path.join(FT_PATH, "checkpoints", f"cross_attention-{load_step}.pt")
    model = load_imageca(model, image_ca_path)
    model.model.freeze_imageca()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Move model to GPU and convert to half precision.
    model = model.to(device)
    sampler = DDIMSampler_multicond(model)
    print("Model loaded and moved to GPU with half precision.", flush=True)

def gradio_generate(reference_image, image_scale, guidance_scale, prompt):
    """
    Takes a reference image, slider values, and a text prompt to generate four personalized alternatives.
    """
    global model, sampler, device
    print("Generation started.", flush=True)
    # Process image and move to GPU.
    image_tensor = process_input_image(reference_image).to(device)
    print("Image processed.", flush=True)
    batch_size = 4
    negative_prompt = "Open mouth, tongue out, teeth visible, bad photo, low quality, blurred, deformed"
    
    precision_scope = autocast #if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                print("Before encode_first_stage", flush=True)
                encoder_posterior = model.encode_first_stage(image_tensor)
                torch.cuda.synchronize()
                print("After encode_first_stage", flush=True)
                
                xr1 = model.get_first_stage_encoding(encoder_posterior).detach()
                xr1 = xr1.expand(batch_size, -1, -1, -1)
                torch.cuda.synchronize()
                print("After get_first_stage_encoding", flush=True)
                
                uc = model.get_learned_conditioning(batch_size * [negative_prompt]).to(device)
                print("After negative conditioning", flush=True)
                
                t = torch.tensor([1], dtype=torch.long, device=device)
                x_noisy = model.q_sample(x_start=xr1, t=t)
                print("After q_sample", flush=True)
                
                h1 = model.apply_model_get_h(x_noisy, xr1, t, uc, uc, [0,1], label=None)
                print("After apply_model_get_h", flush=True)
                
                conditioning_prompts = [prompt] * batch_size
                c = model.get_learned_conditioning(conditioning_prompts).to(device)
                print("After conditioning for prompt", flush=True)
                
                shape = [4, 512 // 8, 512 // 8]
                start_code = None  # Use None for random starting noise
                
                samples_ddim, _ = sampler.sample(
                    S=50,
                    conditioning=c,
                    image_cond=h1,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=guidance_scale,
                    image_scale=image_scale,
                    unconditional_conditioning=uc,
                    eta=0.0,
                    x_T=start_code,
                    label=None,
                )
                print("After sampler.sample", flush=True)
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                print("After decode_first_stage", flush=True)
                
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                images_out = []
                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample_np = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    x_sample_np = np.clip(x_sample_np, 0, 255).astype(np.uint8)
                    images_out.append(Image.fromarray(x_sample_np))
                print("Generation complete.", flush=True)
                return images_out

# --------------------------------------------------
# Gradio Interface Setup
# --------------------------------------------------

demo = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Image(type="pil", label="Reference Image"),
        gr.Slider(minimum=0, maximum=11.5, step=0.1, value=5.0, label="Image Scale"),
        gr.Slider(minimum=0, maximum=11.5, step=0.1, value=7.5, label="Guidance Scale"),
        gr.Textbox(lines=2, placeholder="Enter prompt here...", label="Prompt", value="a personalized version"),
    ],
    outputs=gr.Gallery(label="Generated Alternatives"),
    title="Personalized Generation Demo",
    description="Upload a reference image, adjust the sliders, and provide a prompt. The model will generate four personalized alternatives."
)

if __name__ == "__main__":
    init_model()
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)
    