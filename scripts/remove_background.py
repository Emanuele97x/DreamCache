import os
from pathlib import Path
from PIL import Image
from rembg import new_session, remove
import io

def process_images(input_folder, output_folder):
    # Define the model for the background removal session
    model_name = "isnet-general-use"
    rembg_session = new_session(model_name)
    
    # Ensure the output directory exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Walk through the input directory and process each PNG or JPG file
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # Set up output file path
                output_path = os.path.join(output_subfolder, file)

                # Open the image, remove the background, and save the output
                with Image.open(input_path) as img:
                    output_image = remove(img, session=rembg_session)
                    white_bg = Image.new("RGB", output_image.size, "WHITE")
                    # Paste the transparent image onto the white background
                    white_bg.paste(output_image.convert('RGBA'), (0, 0), output_image)
                    # Convert the final image to RGB to remove the alpha channel
                    final_image = white_bg.convert('RGB')
                    # Save the image without an alpha channel
                    final_image.save(output_path)

                print(f"Processed {output_path}")

if __name__ == "__main__":
    input_folder = '/path/to/dreambooth/dataset'
    output_folder = '/path/to/dreambooth_nobackground/dataset'
    process_images(input_folder, output_folder)
