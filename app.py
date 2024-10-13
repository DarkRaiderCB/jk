import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display

def generate_comic_image(user_prompt):
    # Append style modifiers to the user's prompt
    comic_style_prompt = f"{user_prompt}, comic book style, comic illustration, bold lines, vibrant colors"

    # Load the Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure you have access to the model. You might need to accept the terms on Hugging Face.
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Generate the image
    with torch.autocast(device):
        image = pipe(comic_style_prompt).images[0]

    # Display the image in the notebook
    display(image)

if __name__ == "__main__":
    user_input = input("Enter a description for the image: ")
    generate_comic_image(user_input)
