import streamlit as st
from diffusers import StableDiffusionPipeline
import torch


@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    # Use Accelerate's device_map to automatically handle device placement
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    return pipe


def generate_image(prompt):
    pipe = load_model()
    image = pipe(prompt).images[0]
    return image


def main():
    st.title("Custom Jewelry Design Generator")
    st.write("Describe your dream jewelry piece, and we'll generate an image of it!")

    prompt = st.text_area("Enter your jewelry description here:", height=150)

    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image..."):
                image = generate_image(prompt)
                st.image(image, caption="Generated Jewelry Design")
        else:
            st.warning("Please enter a description.")


if __name__ == "__main__":
    main()
