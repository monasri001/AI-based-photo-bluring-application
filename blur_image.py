import streamlit as st
from PIL import Image, ImageOps, ImageFilter
from transformers import pipeline
import torch
import os
import logging
import numpy as np
from typing import Tuple, Optional
import io

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Background Blur Class --------------------
class BackgroundBlur:
    def __init__(self, model_path: str):
        """Initialize the BackgroundBlur class with model path."""
        self.model_path = model_path
        self.device = 0 if torch.cuda.is_available() else -1
        self.load_model()

    def load_model(self) -> None:
        """Load the image segmentation model."""
        try:
            self.model = pipeline(
                task="image-segmentation",
                model=self.model_path,
                device=self.device
            )
            logger.info(f"Model loaded successfully on device {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error("Failed to load the segmentation model. Check model path.")

    def refine_mask(self, mask: Image.Image) -> Image.Image:
        """Refine segmentation mask for smoother edges."""
        mask_array = np.array(mask)

        # Binary threshold
        threshold = 128
        mask_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)

        refined_mask = Image.fromarray(mask_array)

        # Slight blur for smooth edges
        refined_mask = refined_mask.filter(
            ImageFilter.GaussianBlur(radius=0.5)
        )

        return refined_mask

    def process_image(
        self,
        image: Image.Image,
        blur_level: int
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Process image and apply background blur."""
        try:
            result = self.model(images=image)
            mask = result[0]["mask"]

            refined_mask = self.refine_mask(mask)
            inverted_mask = ImageOps.invert(refined_mask)

            background = image.copy()
            background = background.filter(
                ImageFilter.GaussianBlur(radius=blur_level)
            )

            final_image = Image.composite(
                image, background, inverted_mask
            )

            return final_image, None

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None, str(e)

# -------------------- Streamlit UI --------------------
def main():
    st.set_page_config(
        page_title="Background Blur App",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stButton button { width: 100%; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🖼️ AI Background Blur App")

    if "processed_image" not in st.session_state:
        st.session_state.processed_image = None

    # ⚠️ Update this path if needed
    model_path = os.path.join(
        "..",
        "Model",
        "models--nvidia--segformer-b2-clothes",
        "snapshots",
        "fc92b3abe7b123c814ca7910683151f2b7b7281e"
    )

    processor = BackgroundBlur(model_path)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image:
            original_image = Image.open(uploaded_image).convert("RGB")
            st.image(
                original_image,
                caption="Original Image",
                use_column_width=True
            )

    with col2:
        st.subheader("Processed Image")

        if uploaded_image:
            blur_level = st.slider(
                "Blur Intensity",
                min_value=0,
                max_value=30,
                value=15,
                step=1
            )

            process_button = st.button("Process Image")

            if process_button:
                with st.spinner("Processing image..."):
                    final_image, error = processor.process_image(
                        original_image,
                        blur_level
                    )

                if error:
                    st.error(f"Error: {error}")
                else:
                    st.session_state.processed_image = final_image
                    st.image(
                        final_image,
                        caption="Processed Image",
                        use_column_width=True
                    )

                    # Download Button
                    buf = io.BytesIO()
                    final_image.save(buf, format="PNG")
                    st.download_button(
                        label="Download Processed Image",
                        data=buf.getvalue(),
                        file_name="processed_image.png",
                        mime="image/png"
                    )

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    main()
