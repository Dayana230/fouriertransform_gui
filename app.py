from typing import final
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Sidebar controls
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#FFFFFF")
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

def normalize_image(img):
    img = img / np.max(img)
    return (img * 255).astype('uint8')

def write_background_images(images, names):
    for image, name in zip(images, names):
        image3 = cv2.merge((image, image, image))
        image_3_nor = normalize_image(image3)
        Image.fromarray(image_3_nor).save(name)  # Save in PIL format

def create_canvas_draw_instance(background_image, key, height, width):
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=background_image,  # Now passing PIL image directly
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        height=height,
        width=width,
        key=key,
    )
    return canvas_result

def rgb_fft(image):
    fft_images = []
    fft_images_log = []
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2(image[:, :, i]))
        fft_images.append(rgb_fft)
        fft_images_log.append(np.log(abs(rgb_fft)))
    return fft_images, fft_images_log

def main():
    st.header("Fourier Transformation - ")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "png", "jpg"])
    
    if uploaded_file is not None:
        original = Image.open(uploaded_file)
        img = np.array(original)
        st.image(img, use_column_width=True)
        
        fft_images, fft_images_log = rgb_fft(img)
        names = ["bg_image_r.png", "bg_image_g.png", "bg_image_b.png"]
        
        write_background_images(fft_images_log, names)
        
        st.text("Red Channel in frequency domain - ")
        canvas_r = create_canvas_draw_instance(Image.open(names[0]), key="red", height=img.shape[0], width=img.shape[1])
        st.text("Green Channel in frequency domain - ")
        canvas_g = create_canvas_draw_instance(Image.open(names[1]), key="green", height=img.shape[0], width=img.shape[1])
        st.text("Blue channel in frequency domain - ")
        canvas_b = create_canvas_draw_instance(Image.open(names[2]), key="blue", height=img.shape[0], width=img.shape[1])
        
if __name__ == "__main__":
    main()
