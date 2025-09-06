"""
Streamlit app for recoloring PNG logos while preserving transparency.

This application allows users to upload a PNG file with a transparent background,
select a new color via either a hex code input or individual RGB sliders, and
generate a recolored version of the uploaded logo. The recoloring process
preserves the original alpha (transparency) channel so that the transparent
regions remain completely transparent in the output. The app displays both
the original and recolored images side by side for easy comparison and
provides a download button to save the recolored image at the same
resolution and quality as the input.

To run this app locally, execute:

    streamlit run app.py

Dependencies are listed in requirements.txt.
"""

import io
import os
import zipfile
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin
import streamlit as st


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert a hex color string to an (R, G, B) tuple.

    Parameters
    ----------
    hex_color : str
        A hex color string beginning with '#', e.g., '#FF5733'.

    Returns
    -------
    tuple
        A tuple of three integers representing the red, green, and blue
        components of the color.

    Raises
    ------
    ValueError
        If the provided string is not in a valid hex format.
    """
    color = hex_color.lstrip('#')
    if len(color) != 6:
        raise ValueError("Hex color must be in the format #RRGGBB")
    return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an (R, G, B) tuple to a hex string beginning with '#'."""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def has_alpha(image: Image.Image) -> bool:
    """Return True if the image contains an alpha channel."""
    return 'A' in image.getbands()


def apply_white_threshold(image: Image.Image, threshold: int) -> Image.Image:
    """
    Treat near-white pixels as transparent by generating an alpha channel based on a threshold.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in RGB or RGBA mode.
    threshold : int
        A value between 0 and 255. Pixels with all RGB components greater than
        or equal to this threshold will become fully transparent.

    Returns
    -------
    PIL.Image.Image
        An RGBA image with updated alpha channel.
    """
    # Convert to numpy array for efficient processing
    arr = np.array(image.convert('RGBA'))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    # Identify near-white pixels: all channels >= threshold
    mask = np.all(rgb >= threshold, axis=2)
    # Set alpha to 0 where mask is True, else keep existing alpha (255 if no alpha existed)
    alpha = np.where(mask, 0, alpha)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, 'RGBA')


def recolor_flat(image: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
    """
    Recolor the non-transparent parts of an image by completely replacing
    their RGB values with the selected color, preserving the original alpha
    channel. This ignores the original luminance, so black, gray or any
    color becomes exactly the chosen color in the output.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image in RGBA or RGB mode.
    color : tuple
        An (R, G, B) tuple representing the target color.

    Returns
    -------
    PIL.Image.Image
        The recolored image in RGBA mode.
    """
    rgba = image.convert('RGBA')
    r, g, b, alpha = rgba.split()
    solid = Image.new('RGBA', rgba.size, color + (255,))
    solid.putalpha(alpha)
    return solid


def recolor_tone_preserving(image: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
    """
    Recolor the non-transparent parts of an image while preserving the
    luminance (shading) of the original pixels. Dark areas remain dark and
    light areas become lighter shades of the selected color. The alpha
    channel is preserved.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in RGBA or RGB mode.
    color : tuple
        Target color as an (R, G, B) tuple.

    Returns
    -------
    PIL.Image.Image
        The recolored image in RGBA mode.
    """
    rgba = image.convert('RGBA')
    # Extract alpha channel
    alpha = rgba.getchannel('A')
    # Convert to grayscale to capture luminance
    gray = ImageOps.grayscale(rgba)
    # Colorize: map black to black and white to selected color
    colored = ImageOps.colorize(gray, black=(0, 0, 0), white=color)
    recolored = colored.convert('RGBA')
    recolored.putalpha(alpha)
    return recolored


def extract_png_metadata(image: Image.Image) -> Tuple[PngImagePlugin.PngInfo, Tuple[int, int]]:
    """
    Extract metadata and DPI information from a PNG image. This helps
    preserve metadata when saving a new PNG.

    Parameters
    ----------
    image : PIL.Image.Image
        The image from which to extract metadata.

    Returns
    -------
    tuple
        A tuple containing a PngInfo object with text metadata and a
        tuple for DPI (or None if not present).
    """
    metadata = PngImagePlugin.PngInfo()
    for k, v in image.info.items():
        # PIL stores some values like ICC profiles and gamma that cannot be
        # set via add_text. Ignore those.
        try:
            metadata.add_text(k, str(v))
        except Exception:
            continue
    dpi = image.info.get('dpi')
    return metadata, dpi


def recolor_image(*args, **kwargs):
    """
    Deprecated placeholder maintained for backward compatibility.

    This function will be removed in a future release. Use
    `recolor_flat` or `recolor_tone_preserving` instead.
    """
    raise NotImplementedError(
        "recolor_image is deprecated. Use recolor_flat or recolor_tone_preserving instead."
    )


def main() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(page_title="Anthony’s Special Little Color Helper")
    st.title("Anthony’s Special Little Color Helper")
    st.write(
        """
        Upload one or more PNG logos with transparency, choose a new color, and
        recolor the non-transparent parts of each image. You can also opt to
        preserve shading or treat near‑white backgrounds as transparent when
        your images lack an alpha channel. The recolored images maintain the
        original resolution and metadata (such as DPI), and remain fully
        transparent where appropriate.
        """
    )

    # Color selection with synchronized hex and RGB sliders
    # Initialize session state
    if 'hex_color' not in st.session_state:
        st.session_state['hex_color'] = '#FF0000'
    if 'r' not in st.session_state:
        st.session_state['r'], st.session_state['g'], st.session_state['b'] = hex_to_rgb(st.session_state['hex_color'])

    def _update_rgb_from_hex() -> None:
        """Update RGB sliders when the hex color picker changes."""
        try:
            r, g, b = hex_to_rgb(st.session_state['hex_color'])
            st.session_state['r'] = r
            st.session_state['g'] = g
            st.session_state['b'] = b
        except Exception:
            # If invalid hex, do nothing
            pass

    def _update_hex_from_rgb() -> None:
        """Update hex color picker when any RGB slider changes."""
        st.session_state['hex_color'] = rgb_to_hex((st.session_state['r'], st.session_state['g'], st.session_state['b']))

    st.subheader("Select Color")
    # Color picker (hex) and update callback
    st.color_picker(
        "Choose a color",
        key='hex_color',
        value=st.session_state['hex_color'],
        on_change=_update_rgb_from_hex,
    )
    # RGB sliders with update callback
    cols = st.columns(3)
    with cols[0]:
        st.slider("Red", 0, 255, key='r', on_change=_update_hex_from_rgb)
    with cols[1]:
        st.slider("Green", 0, 255, key='g', on_change=_update_hex_from_rgb)
    with cols[2]:
        st.slider("Blue", 0, 255, key='b', on_change=_update_hex_from_rgb)
    selected_rgb = (st.session_state['r'], st.session_state['g'], st.session_state['b'])
    st.write(f"Selected color hex: {st.session_state['hex_color']}")

    # Choose recolor mode
    mode = st.radio("Recoloring mode", ("Flat Recolor", "Tone-Preserving"), index=0)

    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Upload one or more PNG files", type=["png"], accept_multiple_files=True
    )
    # List to collect recolored image buffers and filenames for ZIP creation
    zip_entries: List[Tuple[str, bytes]] = []

    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                img = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error loading image '{uploaded_file.name}': {e}")
                continue

            # Preserve original metadata for later
            metadata, dpi = extract_png_metadata(img)

            # Determine if the image has an alpha channel
            has_alpha_channel = has_alpha(img)

            # Process per-image UI in a container to keep controls grouped
            with st.container():
                st.markdown(f"### {uploaded_file.name}")
                # Handle images without alpha channel
                if not has_alpha_channel:
                    st.info(
                        "This image lacks an alpha (transparency) channel. You can choose to "
                        "treat near‑white pixels as transparent using the option below."
                    )
                    treat_key = f"treat_white_{idx}"
                    thresh_key = f"threshold_{idx}"
                    treat_white = st.checkbox(
                        "Treat near‑white as transparent", key=treat_key, value=False
                    )
                    if treat_white:
                        threshold = st.slider(
                            "White threshold (higher makes more pixels transparent)",
                            0,
                            255,
                            250,
                            key=thresh_key,
                        )
                        # Apply threshold to create alpha channel
                        img_rgba = apply_white_threshold(img, threshold)
                    else:
                        img_rgba = img.convert('RGBA')
                else:
                    img_rgba = img.convert('RGBA')

                # Recolor according to selected mode
                if mode == "Flat Recolor":
                    recolored = recolor_flat(img_rgba, selected_rgb)
                else:
                    recolored = recolor_tone_preserving(img_rgba, selected_rgb)

                # Display original and recolored images side by side
                orig_col, recol_col = st.columns(2)
                with orig_col:
                    st.subheader("Original")
                    st.image(img_rgba, caption=uploaded_file.name, use_container_width=True)
                with recol_col:
                    st.subheader("Recolored")
                    st.image(recolored, caption=f"Recolored", use_container_width=True)

                # Prepare download buffer for this recolored image
                buffer = io.BytesIO()
                # Determine output filename based on selected color
                base, _ = os.path.splitext(uploaded_file.name)
                out_name = f"{base}_{st.session_state['hex_color'].lstrip('#')}.png"
                # Save image with metadata and DPI
                save_kwargs = {}
                if dpi:
                    save_kwargs['dpi'] = dpi
                try:
                    recolored.save(buffer, format='PNG', pnginfo=metadata, **save_kwargs)
                except Exception:
                    # If metadata injection fails, fallback without metadata
                    buffer = io.BytesIO()
                    recolored.save(buffer, format='PNG', **save_kwargs)
                buffer.seek(0)

                # Single download button
                st.download_button(
                    label=f"Download {out_name}",
                    data=buffer.getvalue(),
                    file_name=out_name,
                    mime="image/png",
                    key=f"download_{idx}"
                )
                # Collect for ZIP if multiple
                zip_entries.append((out_name, buffer.getvalue()))

        # Provide batch download as ZIP if more than one file uploaded
        if len(zip_entries) > 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname, data in zip_entries:
                    zf.writestr(fname, data)
            zip_buffer.seek(0)
            st.download_button(
                label="Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name="recolored_images.zip",
                mime="application/zip"
            )


if __name__ == "__main__":
    main()