"""Encoding helpers for hiding data in images using LSB steganography."""

import base64
import os
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple

from PIL import Image

TWITTER_MAX_BYTES = 900 * 1024
OUTPUT_FORMATS = {
    "png": {"ext": ".png", "mime": "image/png"},
    "jpeg": {"ext": ".jpeg", "mime": "image/jpeg"},
}


def normalize_output_format(output_format: Optional[str]) -> str:
    if not output_format:
        return "png"
    fmt = output_format.strip().lower()
    if fmt.startswith("."):
        fmt = fmt[1:]
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format '{output_format}'. Use png or jpeg.")
    return fmt


def output_format_extension(output_format: Optional[str]) -> str:
    fmt = normalize_output_format(output_format)
    return OUTPUT_FORMATS[fmt]["ext"]


def output_format_mime(output_format: Optional[str]) -> str:
    fmt = normalize_output_format(output_format)
    return OUTPUT_FORMATS[fmt]["mime"]


def _save_png(img: Image.Image, output_path: Path) -> None:
    img.save(output_path, format="PNG", optimize=True)


def convert_to_png(image_path: Path) -> Path:
    """Convert the image to PNG format if needed."""
    try:
        img = Image.open(image_path)
        # Verify the image by loading it
        img.verify()
        # Reopen after verify since verify closes the file
        img = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Failed to open image file '{image_path}': {str(e)}")

    try:
        if img.format != "PNG":
            new_image_path = image_path.with_suffix(".png")
            img.save(new_image_path, "PNG")
            return new_image_path
        return image_path
    except Exception as e:
        raise IOError(f"Failed to convert image to PNG format: {str(e)}")


def compress_image_before_encoding(image_path: Path, output_image_path: Path) -> None:
    """
    Compress the image before encoding to keep the file under the ~900 KB
    threshold where Twitter avoids heavy recompression that would destroy LSBs.
    """
    try:
        png_path = convert_to_png(image_path)
    except Exception as e:
        raise ValueError(f"Failed to convert image to PNG during compression: {str(e)}")

    try:
        img = Image.open(png_path)
        _save_png(img, output_image_path)
    except Exception as e:
        raise IOError(f"Failed to save initial compressed image: {str(e)}")

    # Compress until we are under ~900 KB to survive Twitter's pipeline.
    # Safety limit to prevent infinite loop if image can't be compressed enough
    max_iterations = 20
    iteration = 0

    while os.path.getsize(output_image_path) > TWITTER_MAX_BYTES:
        if iteration >= max_iterations:
            raise ValueError(
                f"Unable to compress image below 900KB after {max_iterations} iterations. "
                f"Current size: {os.path.getsize(output_image_path)} bytes"
            )

        try:
            img = Image.open(output_image_path)
            new_width = max(1, img.width // 2)
            new_height = max(1, img.height // 2)

            if new_width < 2 or new_height < 2:
                raise ValueError(
                    f"Image too small to compress further (size: {new_width}x{new_height}). "
                    f"Current file size: {os.path.getsize(output_image_path)} bytes exceeds 900KB limit."
                )

            img = img.resize((new_width, new_height))
            _save_png(img, output_image_path)
            iteration += 1
        except ValueError:
            raise
        except Exception as e:
            raise IOError(f"Failed to resize/save image during compression iteration {iteration}: {str(e)}")


def encode_text_into_plane(
    image: Image.Image, text: str, output_path: Path, plane: str = "RGB"
) -> None:
    """Embed text into selected color plane(s)."""
    if not isinstance(text, str):
        raise TypeError(f"Text must be a string, got {type(text).__name__}")

    if not text:
        raise ValueError("Cannot encode empty text")

    try:
        img = image.convert("RGBA")
    except Exception as e:
        raise ValueError(f"Failed to convert image to RGBA format: {str(e)}")

    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    try:
        binary_text = "".join(format(ord(char), "08b") for char in text) + "00000000"
    except Exception as e:
        raise ValueError(f"Failed to encode text to binary: {str(e)}")

    channels = [ch for ch in plane if ch in "RGBA"]
    if not channels:
        raise ValueError("plane must include at least one of R,G,B,A")

    max_bits = width * height * len(channels)
    if len(binary_text) > max_bits:
        raise ValueError(
            f"The message is too long for this image. "
            f"Message requires {len(binary_text)} bits but image only has capacity for {max_bits} bits. "
            f"Try using a larger image or fewer color channels."
        )

    try:
        index = 0
        for y in range(height):
            for x in range(width):
                if index >= len(binary_text):
                    break
                r, g, b, a = img.getpixel((x, y))

                for ch in channels:
                    bit = int(binary_text[index])
                    index += 1
                    if ch == "R":
                        r = (r & 0xFE) | bit
                    elif ch == "G":
                        g = (g & 0xFE) | bit
                    elif ch == "B":
                        b = (b & 0xFE) | bit
                    elif ch == "A":
                        a = (a & 0xFE) | bit
                    if index >= len(binary_text):
                        break

                img.putpixel((x, y), (r, g, b, a))
            if index >= len(binary_text):
                break

        _save_png(img, output_path)
    except Exception as e:
        raise IOError(f"Failed to encode text into image: {str(e)}")


def encode_zlib_into_image(
    image: Image.Image, file_data: bytes, output_path: Path, plane: str = "RGB"
) -> None:
    """Embed zlib-compressed data into selected color plane(s)."""
    if not isinstance(file_data, bytes):
        raise TypeError(f"File data must be bytes, got {type(file_data).__name__}")

    if not file_data:
        raise ValueError("Cannot encode empty file data")

    try:
        compressed_data = zlib.compress(file_data)
    except Exception as e:
        raise ValueError(f"Failed to compress file data with zlib: {str(e)}")

    try:
        binary_data = _bits_from_zlib(compressed_data)
    except Exception as e:
        raise ValueError(f"Failed to convert compressed data to binary: {str(e)}")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    channels = [ch for ch in plane if ch in "RGBA"]
    if not channels:
        raise ValueError("plane must include at least one of R,G,B,A")

    max_bits = width * height * len(channels)
    if len(binary_data) > max_bits:
        raise ValueError(
            f"The compressed data is too long for this image. "
            f"Compressed data requires {len(binary_data)} bits but image only has capacity for {max_bits} bits. "
            f"Original size: {len(file_data)} bytes, compressed size: {len(compressed_data)} bytes. "
            f"Try using a larger image or fewer color channels."
        )

    try:
        img = image.convert("RGBA")
    except Exception as e:
        raise ValueError(f"Failed to convert image to RGBA format: {str(e)}")

    try:
        index = 0
        for y in range(height):
            for x in range(width):
                if index >= len(binary_data):
                    break
                r, g, b, a = img.getpixel((x, y))

                for ch in channels:
                    bit = int(binary_data[index])
                    index += 1
                    if ch == "R":
                        r = (r & 0xFE) | bit
                    elif ch == "G":
                        g = (g & 0xFE) | bit
                    elif ch == "B":
                        b = (b & 0xFE) | bit
                    elif ch == "A":
                        a = (a & 0xFE) | bit
                    if index >= len(binary_data):
                        break

                img.putpixel((x, y), (r, g, b, a))
            if index >= len(binary_data):
                break

        _save_png(img, output_path)
    except Exception as e:
        raise IOError(f"Failed to encode compressed data into image: {str(e)}")


def encode_payload(
    image_bytes: bytes,
    *,
    filename: str = "input.png",
    mode: str = "text",
    plane: str = "RGB",
    text: Optional[str] = None,
    file_data: Optional[bytes] = None,
    output_format: str = "png",
    lossy_output: bool = True,
) -> Tuple[str, bytes]:
    """
    Encode either text or binary data into an image and return the encoded image bytes.
    Returns tuple of (filename, bytes).
    """
    if not isinstance(image_bytes, bytes):
        raise TypeError(f"image_bytes must be bytes, got {type(image_bytes).__name__}")

    if not image_bytes:
        raise ValueError("Cannot encode into empty image data")

    if mode not in {"text", "zlib"}:
        raise ValueError(f"mode must be 'text' or 'zlib', got '{mode}'")

    output_format = normalize_output_format(output_format)
    if output_format == "jpeg" and lossy_output and "A" in plane.upper():
        raise ValueError("Alpha plane requires PNG output. Choose .png for A channel payloads.")

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / filename

        try:
            with open(input_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            raise IOError(f"Failed to write input image to temporary file: {str(e)}")

        output_path = tmp_dir / "encoded.png"

        try:
            compress_image_before_encoding(input_path, output_path)
        except Exception as e:
            raise ValueError(f"Failed to compress image before encoding: {str(e)}")

        try:
            img = Image.open(output_path)
        except Exception as e:
            raise ValueError(f"Failed to open compressed image: {str(e)}")

        if mode == "text":
            if not text:
                raise ValueError("Text payload required for text mode.")
            try:
                encode_text_into_plane(img, text, output_path, plane)
            except Exception as e:
                raise ValueError(f"Failed to encode text payload: {str(e)}")
        else:
            if file_data is None:
                raise ValueError("file_data required for zlib mode.")
            try:
                encode_zlib_into_image(img, file_data, output_path, plane)
            except Exception as e:
                raise ValueError(f"Failed to encode file payload: {str(e)}")

        output_name = f"encoded{output_format_extension(output_format)}"
        if output_format == "png" or (output_format == "jpeg" and not lossy_output):
            try:
                encoded_bytes = output_path.read_bytes()
            except Exception as e:
                raise IOError(f"Failed to read encoded image file: {str(e)}")
            return output_name, encoded_bytes

        converted_path = tmp_dir / output_name
        try:
            _convert_output_image(output_path, output_format, converted_path)
            encoded_bytes = converted_path.read_bytes()
        except Exception as e:
            raise IOError(f"Failed to convert encoded image to {output_format}: {str(e)}")

        return output_name, encoded_bytes


def as_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    """Return a data URL for the provided image bytes."""
    if not isinstance(img_bytes, bytes):
        raise TypeError(f"img_bytes must be bytes, got {type(img_bytes).__name__}")

    if not img_bytes:
        raise ValueError("Cannot create data URL from empty image data")

    try:
        b64 = base64.b64encode(img_bytes).decode()
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        raise ValueError(f"Failed to encode image as base64 data URL: {str(e)}")


def _bits_from_text(text: str) -> str:
    payload = text.encode("utf-8") + b"\x00"
    return "".join(format(b, "08b") for b in payload)


def _bits_from_zlib(data: bytes) -> str:
    length = len(data)
    prefix = length.to_bytes(4, "big")
    payload = prefix + data
    return "".join(format(b, "08b") for b in payload)


def _convert_output_image(source_path: Path, output_format: str, output_path: Path) -> None:
    fmt = normalize_output_format(output_format)
    if fmt == "png":
        output_path.write_bytes(source_path.read_bytes())
        return

    try:
        img = Image.open(source_path)
    except Exception as e:
        raise ValueError(f"Failed to open encoded image for conversion: {str(e)}")

    if img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")

    try:
        img.save(
            output_path,
            format="JPEG",
            quality=95,
            optimize=True,
            subsampling=0,
        )
    except Exception as e:
        raise IOError(f"Failed to save JPEG output: {str(e)}")


def encode_multi_channel(
    image_bytes: bytes,
    channel_payloads: Dict[str, Dict[str, Optional[bytes | str]]],
    *,
    filename: str = "input.png",
    twitter_safe_preprocess: bool = True,
    output_format: str = "png",
) -> Tuple[str, bytes]:
    """
    Encode independent payloads per channel (R/G/B/A) into one image.
    channel_payloads example:
      {"R": {"enabled": True, "type": "text", "text": "hi"},
       "G": {"enabled": True, "type": "file", "file_data": b"..."},
       "B": {"enabled": False}, ...}
    """
    if not isinstance(image_bytes, bytes):
        raise TypeError(f"image_bytes must be bytes, got {type(image_bytes).__name__}")

    if not image_bytes:
        raise ValueError("Cannot encode into empty image data")

    if not isinstance(channel_payloads, dict):
        raise TypeError(f"channel_payloads must be a dict, got {type(channel_payloads).__name__}")

    output_format = normalize_output_format(output_format)
    if output_format == "jpeg":
        raise ValueError("JPEG output is disabled for advanced mode. Use PNG.")

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / filename

        try:
            with open(input_path, "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            raise IOError(f"Failed to write input image to temporary file: {str(e)}")

        output_path = tmp_dir / "encoded.png"

        try:
            if twitter_safe_preprocess:
                compress_image_before_encoding(input_path, output_path)
            else:
                _save_png(Image.open(input_path), output_path)
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

        try:
            base_img = Image.open(output_path).convert("RGBA")
        except Exception as e:
            raise ValueError(f"Failed to open and convert image to RGBA: {str(e)}")

        bits_by_channel: Dict[str, str] = {}
        for ch in ["R", "G", "B", "A"]:
            cfg = channel_payloads.get(ch) or {}
            if not cfg.get("enabled"):
                continue

            payload_type = cfg.get("type")
            try:
                if payload_type == "text":
                    text = cfg.get("text") or ""
                    bits_by_channel[ch] = _bits_from_text(text)
                elif payload_type == "file":
                    file_data = cfg.get("file_data")
                    if file_data is None:
                        raise ValueError(f"Missing file payload for channel {ch}")
                    try:
                        compressed = zlib.compress(file_data)
                    except Exception as e:
                        raise ValueError(f"Failed to compress file data for channel {ch}: {str(e)}")
                    bits_by_channel[ch] = _bits_from_zlib(compressed)
                else:
                    raise ValueError(f"Unknown payload type '{payload_type}' for channel {ch}")
            except Exception as e:
                raise ValueError(f"Failed to encode channel {ch}: {str(e)}")

        def embed_channel(pixels, width: int, height: int, channel: str, bits: str) -> None:
            try:
                idx = 0
                if channel == "R":
                    channel_idx = 0
                elif channel == "G":
                    channel_idx = 1
                elif channel == "B":
                    channel_idx = 2
                else:
                    channel_idx = 3

                for y in range(height):
                    for x in range(width):
                        if idx >= len(bits):
                            return
                        r, g, b, a = pixels[x, y]
                        val = (r, g, b, a)
                        channel_val = val[channel_idx]
                        channel_val = (channel_val & 0xFE) | int(bits[idx])
                        idx += 1
                        if channel_idx == 0:
                            val = (channel_val, g, b, a)
                        elif channel_idx == 1:
                            val = (r, channel_val, b, a)
                        elif channel_idx == 2:
                            val = (r, g, channel_val, a)
                        else:
                            val = (r, g, b, channel_val)
                        pixels[x, y] = val
            except Exception as e:
                raise IOError(f"Failed to embed data into {channel} channel: {str(e)}")

        max_iterations = 20
        iteration = 0

        while True:
            width, height = base_img.size
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {width}x{height}")

            capacity = width * height  # bits per channel
            for ch, bits in bits_by_channel.items():
                if len(bits) > capacity:
                    raise ValueError(
                        f"Payload too large for {ch} channel. "
                        f"Requires {len(bits)} bits but channel capacity is {capacity} bits. "
                        f"Try using a larger image."
                    )

            working = base_img.copy()
            try:
                pixels = working.load()
            except Exception as e:
                raise ValueError(f"Failed to load image pixels: {str(e)}")

            for ch, bits in bits_by_channel.items():
                if bits:
                    embed_channel(pixels, width, height, ch, bits)

            try:
                _save_png(working, output_path)
            except Exception as e:
                raise IOError(f"Failed to save encoded image: {str(e)}")

            if not twitter_safe_preprocess:
                break

            if output_path.stat().st_size <= TWITTER_MAX_BYTES:
                break

            iteration += 1
            if iteration >= max_iterations:
                raise ValueError(
                    f"Unable to keep encoded image below {TWITTER_MAX_BYTES // 1024}KB after "
                    f"{max_iterations} iterations. Try a smaller payload or larger image."
                )

            new_width = max(1, int(width * 0.9))
            new_height = max(1, int(height * 0.9))
            if new_width >= width and width > 1:
                new_width = width - 1
            if new_height >= height and height > 1:
                new_height = height - 1

            if new_width < 2 or new_height < 2:
                raise ValueError(
                    f"Image too small to compress further (size: {new_width}x{new_height}). "
                    f"Current file size: {output_path.stat().st_size} bytes exceeds limit."
                )

            base_img = base_img.resize((new_width, new_height), resample=Image.LANCZOS)

        output_name = f"encoded{output_format_extension(output_format)}"
        if output_format == "png":
            try:
                return output_name, output_path.read_bytes()
            except Exception as e:
                raise IOError(f"Failed to read encoded image file: {str(e)}")

        converted_path = tmp_dir / output_name
        try:
            _convert_output_image(output_path, output_format, converted_path)
            return output_name, converted_path.read_bytes()
        except Exception as e:
            raise IOError(f"Failed to convert encoded image to {output_format}: {str(e)}")
