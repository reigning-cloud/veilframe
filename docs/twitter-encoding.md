# Twitter-Compatible Encoding Logic

The encoder preserves LSB data on Twitter by pre-compressing and, if needed, downscaling the cover image so it stays under ~900 KB (Twitter's recompression threshold). This is the exact segment used:

```python
from pathlib import Path
from PIL import Image
import os

def compress_image_before_encoding(image_path: Path, output_image_path: Path) -> None:
    png_path = convert_to_png(image_path)
    img = Image.open(png_path)
    img.save(output_image_path, optimize=True, format="PNG")

    # Compress until we are under ~900 KB to survive Twitter's pipeline.
    while os.path.getsize(output_image_path) > 900 * 1024:
        img = Image.open(output_image_path)
        img = img.resize((max(1, img.width // 2), max(1, img.height // 2)))
        img.save(output_image_path, optimize=True, format="PNG")
```

This step is run before any payload is embedded, ensuring Twitter does not recompress away the LSB-encoded data.
