
---

# Antidupe

Image deduplicator using CNN, Cosine Similarity, Image Hashing, Structural Similarity Index Measurement, and Euclidean Distance
## Installation

You can install Antidupe using pip:

```bash
pip install antidupe
```

## Usage

### Basic Usage

```python
from antidupe import Antidupe
from PIL import Image

# Initialize Antidupe
antidupe = Antidupe()

# Load images (as numpy arrays or PIL.Image objects)
image1 = Image.open("image1.jpg")
image2 = Image.open("image2.jpg")

# Check for duplicates
is_duplicate = antidupe.predict([image1, image2])

if is_duplicate:
    print("Duplicate images detected!")
else:
    print("Images are not duplicates.")
```

### Customizing Thresholds

You can customize the similarity thresholds for each technique during runtime or initialization:

```python
# Initialize Antidupe with custom thresholds
custom_thresholds = {
    'ih': 0.2,    # Image Hash
    'ssim': 0.2,  # SSIM
    'cs': 0.2,    # Cosine Similarity
    'cnn': 0.2,   # CNN
    'dedup': 0.85 # Mobilenet
}
antidupe = Antidupe(limits=custom_thresholds)

# Check for duplicates
is_duplicate = antidupe.predict([image1, image2])
```

### Debugging

You can enable debug mode to print debugging messages:

```python
# Initialize Antidupe with debug mode enabled
antidupe = Antidupe(debug=True)

# Check for duplicates
is_duplicate = antidupe.predict([image1, image2])
```

### Changing Limits During Runtime

You can change the similarity thresholds during runtime:

```python
# Set new limits during runtime
new_thresholds = {
    'ih': 0.1,
    'ssim': 0.1,
    'cs': 0.1,
    'cnn': 0.1,
    'dedup': 0.8
}
antidupe.set_limits(limits=new_thresholds)

# Check for duplicates
is_duplicate = antidupe.predict([image1, image2])
```

## Requirements

- Python 3.x
- SSIM PIL
- ImageDeDup
- NumPy
- MatPlotLib
- Pillow
- ImageHash
- Torch
- Efficientnet Pytorch
- TorchVision

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.