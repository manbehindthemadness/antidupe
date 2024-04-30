"""
Utility code.
"""
import imagehash
import numpy as np
from PIL import Image
import torch
from SSIM_PIL import compare_ssim
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt


def plot_images(image1, image2):
    """
    Plot the input images side by side for comparison.

    Args:
    - image1: First input image in PIL format.
    - image2: Second input image in PIL format.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1)
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    axes[1].imshow(image2)
    axes[1].set_title('Image 2')
    axes[1].axis('off')
    plt.show()


def resize_image(image1, image2, size: int = 100, channel_format: str = 'RGB', show: bool = False):
    """
    Resize two images to the specified size, maintaining aspect ratio and cropping to square if necessary.

    Args:
    - image1: First input image in PIL or NumPy format.
    - image2: Second input image in PIL or NumPy format.
    - size: Desired size for the square images.
    - channel_format: Desired channel format ('RGB' or 'L' for grayscale).

    Returns:
    - Resized and cropped versions of the input images.
    """
    if isinstance(image1, np.ndarray):
        image1 = Image.fromarray(image1)
    if isinstance(image2, np.ndarray):
        image2 = Image.fromarray(image2)

    aspect_ratio_1 = image1.width / image1.height
    aspect_ratio_2 = image2.width / image2.height

    if abs(aspect_ratio_1 - aspect_ratio_2) < 1e-6:
        image1 = resize_and_crop(image1, size, channel_format)
        image2 = resize_and_crop(image2, size, channel_format)
    else:
        min_size = min(image1.width, image1.height, image2.width, image2.height)
        image1 = resize_and_crop(image1, min_size, channel_format)
        image2 = resize_and_crop(image2, min_size, channel_format)

    if show:
        plot_images(image1, image2)

    return image1, image2


def resize_and_crop(image, size, channel_format):
    """
    Resize the input image to a square of the specified size, cropping and centering if necessary.

    Args:
    - image: Input image in PIL format.
    - size: Desired size for the square image.
    - channel_format: Desired channel format ('RGB' or 'L' for grayscale).

    Returns:
    - Resized and cropped image in PIL format.
    """
    image.thumbnail((size, size))

    if channel_format == 'RGB':
        new_image = Image.new("RGB", (size, size), (255, 255, 255))
    elif channel_format == 'L':
        new_image = Image.new("L", (size, size), 255)
    else:
        raise ValueError("Unsupported channel format. Please use 'RGB' or 'L'.")

    left = (size - image.width) // 2
    top = (size - image.height) // 2
    new_image.paste(image, (left, top))

    return new_image


def image_converter(image: [np.ndarray, Image.Image], size: int = 512, channel_format: str = 'RGB') -> Image.Image:
    """
    Converts a numpy array into an image object.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.size == (size, size) and image.mode == channel_format:
        return image
    image = image.resize((size, size))
    if image.mode != channel_format:
        image = image.convert(channel_format)
    return image


def euclidean_distance(image_1, image_2):
    """
    Calculates the Euclidean distance between two images.

    This works well for detecting identical images.
    """
    value = np.linalg.norm(np.array(image_1) - np.array(image_2))
    scaled_distance = value / (1 + value)
    return scaled_distance


def image_hash(image_1: Image.Image, image_2: Image.Image, hash_size: int = 8) -> float:
    """
    Calculates the image hash based on two images.

    This seems to be fairly effective for measuring images of variable differences.
    """
    image_1 = image_converter(image_1, size=hash_size, channel_format='L')
    image_2 = image_converter(image_2, size=hash_size, channel_format='L')
    hash1 = imagehash.average_hash(image_1)
    hash2 = imagehash.average_hash(image_2)
    value = hash1 - hash2
    normalized_difference = value / 64.0
    return normalized_difference


def ssim(image_1: Image.Image, image_2: Image.Image, gpu: bool = False) -> float:
    """
    Calculates the SSIM between two images.
    """
    value = compare_ssim(image_1, image_2, GPU=gpu)
    value = np.round(value, 8)
    return 1.0 - value


def cosine_similarity(image_1: Image.Image, image_2: Image.Image, device: str = 'cpu') -> float:
    """
    Calculates the cosine similarity between two images.
    """
    device = torch.device(device)
    image1_tensor = transforms.ToTensor()(image_1).reshape(1, -1).squeeze().to(device)
    image2_tensor = transforms.ToTensor()(image_2).reshape(1, -1).squeeze().to(device)

    cos = torch.nn.CosineSimilarity(dim=0)
    value = 1 - float(cos(image1_tensor.to(torch.device('cpu')), image2_tensor))
    value = np.clip(value * 10, 0, 1)
    return value


class CNN:
    """
    FInd differences using convolutional neural networks.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model.eval().to(self.device)

    def cnn(self, image_1: Image.Image, image_2: Image.Image) -> float:
        """
        Calculates the CNN similarity between two images.
        """
        image_1 = image_converter(image_1, size=224)
        image_2 = image_converter(image_2, size=224)
        transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image1_tensor = transform(image_1).to(self.device)
        image2_tensor = transform(image_2).to(self.device)
        features1 = self.model.extract_features(image1_tensor.unsqueeze(0))
        features2 = self.model.extract_features(image2_tensor.unsqueeze(0))
        value = round(np.linalg.norm(np.array(features1.detach()) - np.array(features2.detach())), 4)
        return value / 1000


def test():
    """
    This will test the various math functions
    """
    cnn = CNN()
    print('resources loaded')

    image_unique = image_converter(Image.open('images/unique_1.jpg'))
    image_duplicate_1 = image_converter(Image.open('images/Bead_necklace_1.jpg'))
    image_duplicate_2 = image_converter(Image.open('images/Bead_necklace_2.jpg'))

    e_d_u = euclidean_distance(image_unique, image_duplicate_1)
    e_d_d = euclidean_distance(image_duplicate_1, image_duplicate_2)
    e_d_d_d = euclidean_distance(image_duplicate_1, image_duplicate_1)
    report = (f"euclidean_distance unique: {e_d_u}\neuclidean_distance duplicate size difference: {e_d_d}\n"
              f"euclidean_distance duplicate identical: {e_d_d_d}\n")

    i_h_u = image_hash(image_unique, image_duplicate_1)
    i_h_d = image_hash(image_duplicate_1, image_duplicate_2)
    i_h_d_d = image_hash(image_duplicate_1, image_duplicate_1)
    report += (f"image_hash unique: {i_h_u}\nimage_hash duplicate size_difference: {i_h_d}\n"
               f"image_hash duplicate identical: {i_h_d_d}\n")

    s_i_u = ssim(image_unique, image_duplicate_1)
    s_i_d = ssim(image_duplicate_1, image_duplicate_2)
    s_i_d_d = ssim(image_duplicate_1, image_duplicate_1)
    report += (f"ssim unique: {s_i_u}\nssim duplicate size_difference: {s_i_d}\n"
               f"ssim duplicate identical: {s_i_d_d}\n")

    c_s_u = cosine_similarity(image_unique, image_duplicate_1)
    c_s_d = cosine_similarity(image_duplicate_1, image_duplicate_2)
    c_s_d_d = cosine_similarity(image_duplicate_1, image_duplicate_1)
    report += (f"cosine_similarity unique: {c_s_u}\ncosine_similarity duplicate size_difference: {c_s_d}\n"
               f"cosine_similarity identical: {c_s_d_d}\n")

    c_n_u = cnn.cnn(image_unique, image_duplicate_1)
    c_n_d = cnn.cnn(image_duplicate_1, image_duplicate_2)
    c_n_d_d = cnn.cnn(image_duplicate_1, image_duplicate_1)
    report += f"CNN unique: {c_n_u}\nCNN duplicate size_difference: {c_n_d}\nCNN identical: {c_n_d_d}"
    print(report)
