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


def resize_image(image, size: int = 100, channel_format: str = 'RGB') -> Image.Image:
    """
    Resize the input image to 512x512 RGB if it's not already in the correct format.

    Args:
    - image: Input image in PIL or NumPy format.

    Returns:
    - Resized image in PIL format.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.size == (size, size) and image.mode == channel_format:
        return image
    image = image.resize((size, size))
    if image.mode != channel_format:
        image = image.convert(channel_format)
    return image


def image_converter(image: [np.ndarray, Image.Image], size: int = 512, channel_format: str = 'RGB') -> Image.Image:
    """
    Converts a numpy array into an image object.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = resize_image(image, size, channel_format)
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
    return value


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
