import random
import numpy as np
import torch
import json
from pathlib import Path
from skimage import io


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)


def write_json(dict_to_write, output_file):
    """
    Outputs a given dictionary as a JSON file with indents.

    Args:
        dict_to_write (dict): Input dictionary to output.
        output_file (str): File path to write the dictionary.

    Returns:
        None
    """
    with Path(output_file).open('w') as file:
        json.dump(dict_to_write, file, indent=4)


def load_json(json_file):
    """
    Loads a JSON file as a dictionary and return it.

    Args:
        json_file (str): Input JSON file to read.

    Returns:
        dict: Dictionary loaded from the JSON file.
    """
    with Path(json_file).open('r') as file:
        return json.load(file)


def save_tensor_image(x, output_file):
    """
    Saves an input image tensor as some numpy array, useful for tests.

    Args:
        x (Tensor): A 3D tensor image of shape (3, H, W).
        output_file (str): The output image file to save the tensor.

    Returns:
        None
    """
    folder = Path(output_file).parent
    Path(folder).mkdir(exist_ok=True)

    x = x.permute(1, 2, 0).numpy()
    io.imsave(output_file, x)


def load_images(n=1, size=32):
    """
    Load n image tensors with some fake labels.

    Args:
        n (int): Number of random images to load.
        size (int): Spatial size of random image.

    Returns:
        Tensor: Random images of shape (n, 3, size, size) and 0-valued labels.
    """
    images = torch.randn(n, 3, size, size)
    labels = torch.from_numpy(np.array([0 * n]))

    return images, labels