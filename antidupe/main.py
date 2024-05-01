"""
Main library.
"""
from PIL import Image
try:
    from utilities import (
        euclidean_distance,
        image_hash,
        ssim,
        cosine_similarity,
        CNN,
        resize_image,
    )
except ImportError:
    from .utilities import (
        euclidean_distance,
        image_hash,
        ssim,
        cosine_similarity,
        CNN,
        resize_image,
    )


DEFAULTS = {
    'ih': 0.1,
    'ssim': 0.15,
    'cs': 0.1,
    'cnn': 0.15
}


class Antidupe:
    """
    Discover duplicate images.
    """
    def __init__(self, limits: dict = DEFAULTS, device: str = 'cpu', debug: bool = False, show_plots: bool = False):  # noqa
        self.show_plots = show_plots
        self.device = device
        self.limits = limits
        self.debug = debug
        self.cnn = CNN(device=device)

    def d_print(self, *args, **kwargs):
        """
        Debug messanger.
        """
        if self.debug:
            print(*args, **kwargs)
        return self

    def set_limits(self, limits: dict = DEFAULTS):  # noqa
        """
        Allows us to change limis during run time.
        """
        self.limits = limits
        return self

    def predict(self, images: [list, tuple], size: int = 512) -> bool:
        """
        Lets measure our images.
        """
        switch = False
        if self.device != 'cpu':
            switch = True
        im_1, im_2 = resize_image(*images, size, show=self.show_plots)
        ed = euclidean_distance(im_1, im_2)
        self.d_print(f'euclidean distance detected: {ed}')
        if ed == 0.0:
            self.d_print('euclidean distance found duplicate')
            return True
        else:
            ih = image_hash(im_1, im_2)
            self.d_print(f'image hash detected: {ih}')
            if ih < self.limits['ih']:
                self.d_print('image hash found duplicate')
                return True
            ss = ssim(im_1, im_2, switch)
            self.d_print(f'ssim detected: {ss}')
            if ss < self.limits['ssim']:
                self.d_print('ssim found duplicate')
                return True
            cs = cosine_similarity(im_1, im_2, self.device)
            self.d_print(f'cosine similarity detected: {cs}')
            if cs < self.limits['cs']:
                self.d_print('cosine similarity found duplicate')
                return True
            cnn = self.cnn.cnn(im_1, im_2)
            self.d_print(f'cnn detected: {cnn}')
            if cnn < self.limits['cnn']:
                self.d_print('cnn found duplicate')
                return True
        return False

    def test(self, images: [list, tuple, None] = None):
        """
        Tests the prediction logic.
        """
        self.debug = True
        if not images:
            im_1 = Image.open('images/unique_1.jpg')
            im_2 = Image.open('images/Bead_necklace_1.jpg')
            im_3 = Image.open('images/Bead_necklace_2.jpg')
        else:
            im_1, im_2, im_3 = images
        line = '------'
        print('testing unique')
        print(f"{line}duplicate? {self.predict((im_1, im_2))}")
        print('testing duplicates resized')
        print(f"{line}duplicate? {self.predict((im_2, im_3))}")
        print('testing identical duplicates')
        print(f"{line}duplicate? {self.predict((im_2, im_2))}")
        self.debug = False
