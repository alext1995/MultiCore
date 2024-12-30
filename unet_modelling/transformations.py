import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch
class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        # Generate a random decision
        if torch.rand(1).item() < self.p:
            image = F.hflip(image)  # Flip the image
            mask = F.hflip(mask)    # Flip the mask
        return image, mask
    
class RandomVerticalFlipPair:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        # Generate a random decision
        if torch.rand(1).item() < self.p:
            image = F.vflip(image)  # Flip the image
            mask = F.vflip(mask)    # Flip the mask
            
        return image, mask
    
class RandomRotationPair:
    def __init__(self, degrees, interpolation=F.InterpolationMode.NEAREST):
        """
        Args:
            degrees (float or tuple): Range of degrees to select from.
            interpolation (InterpolationMode, optional): Interpolation method for image.
        """
        self.degrees = degrees
        self.interpolation = interpolation

    def __call__(self, image, mask):
        # Generate random rotation degree
        angle = T.RandomRotation.get_params(self.degrees)
        # Apply rotation
        image = F.rotate(image, angle, interpolation=self.interpolation)
        mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
        return image, mask

class RandomAffinePair:
    def __init__(self, degrees, translate=None, scale=None, shear=None, interpolation=F.InterpolationMode.NEAREST):
        """
        Args:
            degrees (float or tuple): Range of degrees for rotation.
            translate (tuple, optional): Maximum absolute pixel translation (fraction of image size).
            scale (tuple, optional): Scaling factor range.
            shear (float or tuple, optional): Shear angle range.
            interpolation (InterpolationMode, optional): Interpolation method for image.
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation

    def __call__(self, image, mask):
        # Generate random affine parameters
        params = T.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, image.size()
        )
        # Apply affine transformation
        image = F.affine(image, *params, interpolation=self.interpolation)
        mask = F.affine(mask, *params, interpolation=F.InterpolationMode.NEAREST)
        return image, mask

class ColorJitterPair:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """
        Args:
            brightness (float or tuple): How much to jitter brightness. 
                                         0 means no change. 
                                         Can be a float or tuple (min, max).
            contrast (float or tuple): How much to jitter contrast.
                                        0 means no change.
            saturation (float or tuple): How much to jitter saturation.
                                          0 means no change.
            hue (float or tuple): How much to jitter hue.
                                  0 means no change.
        """
        self.color_jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, image, mask):
        # Apply color jitter to the image only
        image = self.color_jitter(image)
        return image, mask

class RandomNoisePair:
    def __init__(self, mean_range=(0, 0.1), std_range=(0.01, 0.1)):
        """
        Args:
            mean_range (tuple): Range of mean values for the noise (min, max).
            std_range (tuple): Range of standard deviation values for the noise (min, max).
        """
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, image, mask):
        # Randomly sample mean and std
        mean = torch.FloatTensor(1).uniform_(*self.mean_range).item()
        std = torch.FloatTensor(1).uniform_(*self.std_range).item()

        # Generate random noise with the sampled mean and std
        noise = torch.normal(mean=mean, std=std, size=image.shape, device=image.device)

        # Add noise to the image
        noisy_image = image + noise

        return noisy_image, mask