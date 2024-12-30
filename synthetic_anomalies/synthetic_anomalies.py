import numpy as np
import cv2
from polygenerator import (
    random_polygon,
)
from torchvision import transforms
import torch

def make_centers(fractions, image_width):
    midpoints = np.linspace(0, image_width, fractions)[:-1]//1
    xx, yy = np.meshgrid(midpoints, midpoints)
    
    points = []
    for x_row, y_row in zip(xx, yy):
        for x, y in zip(x_row, y_row):
            points.append((x, y))
    return points

def create_poly(num):
    coords = random_polygon(num)
    return [((a-0.5)*256, (b-0.5)*256) for a, b in coords]

def make_polygon_points(center_point, num_points, scale, stretch_y, stretch_x):
    y, x   = center_point
    coords = create_poly(num_points)
    coords = [(int(y+a*scale*stretch_y), int(x+b*scale*stretch_x)) for a, b in coords]
    return coords
    
def create_mask_from_points(points):
    src_mask = np.zeros((256, 256), dtype='int8')
    return cv2.fillPoly(src_mask, [np.array([points])], (255, 255, 255)).astype(np.uint8)

def add_smoothing(mask, kernel_size):
    return 255*cv2.blur(mask/255, (kernel_size, kernel_size))

def add_blur(mask, kernel_size=5):
    """
    Applies Gaussian smoothing to a mask.

    Parameters:
    - mask (numpy.ndarray): The input binary mask of dtype uint8.
    - kernel_size (int): Size of the Gaussian kernel. Must be an odd number. Default is 5.

    Returns:
    - numpy.ndarray: The smoothed mask.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1 
    smoothed_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return smoothed_mask

class StratifiedMaskHolder:
    '''
    This delivers the masks
    '''
    def __init__(self, 
                 weight=0.6, 
                 no_mask_probability=0.3,
                 weight_type="gaussian",
                 fractions=20, 
                 image_width=256, 
                 smoothing=True, 
                 blur=True,
                 multi_mask_factor=0,
                 weight_min=None, 
                 weight_max=None):
        self.center_points = make_centers(fractions   = fractions, 
                                          image_width = image_width)
        self.smoothing = smoothing
        self.blur      = blur
        self.weight    = weight
        self.served_mask_trace = []
        self.no_mask_probability = no_mask_probability
        self.weight_type = weight_type
        self.multi_mask_factor = multi_mask_factor
        self.images = []
        self.pertubed_images = []
        
        if weight_min and weight_max:
            self.weight_min = weight_min
            self.weight_max = weight_max 
            self.weight_variation = True
        else:
            self.weight_variation = False
            
    def serve_mask(self, skip_regular=False):
        if not skip_regular and np.random.rand()<self.no_mask_probability:
            return np.zeros((256, 256))
            
        mask_points = make_polygon_points(center_point = self.center_points[np.random.choice(len(self.center_points))], 
                                           num_points   = np.random.randint(3, 10), 
                                           scale        = np.random.rand(), 
                                           stretch_y    = np.random.rand(), 
                                           stretch_x    = np.random.rand())
        mask = create_mask_from_points(mask_points)
        
        if np.random.rand()>0.5 and self.smoothing:
            mask = add_smoothing(mask, kernel_size=np.random.randint(2, 30))
        
        if np.random.rand()>0.5 and self.blur:
            mask = add_blur(mask, kernel_size=np.random.randint(2, 10)*2 - 1)
        
        return mask>mask.mean()
    
    def add_noise_to_image(self, image):
        mask = self.serve_mask()
        if not self.multi_mask_factor==0:
            val = np.random.rand()
            while val<self.multi_mask_factor:
                new_mask = self.serve_mask(skip_regular=True)
                mask = mask+new_mask
                val = np.random.rand()
            
        self.served_mask_trace.append(mask)
        image = transforms.ToTensor()(image)
        image = transforms.Resize((256, 256), antialias=True)(image)
                
        if self.weight_variation:
            weight = self.weight_min + np.random.rand()*(self.weight_max - self.weight_min)
        else:
            weight = self.weight
            
        if self.weight_type=="uniform":
            amended_image = image[:,...] + weight * np.random.rand(256, 256) * mask.astype(float)
        elif self.weight_type=="gaussian":
            amended_image = image[:,...] + weight * np.random.normal(size=(256, 256)) * mask.astype(float)
        elif self.weight_type=="fixed":
            amended_image = image[:,...] + weight * np.ones((256, 256)) * mask.astype(float)

        # self.pertubed_images.append(amended_image)
        return torch.tensor(amended_image).float(), torch.tensor(mask).float()