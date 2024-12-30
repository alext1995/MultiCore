import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np

class FocalLoss(nn.Module):
    """Implementation of Facal Loss"""
    def __init__(self, weight=None, gamma=2, reduction="mean", do_sigmoid=True):
        super(FocalLoss, self).__init__()
        self.weighted_ce = nn.BCELoss(weight=weight, reduction="none")
        self.ce = nn.BCELoss(reduction="none")
        self.gamma = gamma
        self.do_sigmoid = do_sigmoid
        self.reduction = reduction
        self.__name__ = f"FocalLoss_gamma_{gamma}"
        
    def forward(self, predicted_logits, target):
        """
        predicted: [batch_size, n_classes]
        target: [batch_size]
        """
        if self.do_sigmoid:
            predicted = torch.sigmoid(predicted_logits)
        else:
            predicted = predicted_logits
        pt = 1/torch.exp(self.ce(predicted,target))
        #shape: [batch_size]
        entropy_loss = self.weighted_ce(predicted, target)
        #shape: [batch_size]
        focal_loss = ((1-pt)**self.gamma)*entropy_loss
        #shape: [batch_size]
        if self.reduction =="none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        
def tversky(y_true, y_pred):
    smooth = 1
    
    y_true_pos = y_true.flatten()
    y_pred_pos = y_pred.flatten()
    true_pos = (y_true_pos * y_pred_pos).sum()
    false_neg = (y_true_pos * (1-y_pred_pos)).sum()
    false_pos = ((1-y_true_pos)*y_pred_pos).sum()
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_pred_logits, y_true):
    y_pred = torch.sigmoid(y_pred_logits)
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_pred_logits, y_true):
    y_pred = torch.sigmoid(y_pred_logits)
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return torch.pow((1-pt_1), gamma)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim()==4 and target.dim()==4:
        input_ = input[:,0]
        target_ = target[:,0]
    else:
        input_ = input
        target_ = target
    assert input_.dim() == 3 
    assert target_.dim() == 3 

    sum_dim = (-1, -2) if input_.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input_ * target_).sum(dim=sum_dim)
    sets_sum = input_.sum(dim=sum_dim) + target_.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(torch.sigmoid(input), target, reduce_batch_first=True)

def tversky(y_true, y_pred):
    smooth = 1
    
    y_true_pos = y_true.flatten()
    y_pred_pos = y_pred.flatten()
    true_pos = (y_true_pos * y_pred_pos).sum()
    false_neg = (y_true_pos * (1-y_pred_pos)).sum()
    false_pos = ((1-y_true_pos)*y_pred_pos).sum()
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, torch.sigmoid(y_pred))

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, torch.sigmoid(y_pred))
    gamma = 0.75
    return torch.pow((1-pt_1), gamma)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim()==4 and target.dim()==4:
        input_ = input[:,0]
        target_ = target[:,0]
    else:
        input_ = input
        target_ = target
    assert input_.dim() == 3 
    assert target_.dim() == 3 

    sum_dim = (-1, -2) if input_.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input_ * target_).sum(dim=sum_dim)
    sets_sum = input_.sum(dim=sum_dim) + target_.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(torch.sigmoid(input), target, reduce_batch_first=True)



class SSIMLoss(Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        """Computes the structural similarity (SSIM) index map between two images
        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.__name__ = f"SSIMLoss_kernel_size_{kernel_size}_sigma_{sigma}"
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x_logits: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:
        x = torch.sigmoid(x_logits)
        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        ssim_map = self._ssim(x, y)

        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map

    def _ssim(self, x: Tensor, y_: Tensor) -> Tensor:
        y = y_.float()
        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=1)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=1)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=1)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=1)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=1)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d
    
import torch
import numbers
import math
from torch import Tensor, einsum
from torch import nn
from scipy.ndimage import distance_transform_edt, morphological_gradient, distance_transform_cdt
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from torch.nn import functional as F

def simplex(t: Tensor, axis=1) -> bool:
    return True
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return True# simplex(t, axis) and sset(t, [0, 1])

def contour(x):
    '''
    Differenciable aproximation of contour extraction
    
    '''   
    min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour

def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
        contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

class ContourLoss():
    '''
    inputs shape  (batch, channel, height, width).
    calculate the contour loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    def __init__(self, do_sigmoid=True):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.do_sigmoid = "do_sigmoid"
        self.__name__= f"contour_loss_sigmoid_{int(do_sigmoid)}"
        
    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         pc = probs[:, self.idc, ...].type(torch.float32)
#         tc = target[:, self.idc, ...].type(torch.float32)
        if self.do_sigmoid:
            probs = torch.sigmoid(probs)
        pc = probs[:, :, ...].type(torch.float32)
        tc = target[:, :, ...].type(torch.float32)
        
        b, _, w, h = pc.shape
        cl_pred = contour(pc).sum(axis=(2,3))
        target_contour = contour(tc).sum(axis=(2,3))
        big_pen: Tensor = (cl_pred - target_contour) ** 2
        contour_loss = big_pen / (w * h)
    
        return contour_loss.mean(axis=0)

def compute_morphogradient(segmentation):
    res = np.zeros(segmentation.shape)
    print(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = morphological_gradient(posmask[0].astype(np.float32), size=(3,3))
    return res

class SurfaceLoss():
    '''
    Boundary loss implementation 
    Inputs:
    @probs: probability maps provded from the output of the network 
    @dc: distance maps computed when the dataset class is initialized
    outputs:
    @loss: boundary loss
    @description: 
    the loss finetunes the probability maps by the groundtruth distance map representations.
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.__name__="surface_loss"

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
#         assert simplex(probs)
#         assert not one_hot(dist_maps)
        
        probs = torch.sigmoid(probs)
        pc = probs[:, :, ...].type(torch.float32)
        dc = dist_maps[:, :, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss

def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res

class HDDTBinaryLoss():
    '''
    Hausdorf loss implementation for binary segmentation 
    '''
    def __init__(self, do_sigmoid=True):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.__name__=f"hddtbinary_do_sigmoid_{int(do_sigmoid)}"
        self.do_sigmoid = do_sigmoid
    def __call__(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        if self.do_sigmoid:
            net_output = torch.clone(net_output)
            
        net_output = net_output.clone().detach()
        
        pc = net_output[:, :, ...].type(torch.float32)
        gt = target[:, :, ...].type(torch.float32)
        with torch.no_grad():
            pc_dist = compute_edts_forhdloss(pc.detach().cpu().numpy()>0.5)
            gt_dist = compute_edts_forhdloss(gt.detach().cpu().cpu().numpy()>0.5)
        # print('pc_dist.shape: ', pc_dist.shape)
        
        pred_error = (gt - pc)**2
        dist = pc_dist**2 + gt_dist**2 # \alpha=2 in eq(8)

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        multipled = torch.einsum("bxyz,bxyz->bxyz", 
                                 pred_error.reshape(-1,1,pred_error.shape[1], pred_error.shape[2]), 
                                 dist.reshape(-1,1,dist.shape[1], dist.shape[2]))
        hd_loss = multipled.mean()

        return hd_loss

class soft_cldice_loss():
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        #self.idc: List[int] = kwargs["idc"]
        self.__name__="soft_cldice"
        
    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        probs = torch.sigmoid(probs)
        pc = probs[:, :, ...].type(torch.float32)
        tc = target[:, :, ...].type(torch.float32)
        b, _, w, h = pc.shape
        cl_pred = soft_skeletonize(pc)
        target_skeleton = soft_skeletonize(tc)
        big_pen: Tensor = (cl_pred - target_skeleton) ** 2
        contour_loss = big_pen / (w * h)
    
        return contour_loss.mean()

# def bce_loss(input, target):
#     return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0]).to(device))(input, target.float(), )
class BCE_cl_dice_loss:
    def __init__(self, weight=1):
        self.bce = nn.BCEWithLogitsLoss()
        self.cl_dice = soft_cldice_loss()
        self.weight = weight
        self.__name__ = self.bce.__name__ +f"_{weight}cl"
        
    def __call__(self, probs, targets):
        return self.bce(probs, targets) + self.weight*self.cl_dice(probs, targets)
    
class SSIM_cl_dice_loss:
    def __init__(self, weight=1):
        self.ssim = SSIMLoss(kernel_size=11, sigma=1.5) 
        self.cl_dice = soft_cldice_loss()
        self.weight = weight
        self.__name__ = self.ssim.__name__ +f"{weight}"
        
    def __call__(self, probs, targets):
        return self.ssim(probs, targets) + self.weight*self.cl_dice(probs, targets)
    
def get_name_critertion(criterion):
    try:
        return criterion.__name__
    except:
        if isinstance(criterion, MSELoss):
            return "mse_loss"

ssim_loss = SSIM_cl_dice_loss()