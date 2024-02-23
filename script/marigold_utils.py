import torch
import torch.nn as nn

import matplotlib
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
from tqdm import tqdm
from scipy.optimize import minimize



def calculate_metric_percase(pred, gt):
    from medpy import metric

    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    import SimpleITK as sitk

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def inter_distances(arrays):
    """
    To calculate the distance between each two depth maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(arrays.shape[0])):
        arr1 = arrays[i:i+1]
        arr2 = arrays[j:j+1]
        distances.append(arr1 - arr2)
    if isinstance(arrays, torch.Tensor):
        dist = torch.concatenate(distances, dim=0)
    elif isinstance(arrays, np.ndarray):
        dist = np.concatenate(distances, axis=0)
    return dist
    

def ensemble_depths(input_images, regularizer_strength=0.02, max_iter=2, tol=1e-3, reduction='median', max_res=None, disp=False, device='cuda'):
    """ 
    To ensemble multiple affine-invariant depth images (up to scale and shift),
        by aligning estimating the sacle and shift
    """
    original_input = input_images.copy()
    n_img = input_images.shape[0]
    ori_shape = input_images.shape
            
    if max_res is not None:
        scale_factor = np.min(max_res / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')
            input_images = downscaler(torch.from_numpy(input_images)).numpy()

    # init guess
    _min = np.min(input_images.reshape((n_img, -1)), axis=1)
    _max = np.max(input_images.reshape((n_img, -1)), axis=1)
    s_init = 1.0 / (_max - _min).reshape((-1, 1, 1))
    t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1))
    
    x = np.concatenate([s_init, t_init]).reshape(-1)
    input_images = torch.from_numpy(input_images).to(device)
    
    # objective function
    def closure(x):
        l = len(x)
        s = x[:int(l/2)]
        t = x[int(l/2):]
        s = torch.from_numpy(s).to(device)
        t = torch.from_numpy(t).to(device)
        
        transformed_arrays = input_images * s.view((-1, 1, 1)) + t.view((-1, 1, 1))
        dists = inter_distances(transformed_arrays)
        sqrt_dist = torch.sqrt(torch.mean(dists**2))
        
        if 'mean' == reduction:
            pred = torch.mean(transformed_arrays, dim=0)
        elif 'median' == reduction:
            pred = torch.median(transformed_arrays, dim=0).values
        else:
            raise ValueError
        
        near_err = torch.sqrt((0 - torch.min(pred))**2)
        far_err = torch.sqrt((1 - torch.max(pred))**2)
        
        err = sqrt_dist + (near_err + far_err) * regularizer_strength
        err = err.detach().cpu().numpy()
        return err
    
    res = minimize(closure, x, method='BFGS', tol=tol, options={'maxiter': max_iter, 'disp': disp})
    x = res.x
    l = len(x)
    s = x[:int(l/2)]
    t = x[int(l/2):]
    
    # Prediction
    transformed_arrays = original_input * s[:, np.newaxis, np.newaxis] + t[:, np.newaxis, np.newaxis]
    if 'mean' == reduction:
        aligned_images = np.mean(transformed_arrays, axis=0)
        std = np.std(transformed_arrays, axis=0)
        uncertainty = std
    elif 'median' == reduction:
        aligned_images = np.median(transformed_arrays, axis=0)
        # MAD (median absolute deviation) as uncertainty indicator
        abs_dev = np.abs(transformed_arrays - aligned_images)
        mad = np.median(abs_dev, axis=0)
        uncertainty = mad
    else:
        raise ValueError
    
    # Scale and shift to [0, 1]
    _min = np.min(aligned_images)
    _max = np.max(aligned_images)
    aligned_images = (aligned_images - _min) / (_max - _min)
    uncertainty /= (_max - _min)
    
    return aligned_images, uncertainty


def resize_max_res(img: Image.Image, max_edge_resolution):
    original_width, original_height = img.size
    downscale_factor = min(max_edge_resolution / original_width, max_edge_resolution / original_height)
    
    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)
    
    resized_img = img.resize((new_width, new_height))
    return resized_img


def Marigold_estimation(d_model, input_image):
    n_repeat = 10
    n_denoise = 10

    image = resize_max_res(
        input_image, max_edge_resolution=768
    )
    
    # Convert the image to RGB, to 1.remove the alpha channel 2.convert B&W to 3-channel
    image = image.convert('RGB')
    
    image = np.asarray(image)

    # Normalize rgb values
    rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
    rgb_norm = rgb / 255.0
    rgb_norm = torch.from_numpy(rgb_norm).unsqueeze(0).float()
    rgb_norm = rgb_norm.to('cuda')
    assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

    # Predict depth maps
    d_model.unet.eval()
    depth_pred_ls = []
    with torch.no_grad():
        for i_rep in tqdm(range(n_repeat), desc="multiple inference", leave=False):
            depth_pred_raw = d_model.forward(
                rgb_norm, num_inference_steps=n_denoise, init_depth_latent=None
            )
            # clip prediction
            depth_pred_raw = torch.clip(depth_pred_raw, -1.0, 1.0)
            depth_pred_ls.append(depth_pred_raw.detach().cpu().numpy().copy())

        depth_preds = np.concatenate(depth_pred_ls, axis=0).squeeze()

        # Test-time ensembling
        if n_repeat > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds,
                regularizer_strength=0.02,
                max_iter=5,
                tol=0.001,
                reduction='median',
                max_res=None,
                device='cuda',
            )
        else:
            depth_pred = depth_preds

    # Resize back to original resolution
    pred_img = Image.fromarray(depth_pred)
    pred_img = pred_img.resize(input_image.size)
    depth_pred = np.asarray(pred_img)

    return depth_pred


def colorize(value, vmin=None, vmax=None, cmap='jet', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],98) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img
