import torch
import numpy as np
import cv2
from torchvision import transforms as T

def get_plate(img, model, size_transform=224, threshold_min=0.5, threshold_max=0.9, device=None, margin=10):
    """
    Extract license plate region from an image using the trained UNET model.
    
    Args:
        image_path (str): Path to the input image
        model: Trained UNET model
        threshold_min (float): Minimum threshold for mask binarization
        threshold_max (float): Maximum threshold for mask binarization
    
    Returns:
        tuple: (plate_tensor, original_image, mask) - extracted plate as tensor, original image, and mask
    """
    # Load and transform the image
    # image_load = Image.open(image_path)
    transform = T.Compose([T.Resize([size_transform, size_transform]), T.ToTensor()])  # Using same size as training
    
    image_tensor = transform(img)
    image_batch = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Predict the mask using the model
    model = model.to(device=device)
    image_batch = image_batch.to(device=device)
    
    with torch.no_grad():
        scores = model(image_batch)
        preds = scores[:, 0]
        
        # Move back to CPU for processing
        image_batch = image_batch.cpu()
        preds = preds.cpu()
    
    # Convert to numpy for image processing
    original_img = image_batch[0].permute(1, 2, 0).numpy()
    mask = preds[0].numpy()
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask min/max values: {mask.min():.3f}/{mask.max():.3f}")
    
    # Create binary mask
    binary_mask = cv2.inRange(mask, threshold_min, threshold_max)
    
    # Find bounding box coordinates
    y_indices, x_indices = np.nonzero(binary_mask)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        print("No license plate detected in the image")
        return None, original_img, mask
    
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    print(f'Bounding box - xmin: {x_min}, xmax: {x_max}, ymin: {y_min}, ymax: {y_max}')
    
    # bounding the image
    img_height, img_width = original_img.shape[:2]
    x_min_with_margin = max(0, x_min - margin)
    x_max_with_margin = min(img_width - 1, x_max + margin)
    y_min_with_margin = max(0, y_min - margin)
    y_max_with_margin = min(img_height - 1, y_max + margin)

    # Extract the plate region
    plate_region = original_img[y_min_with_margin:y_max_with_margin + 1, 
                               x_min_with_margin:x_max_with_margin + 1, :]
    
    # Convert back to tensor format
    plate_tensor = torch.from_numpy(plate_region).permute(2, 0, 1).unsqueeze(0)
    
    return plate_tensor, original_img, mask


# enhance image of the plate
def enhance_plate_image(plate_image, kernel_size=5, scale_factor=3):
    """
    Apply image enhancement techniques to improve OCR results.
    """
    if torch.is_tensor(plate_image):
        plate_image_copy = plate_image[0].permute(1, 2, 0).numpy()
    else:
        plate_image_copy = plate_image.copy()

    # resize to larger size (upscaling for better OCR)
    height, width = plate_image_copy.shape[:2]
    new_height, new_width = height*scale_factor, width*scale_factor
    upscaled = cv2.resize(plate_image_copy, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # convert to gray scale
    if len(upscaled.shape) == 3:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
    else:
        gray = upscaled

    # apply median blur to reduce noise
    enhanced_image = cv2.medianBlur(gray, kernel_size)

    # sharpen the image with the box filter 
    box_filter = (-1)*np.ones((3,3),np.float32)
    box_filter[1,1] = 9
    box_filter = box_filter / np.sum(box_filter)
    sharpened = cv2.filter2D(enhanced_image, -1, box_filter)

    # apply bilateral filter for further smoothing
    d = 9             # Diameter of each pixel neighborhood
    sigma_color = 75  # Filter sigma in the color space
    sigma_space = 75  # Filter sigma in the coordinate space
    bilateral = cv2.bilateralFilter(sharpened, d, sigma_color, sigma_space)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)

    return np.array(enhanced_image)
