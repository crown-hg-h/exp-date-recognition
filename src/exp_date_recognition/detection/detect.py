from torchvision.utils import draw_bounding_boxes
from torchvision.ops import nms
from .transformations import get_detect_transform
import torch
import numpy as np

# Dictionary mapping category indices to colors
COLORS_DICT = {
    0: 'cyan',
    1: 'green',
    2: 'red',
    3: 'magenta'
}

def detect(model, image, categories, colors_dict=COLORS_DICT, iou_threshold=0.2):
    """
    Detect objects in an image using a pre-trained model.

    Args:
        model: Pytorch model
        image (PIL.Image): The input image.
        categories (Categories): Object containing methods for converting category indices to strings.
        colors_dict (dict): Dictionary mapping category indices to colors.
        iou_threshold (float): IoU (Intersection over Union) threshold for non-maximum suppression (NMS).

    Returns:
        torch.Tensor: The input image with bounding boxes drawn around detected objects.
        list of torch.Tensor: List of bounding boxes for detected objects.
        list of str: List of labels for detected objects.
    """

    transform = get_detect_transform()
    
    # Determine the device for model inference
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load pre-trained model
    model.to(device)
    model.eval()
    
    
    with torch.no_grad():
        x = transform(image=np.array(image))
        img = x['image'].to(device)
        
        # Perform inference with the model
        predictions = model([img, ])
        pred = predictions[0]

    # Normalize the image
    image = (255.0 * (img - img.min()) / (img.max() - img.min())).to(torch.uint8)

    pred_labels = []
    pred_boxes = []
    colors = []
    scores = []

    # Extract predictions from the model output
    for label, score, box in zip(pred["labels"], pred["scores"], pred["boxes"]):
        pred_labels.append(f"{categories.int2str(label.item())}: {score:.3f}")
        pred_boxes.append(box)
        scores.append(score)
        colors.append(colors_dict[label.item()])

    if pred_boxes:
        # Convert pred_boxes list to a PyTorch tensor
        pred_boxes = torch.stack(pred_boxes)
        scores = torch.stack(scores)
        
        # Apply Non-Maximum Suppression (NMS)
        keep = nms(pred_boxes, scores, iou_threshold)
        
        # Keep only the predictions with high confidence after NMS
        pred_labels_keep = [pred_labels[i] for i in keep]
        colors = [colors[i] for i in keep]
        
        image = draw_bounding_boxes(image, pred_boxes[keep], pred_labels_keep, colors=colors)

        return image, pred_boxes[keep], pred_labels_keep
        
    return image, [], []