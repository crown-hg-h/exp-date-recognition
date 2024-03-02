from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import numpy as np
import torch

class ExpirationDateDataset(torch.utils.data.Dataset):
    def __init__(self, hugging_face_dataset, transforms):
        self.transforms = transforms
        self.base_dataset = hugging_face_dataset.with_format("torch")

    def __getitem__(self, idx):
        # load images
        img_path = self.base_dataset[idx]["image_path"]
        img = read_image(img_path)

        # get bounding boxes
        boxes = self.base_dataset[idx]["bboxes_block"]

        # there is only one class
        labels = self.base_dataset[idx]['categories']

        image_id = idx

        target = {}
        if len(boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        else:
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        
        # Both needed for coco evaluation
        target["iscrowd"] = np.zeros(len(boxes))
        target["area"] = np.array([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes])

        if self.transforms is not None:
            augmented = self.transforms(
                image=np.array(img.permute(1, 2, 0)),
                bboxes=boxes.float(),
                categories=labels.float(),
            )
            # Pytorch expects the channel first
            img = augmented['image']
            # Recalculate bounding boxes
            target["boxes"] = tv_tensors.BoundingBoxes(augmented['bboxes'], format="XYXY", canvas_size=F.get_size(img))
            

        return img, target

    def __len__(self):
      return len(self.base_dataset)