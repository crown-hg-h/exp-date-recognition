import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE_PATH = "src/exp_date_recognition/detection/model"
PATH = f"{BASE_PATH}/checkpoints/resnet50_exp_date.pt"
TEMPLATE_PATH = BASE_PATH + "/checkpoints/resnet50_exp_date_{version}.pt"

def get_model_instance(num_classes, load_fine_tunned=False, fine_tunned_ver="v3", template_path=TEMPLATE_PATH):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if load_fine_tunned:
        checkpoint = torch.load(template_path.format(version=fine_tunned_ver))
        model.load_state_dict(checkpoint['model_state_dict'])

    return model

def save_model(model, optimizer, epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, PATH)