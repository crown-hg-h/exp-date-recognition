from src.common.engine import train_one_epoch, evaluate
from src.exp_date_recognition.detection.model.resnet50 import save_model
import torch


def train(
        model,
        data_loader_train,
        data_loader_test,
        optimizer,
        num_epochs,
        lr_scheduler, 
        save_model_callback=save_model):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model to the right device
    model.to(device)
    # let's train it
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # save the model
        save_model_callback(model, optimizer, epoch)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
