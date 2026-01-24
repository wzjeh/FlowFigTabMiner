import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import argparse
import os
import time
import copy

def train_model(data_dir, output_path, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Data Augmentation & Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # Optional: depends on chemistry/text orientation validity
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load Data
    # Expected structure:
    # data_dir/train/Text/xxx.png
    # data_dir/train/Structure/xxx.png
    # data_dir/val/Text/xxx.png ...
    
    # Auto-split if only 'train' exists or just a flat folder? 
    # For simplicity, standard ImageFolder structure expected.
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"Classes: {class_names}")
    
    # Load Model
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    # Binary classification: Text vs Structure
    model_ft.fc = nn.Linear(num_ftrs, 2) 
    
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Training Loop
    since = time.time()
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()
            else:
                model_ft.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Save
    model_ft.load_state_dict(best_model_wts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model_ft.state_dict(), output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to data directory (containing train/val subfolders)")
    parser.add_argument("--output", default="models/cell_classifier_resnet.pt")
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output)
