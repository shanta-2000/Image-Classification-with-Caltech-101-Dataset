# Image-Classification-with-Caltech-101-Dataset
Image Classification with Caltech-101 Dataset and Explainability using Grad-CAM 

## Objective
To build, train, and evaluate a Convolutional Neural Network (CNN) for image
classification using the Caltech-101 dataset. Students will also learn to apply
Explainable AI (XAI) techniques, specifically Grad-CAM, to visualize model
decision-making. 

## Prerequisites
• Basic knowledge of Python programming.<br>
• Familiarity with machine learning and deep learning concepts.<br>
• Understanding of Convolutional Neural Networks (CNNs).<br>
• Basic knowledge of Explainable AI (XAI).<br>

## Part 1: Introduction to Caltech-101 Dataset
Dataset Overview<br>
• Caltech-101 contains images from 101 object categories and 1 background
category.<br>
• Images per category range from 40 to 800, with a total of approximately 9,146
images.<br>
• Images are diverse in size and can be resized for consistency.
Task<br>
• Classify images into one of the 101 categories using a CNN.<br>

• Visualize model decision-making using Grad-CAM.

## Part 2: Dataset Loading and Preprocessing
Download Dataset<br>
• Download the Caltech-101 dataset from Caltech-101 Dataset.<br>
Load Dataset<br>
Use PyTorch to load and preprocess the dataset. Example in PyTorch:
### python
from torchvision import datasets, transforms<br>
transform = transforms.Compose([<br>
transforms.Resize((128, 128)),<br>
transforms.RandomHorizontalFlip(),<br>
transforms.RandomVerticalFlip(),<br>
transforms.RandomRotation(30),<br>
transforms.ColorJitter(brightness=0.4, contrast=0.4,
saturation=0.4, hue=0.1),<br>
transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
scale=(0.9, 1.1)),<br>
transforms.RandomGrayscale(p=0.2),<br>
transforms.ToTensor(),<br>
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,
0.224, 0.225])<br>
])<br>
dataset = datasets.ImageFolder(root='path_to_caltech101',
transform=transform)<br>
### Split Dataset
Split into training, validation, and test sets.
### python

from torch.utils.data import random_split<br>
train_size = int(0.8 * len(dataset))<br>
val_size = int(0.1 * len(dataset))<br>
test_size = len(dataset) - train_size - val_size<br>
train_data, val_data, test_data = random_split(dataset, [train_size,
val_size, test_size])<br>
Data Loaders<br>
Create data loaders for efficient batching.<br>
### python
from torch.utils.data import DataLoader<br>
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)<br>
val_loader = DataLoader(val_data, batch_size=32)<br>
test_loader = DataLoader(test_data, batch_size=32)<br>

## Part 3: Using Pre-trained Models
VGG19 Model<br>
### python
from torchvision.models import vgg19<br>
model = vgg19(pretrained=True)<br>
model.classifier[6] = nn.Linear(4096, 101) # Adjust the final layer
for 101 classes
ResNet50 Model
### python
from torchvision.models import resnet50<br>
model = resnet50(pretrained=True)<br>

model.fc = nn.Linear(2048, 101) # Adjust the final layer for 101
classes
EfficientNet (EfficientNet-B0)
### python
from torchvision.models import efficientnet_b0<br>
model = efficientnet_b0(pretrained=True)<br>
model.classifier[1] = nn.Linear(1280, 101) # Adjust the final layer
for 101 classes

## Part 4: Training the Model
Define Loss and Optimizer
### python
import torch.optim as optim<br>
criterion = nn.CrossEntropyLoss()<br>
optimizer = optim.Adam(model.parameters(), lr=0.001)<br>
Train the Model
### python
for epoch in range(10):<br>
model.train()<br>
for images, labels in train_loader:<br>
optimizer.zero_grad()<br>
outputs = model(images)<br>
loss = criterion(outputs, labels)<br>
loss.backward()<br>
optimizer.step()<br>
Validate the Model
### python

model.eval()<br>
with torch.no_grad():<br>
for images, labels in val_loader:<br>
outputs = model(images)<br>
# Compute validation metrics
Parameter Tuning with Grid Search
### python
from sklearn.model_selection import ParameterGrid<br>
param_grid = {<br>
'lr': [0.1, 0.01, 0.001],<br>
'batch_size': [16, 32, 64]
}<br>
best_params = None<br>
best_accuracy = 0<br>
for params in ParameterGrid(param_grid):<br>
optimizer = optim.Adam(model.parameters(), lr=params['lr'])<br>
train_loader = DataLoader(train_data,<br>
batch_size=params['batch_size'], shuffle=True)<br>
# Perform training and validation here
# Compare and store the best parameters based on validation accuracy
print(f"Best Params: {best_params}")

## Part 5: Evaluating the Model
Evaluate on Test Data
### python
model.eval()<br>

with torch.no_grad():<br>
for images, labels in test_loader:<br>
outputs = model(images)<br>
# Compute test metrics
Confusion Matrix
### python
from sklearn.metrics import confusion_matrix<br>
y_pred = []<br>
y_true = []<br>
with torch.no_grad():<br>
for images, labels in test_loader:<br>
outputs = model(images)<br>
_, preds = torch.max(outputs, 1)<br>
y_pred.extend(preds.numpy())<br>
y_true.extend(labels.numpy())<br>
cm = confusion_matrix(y_true, y_pred)<br>
print(cm)<br>
Classification Report
### python
from sklearn.metrics import classification_report<br>
print(classification_report(y_true, y_pred, target_names=class_names))<br>
##### Top-k Accuracy
### python
def top_k_accuracy(output, target, k=5):<br>
with torch.no_grad():<br>
max_k_preds = torch.topk(output, k, dim=1).indices<br>
correct = max_k_preds.eq(target.view(-1,<br>
1).expand_as(max_k_preds))<br>
return correct.any(dim=1).float().mean().item()<br>

##### Per-Class Accuracy
### python
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)<br>
for i, acc in enumerate(per_class_accuracy):<br>
print(f"Class {class_names[i]} Accuracy: {acc:.2f}")<br>
##### t-SNE Visualization
### python
from sklearn.manifold import TSNE<br>
import matplotlib.pyplot as plt<br>
features = []<br>
labels_list = []<br>
model.eval()<br>
with torch.no_grad():<br>
for images, labels in test_loader:<br>
output = model(images)<br>
features.append(output)<br>
labels_list.append(labels)<br>
features = torch.cat(features).numpy()<br>
labels_list = torch.cat(labels_list).numpy()<br>
tsne = TSNE(n_components=2, random_state=42)<br>
reduced_features = tsne.fit_transform(features)<br>
plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
c=labels_list, cmap='tab10')<br>
plt.colorbar()<br>
plt.show()<br>

## Part 6: Explainable AI (XAI) with Grad-CAM

Install Grad-CAM Library<br>
bash<br>
pip install grad-cam<br>
Apply Grad-CAM<br>
Visualize the regions of the image that contribute most to the predictions.<br>
### python
from pytorch_grad_cam import GradCAM<br>
from pytorch_grad_cam.utils.image import show_cam_on_image<br>
target_layer = model.layer4[-1] # Adjust based on the model's
architecture<br>
cam = GradCAM(model=model, target_layer=target_layer)<br>
for images, labels in test_loader:<br>
grayscale_cam = cam(input_tensor=images,<br>
target_category=labels[0])<br>
cam_image = show_cam_on_image(images[0].permute(1, 2, 0).numpy(),
grayscale_cam)<br>
plt.imshow(cam_image)<br>
plt.show()<br>

#### The codes might not work directly, there will be some errors, which you need to fix it.
