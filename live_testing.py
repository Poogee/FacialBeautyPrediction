import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import pandas as pd 
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50()
model = model.to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model.load_state_dict(torch.load("resnet50_finetuned.pth",map_location=torch.device('cpu'))) #питон попросил сделать это на маке, на компе убрать надо будет map_location...
model.eval()

scale_k = 0.3

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
])

def PrintBeautyScore(sub_frame):
    image = transform(Image.fromarray(sub_frame,"RGB"))
    print(model(image.unsqueeze(0)).item())


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("bruh, dead camera")
    exit()

#for delay of printing beauty score
skip_count = 0

while True:
    ret, frame = cap.read()
    skip_count += 1
    # print(skip_count)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        bottom = int(max(0,y-h*scale_k))
        up = int(min(cap.get(4), y + h * (1 + scale_k)))
        left = int(max(0, x - w * scale_k))
        right = int(min(cap.get(3), x + w * (1 + scale_k)))
        cv2.rectangle(frame, (left, bottom), (right, up), (255, 0, 0), 2)
        if skip_count >= 50:
            skip_count = 0
            PrintBeautyScore(frame[bottom:up, left:right])
            
    cv2.imshow('Face Detection', frame)   

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
