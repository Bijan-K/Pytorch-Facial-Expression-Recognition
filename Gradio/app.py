### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

model = torchvision.models.efficientnet_b2()

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=7),
)


for param in model.parameters():
   param.requires_grad = False

model.load_state_dict(
    torch.load(
        f="trained_model.pt",
        map_location=torch.device("cpu"),
    )
)

def preprocessImg(img):
   transform = transforms.Compose([
    #    transforms.Grayscale(),
       transforms.Resize((256,256)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   ])
   img = transform(img)
   return img

def predict(img) -> Tuple[Dict, float]:
    start_time = timer()
    
    img = preprocessImg(img).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time


title = "Facial Expression Classifier"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of facial expressions"
article = "for source code you can visit [my github](https://github.com/Bijan-K/Pytorch-Facial-Expression-Recognition)."

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), 
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

demo.launch()