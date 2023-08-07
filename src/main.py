from fastapi import FastAPI, File, UploadFile
from PIL import Image
import json
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets.utils import download_url
from pathlib import Path
from torch.nn import functional as F

app = FastAPI()

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_classes():
    if not Path("data/imagenet_class_index.json").exists():
        download_url("https://git.io/JebAs", "data", "imagenet_class_index.json")

    with open("data/imagenet_class_index.json") as f:
        data = json.load(f)
        class_names = [x["ja"] for x in data]

    return class_names

device = get_device(use_gpu=True)
model = torchvision.models.resnet50(pretrained=True).to(device)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class_names = get_classes()

@app.post("/predict")
async def predict(file: UploadFile = File(..., alias="image")):
    img = Image.open(file.file)
    inputs = transform(img)
    inputs = inputs.unsqueeze(0).to(device)
    model.eval()
    outputs = model(inputs)
    probs = F.softmax(outputs, dim=1)
    probs, indices = probs.sort(dim=1, descending=True)

    top_class = class_names[indices[0][0]]
    return {"prediction": top_class}
