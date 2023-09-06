from fastapi import FastAPI, File, UploadFile
from PIL import Image
import pyheif
import json
import torch
import torchvision
from torchvision import transforms
from pathlib import Path
from torch.nn import functional as F

app = FastAPI()

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def heic_to_image(heic_path):
    heif_file = pyheif.read(heic_path)
    image = Image.frombytes(
        heif_file.mode, 
        (heif_file.width, heif_file.height),
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    return image

with open('data/flyer_class_names.json', 'r') as f:
    class_names = json.load(f)

device = get_device(use_gpu=True)

model = torchvision.models.resnet18(pretrained=False)  
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('src/models/flyer_model.pth', map_location=device))
model.to(device)

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(..., alias="image")):
    file_ext = Path(file.filename).suffix  # ファイル拡張子を取得

    if file_ext.lower() == '.heic':
        # 一時ファイルとして保存してからHEICを変換
        temp_path = "temp.heic"
        with open(temp_path, "wb") as buffer:
            buffer.write(file.file.read())
        img = heic_to_image(temp_path)
    else:
        img = Image.open(file.file).convert("RGB")

    inputs = transform(img)
    inputs = inputs.unsqueeze(0).to(device)
    
    model.eval()
    outputs = model(inputs)
    probs = F.softmax(outputs, dim=1)
    probs, indices = probs.sort(dim=1, descending=True)

    top_class = class_names[indices[0][0]]
    return {"prediction": top_class}
