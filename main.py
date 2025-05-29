from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io

class Item(BaseModel):
    text: str

class Cifar10_clf(nn.Module):
    def __init__(self):
        super(Cifar10_clf, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self ,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

app = FastAPI()
model = Cifar10_clf()
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

@app.get("/")
def root():
    return {"hello": "world"}

@app.post("/clf")
def clf(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Must match your model's expected size
        transforms.ToTensor(),
    ])
    
    x = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(x)
        prediction = torch.argmax(output, dim=1).item()
        label = classes[prediction]
    return {"filename": file.filename, "prediction": prediction, "label":label}
