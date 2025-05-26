from fastapi import FastAPI
import torch
from torch import nn

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

@app.get("/")
def root():
    return {"hello": "world"}

@app.post("/items")
def create_item(item: str):
    items.append(item)
    return items

