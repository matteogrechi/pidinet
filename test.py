import torch
from models.pidinet import PiDiNet
from models.config import config_model

with open("trained_models/table5_pidinet.pth", "rb") as f:
    model = torch.load(f)

pdcs = config_model("carv4")
model = PiDiNet(60, pdcs, dil=24, sa=True)
x = torch.ones((1,3,100,100))
y = model(x)
# onnx = torch.onnx.export(model, x, "pidinet_model.onnx")
pth = torch.save(model, "pidinet_model.pth")