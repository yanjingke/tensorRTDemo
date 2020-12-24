import torch
from torch.autograd import Variable
import onnx
print(torch.__version__)
# torch  -->  onnx
input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, 224, 224)).cuda()
# model = torchvision.models.resnet50(pretrained=True).cuda()
model = torch.load('resnet50.pth', map_location="cuda:0")
torch.onnx.export(model, input, 'resnet50.onnx', input_names=input_name, output_names=output_name, verbose=True)
# 模型可视化
# netron.start('resnet50.onnx')