
import  torch
from YOLO import YoloBody
def main():
    input_shape = (3, 416, 416)
    model_onnx_path = "yolov4tiny.onnx"

    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #                        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model = YoloBody(3, 12).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 3, 416, 416, device=device)
    # 用1.7版本读取权重
    state_dict = torch.load('logs/Epoch120-Total_Loss0.5324-Val_Loss0.8735.pth', map_location=device)
    model.load_state_dict(state_dict)
    # 保存成1.4版本支持的格式
    torch.save(model.state_dict(), 'logs/for_onnx.pth', _use_new_zipfile_serialization=False)

    # Python解释器换成torch1.4的环境，重新读取，导出pytorch1.4版本对应的onnx
    state_dict = torch.load('logs/for_onnx.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.train(False)

    inputs = ['input_1']
    outputs = ['output_1', 'output_2']
    dynamic_axes = {'input_1': {0: 'batch'}, 'output_1': {0: 'batch'}}
    torch.onnx.export(model,
                      dummy_input,
                      model_onnx_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=inputs, output_names=outputs,
