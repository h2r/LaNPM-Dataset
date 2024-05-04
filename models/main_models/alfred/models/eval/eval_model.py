from ai2thor.controller import Controller
import argparse
from model.seq2seq_im_mask import Module # model class

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default="model.pth")

args = parser.parse_args()

def init(scene):
    controller = Controller(
        massThreshold = 1,
        agentMode="arm",
        scene = scene,
        snapToGrid=False,
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width= 1280,
        height= 720,
        fieldOfView=60
    )


def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = Module(checkpoint['args'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def step():
    pass

def run():
    pass