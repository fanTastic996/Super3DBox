import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from .utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model = VGGT()
_URL = "/home/lanyuqing/myproject/vggt/training/logs/exp001/ckpts/checkpoint.pt"
data_root = '/data/lyq/ca1m/ca1m/train-CA-1M-slam'
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

# Load and preprocess example images (replace with your own image paths) 
scene_id = '42897951'
image_names = [f"{data_root}/{scene_id}/799.png", f"{data_root}/{scene_id}/341.png"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        print("")