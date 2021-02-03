import io
import os
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
from flask import current_app as app

num_classes = 12
model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
model.classifier = DeepLabHead(2048, num_classes)
model.load_state_dict(torch.load(app.config['MODEL_PATH']))

device = 'cuda'
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()
model.to(device)

colors = [   0,   0,   0,
           192, 128, 128,
             0, 128,   0,
           128, 128, 128,
           128,   0,   0,
             0,   0, 128,
           192,   0, 128,
           192,   0,   0,
           192, 128,   0,
             0,  64,   0,
           128, 128,   0,
             0, 128, 128]

def transform_image(image):
    my_transforms = transforms.ToTensor()
    return my_transforms(image)

def supported_image_type(img):
    try:
        image = Image.open(img)
        return image.mode == 'RGB'
    except:
        return False

def predict(image_file):
    try:
        image = Image.open(image_file)
        tensor = transform_image(image=image)
        input_batch = tensor.unsqueeze(0)
        input_batch = input_batch.to(device)
        with torch.no_grad():
            outputs = model(input_batch)['out'][0]
        output_predictions = outputs.argmax(0)
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(image.size)
        r.putpalette(colors)
        file_path = os.path.splitext(image_file)[0] + "_mask" + ".jpg"
        head, tail = os.path.split(file_path)
        r.convert('RGB').save(file_path, "JPEG")
        return tail
    except:
        print(f"Something went wrong with the model. May be image format is not supported")
        return ""

#testing
def test():
    img = '/tmp/dog.jpg'
    cf = '/tmp/imagenet_class_index.json'

    p = predict(img, cf)
    print(f'Given image is: {p[1]}')

