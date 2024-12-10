import torch
from torchvision.transforms import v2
from torchvision import transforms
from PIL import ImageOps
import numpy as np

from class_MyModel import MyModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyModel(784, 10).to(device)

load_model = torch.load('/Users/alyona/Task/model_state_dict.pt', weights_only=True)
model.load_state_dict(load_model['state_model'])


def pred(image):
    sample = transform(image)
    model.eval()
    with torch.no_grad():
        output = model(sample.view(1, 28*28).to(device)).detach()
        return f'Your number is: {np.argmax(output).item()}'


def transform(image):
    sample = ImageOps.grayscale(image)
    resize = transforms.Compose([transforms.Resize((28, 28))])
    transform_img = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5,), std=(0.5,))
        ]
    )

    sample = resize(sample)
    sample = transform_img(sample)
    return sample
