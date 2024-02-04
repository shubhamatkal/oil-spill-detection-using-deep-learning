import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import time
from flask import Flask, render_template, request
import os


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            #Convolutional block 1
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            #Convolutional block 2
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            #Convolutional block 3
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            #Convolutional block 4
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            #Convolutional block 5
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            #Flattening
            nn.Flatten()
        )

        #Fully connected layer
        self.fc = nn.Sequential(
            #Linear 1024 -> 256
            nn.Linear(2048,256),
            nn.ReLU(),
            nn.Dropout(0.1),

            #Linear 256 -> 2
            nn.Linear(256, 2),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Instantiate the model
model = CNN()

# Load the saved state dictionary
model.load_state_dict(torch.load('./model/spill.pth'))

crop_size = 128
loader = transforms.Compose([
            transforms.Resize(70),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()])


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        time.sleep(2)
        images = image_loader(file_path)
        label2cat = ['nospill', 'oilspill']


        with torch.no_grad():
            model.eval()
            output = model (images)
            result_string = f"there is %s with probability %s" %(label2cat[output.argmax(1).item()], "{:.2f}".format(max(np.exp(output)[0])))
            print(f"there is %s with probability %s" %(label2cat[output.argmax(1).item()], "{:.2f}".format(max(np.exp(output)[0]))))
        # Add your image processing logic here
        # For example, you can use a library like PIL to process the image
        # And print something in the terminal
        print(f"Processing image: {file_path}")

        # Render the result template with the result_string
        return render_template('result.html', input=result_string)

@app.route('/result', methods=['GET'])
def show_result():
    # You can add logic here if needed
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)