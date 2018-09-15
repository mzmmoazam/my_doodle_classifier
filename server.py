from flask import Flask, render_template, request
import numpy as np
import re
import base64
from PIL import Image
from scipy.misc import imsave, imread, imresize

from util.train import conv
from util.prepare_data import normalize
import json

app = Flask(__name__)

model = conv(classes=9,input_shape=(28, 28, 1))
model.load("./model/doodle_classifier_1.0.tflearn")




@app.route("/", methods=["GET", "POST"])
def ready():
    global model
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]

        img = base64.decodestring(data.encode('ascii'))
        with open('temp.png', 'wb') as output:
            output.write(img)
        x = imread('temp.png', mode='L')
        # resize input image to 28x28
        x = imresize(x, (28, 28))

        x = np.expand_dims(x, axis=0)
        x = np.reshape(x, (28, 28, 1))

        # invert the colors
        x = np.invert(x)
        # brighten the image by 60%
        for i in range(len(x)):
            for j in range(len(x)):
                if x[i][j] > 50:
                    x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = normalize(x)
        val = model.predict(np.array([x]))
        classes = ["Bird", "Grapes", "Circle", "Book", "Candle", "Banana", "Apple", "Cloud", "Pineapple"]
        pred = classes[int(np.argmax(val))]
        print(pred)
        print(list(val[0]))
        return render_template("index.html", preds=list(val[0]), classes=json.dumps(classes), chart=True,
                               putback=request.form["payload"])


app.run()
