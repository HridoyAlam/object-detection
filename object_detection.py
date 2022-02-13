import argparse
import io
import base64
from io import BytesIO, StringIO

import torch
from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

BIKE_DETECTION_URL = "/motorcycle"
OBJECT_DETECTION_URL = "/object"

@app.route(BIKE_DETECTION_URL, methods=["POST"])
def bike_detect():
    if request.method != "POST":
        return

    res         = request.get_json()
    im          = res['image']
    im          = Image.open(BytesIO(base64.b64decode(im)))
    image       = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)

    img         = Image.fromarray(image).convert('RGB')

    results     = model(img, size=640)  # reduce size=320 for faster inference

    df          = results.pandas().xyxy[0]

    class_name  = df['name'].tolist()

    return jsonify("Accepted") if 'motorcycle' in class_name else jsonify("Invalid Image")


@app.route(OBJECT_DETECTION_URL, methods=["POST"])
def object_detect():
    if request.method != "POST":
        return
    res         = request.get_json()
    name        = res['name']
    im          = res['image']
    im          = Image.open(BytesIO(base64.b64decode(im)))
    image       = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)

    img         = Image.fromarray(image).convert('RGB')
    results     = model(img, size=640)  # reduce size=320 for faster inference

    df          = results.pandas().xyxy[0]
    c_name = name.capitalize()
    response_t =  f'{c_name} exist in the image'
    response_f =  f'{c_name} does not exist in the image'

    class_name  = df['name'].tolist()

    return jsonify(response_t) if name in class_name else jsonify(response_f)



if __name__ == "__main__":
    model   = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, device = 'cpu')  # force_reload to recache
    app.run(host="0.0.0.0", port=5000, debug=True)  # debug=True causes Restarting with stat