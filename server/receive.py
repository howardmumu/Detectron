# -*- coding: utf-8 -*-
# __author__="ZJL"

from flask import Flask
from flask import request
from flask import make_response, Response
import cv2
import json
import numpy as np
from detector import detect_image
app = Flask(__name__)

def Response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST' :
        # datax = request.form.to_dict()
        print("收到")
        # content = str(datax)
        # resp = Response_headers(content)
        resp=request.data

        nparr = np.fromstring(resp, np.uint8)
        img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return detect_image(img_decode)
    else:
        content = json.dumps({"error_code": "1001"})
        resp = Response_headers(content)
        return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=(8888), threaded=True)