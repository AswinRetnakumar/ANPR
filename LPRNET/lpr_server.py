from flask import Flask, request
from flask_cors import CORS
from indian_lprnet import LPR
import cv2
import numpy as np

app = Flask(__name__)
cors = CORS(app)

lpr = LPR('./indian_plate_model_11_07/new_out_model_best.pb')


@app.route('/recog_plate', methods=['POST', 'GET'])
def read_plate():
    if request.method == 'GET':
        return 'LPRNET is enabled and running'

    else:
        print(request.files)
        nparr = np.fromstring(request.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        plate_num = lpr.read_plate_from_array(img)

        return {'plate_number': plate_num}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)