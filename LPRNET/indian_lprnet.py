import tensorflow as tf 
from model import LPRNet
import cv2
import numpy as np

tf.compat.v1.enable_eager_execution()

class LPR():
    
    CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self, weights_path):

        NUM_CLASS = len(LPR.CHARS)+1

        self.net = LPRNet(NUM_CLASS)  
        self.net.load_weights(weights_path)

    def read_plate_from_file(self, plate_img):
        frame = cv2.imread(plate_img)
        img = cv2.resize(frame, (94,24))
        img = np.expand_dims(img,axis = 0)
        #get the output sequence
        pred = self.net.predict(img, LPR.CHARS)
        result_ctc = self.decode_pred(pred, LPR.CHARS)
        result = result_ctc[0].decode('utf-8')
        print("Detected: ", result)

        return result

    def read_plate_from_array(self, image):
        img = cv2.resize(image, (94,24))
        img = np.expand_dims(img,axis = 0)
        #get the output sequence
        pred = self.net.predict(img, LPR.CHARS)
        result_ctc = self.decode_pred(pred, LPR.CHARS)
        result = result_ctc[0].decode('utf-8')
        print("Detected: ", result)

        return result

    def decode_pred(self, pred,classnames):

        pred = np.mean(pred, axis = 1)
        samples, times = pred.shape[:2]
        input_length = tf.convert_to_tensor([times] * samples)
        decodeds, logprobs = tf.keras.backend.ctc_decode(pred, input_length, greedy=True, beam_width=100, top_paths=1)
        decodeds = np.array(decodeds[0])

        results = []
        for d in decodeds:
            text = []
            for idx in d:
                if idx == -1:
                    break
                text.append(classnames[idx])
            results.append(''.join(text).encode('utf-8'))
        return results