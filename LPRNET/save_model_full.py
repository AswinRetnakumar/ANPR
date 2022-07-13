import tensorflow as tf 
from tensorflow import keras
import os 
import argparse
from model import LPRNet
from datetime import datetime
import utils
import evaluate
import math

def save(args):
    CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    NUM_CLASS = len(CHARS)+1

    net = LPRNet(NUM_CLASS)  
    net.load_weights(args['weights_path'])
    tf.compat.v1.enable_eager_execution()
    val_batch_size = 4
    val_len = len(next(os.walk(args["val_dir"]))[2])
    val_batch_len = int(math.floor(val_len / val_batch_size))  
    val_gen = utils.DataIterator(img_dir=args["val_dir"],batch_size = val_batch_size)
    evaluator = evaluate.Evaluator(val_gen,net, CHARS,val_batch_len, val_batch_size)

    val_loss = evaluator.evaluate()
    now = datetime.now()
    net.save(os.path.join(args["saved_dir"], "model_"+now.strftime("%m %d,%H:%M:%S")+"_val_loss_"+str(val_loss)+".pb"))
    print("Model saved... at: "+" val loss "+str(val_loss))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path",default = "./saved", help = "path to the weights directory",)
    parser.add_argument("--val_dir",default = "./valid", help = "path to the validation directory")
    parser.add_argument("--saved_dir",default = "./saved", help = "path to the save directory")

    args = vars(parser.parse_args())
    save(args)
