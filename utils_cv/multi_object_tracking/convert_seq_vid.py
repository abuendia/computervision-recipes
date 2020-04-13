import cv2
import numpy as np
import os
from os.path import isfile, join
import glob
import argparse


def convert_vidtoseq(seq_path,vid_path, fps):
    print("In convert_vidtoseq")
    vdo = cv2.VideoCapture()
    assert os.path.isfile(vid_path), "Error: path error"
    vdo.open(vid_path)
    
    os.mkdir(seq_path)
    idx_frame = 0
    while vdo.grab():        
        idx_frame += 1
        _, image = vdo.retrieve()
        cv2.imwrite(os.path.join(seq_path,"frame{:06d}.jpg".format(idx_frame)), image)

    
def convert_seqtovid(seq_path,vid_path, fps):
    print("In convert_seqtovid")
    img_array = []
    files = glob.glob(os.path.join(seq_path, '*.gif'))
    files.extend(glob.glob(os.path.join(seq_path, '*.png')))
    files.extend(glob.glob(os.path.join(seq_path, '*.jpg')))
    print(files)
    print(os.path.join(seq_path, '*.gif'))
    for filename in files:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)


    out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("SEQ_PATH", type=str)
    parser.add_argument("VID_PATH", type=str)
    parser.add_argument("FROMSEQ_TOVID", type=str, default="yes")
    parser.add_argument("--fps", type=int, default=30) #KIP: Include fps of video
    return parser.parse_args()
   
if __name__ == '__main__':
    args = parse_args()
    
    if args.FROMSEQ_TOVID=="yes":
        convert_seqtovid(args.SEQ_PATH,args.VID_PATH, args.fps)
    else:
        convert_vidtoseq(args.SEQ_PATH,args.VID_PATH, args.fps)
     

