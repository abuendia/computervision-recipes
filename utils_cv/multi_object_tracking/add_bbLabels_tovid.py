import numpy as np
import cv2
import pandas as pd
import argparse
import time
import os

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# Adapted from https://github.com/ZQPei/deep_sort_pytorch
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, frame_no=None, identities=None, offset=(0,0), with_bboxWH="yes"):    
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
         
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0] #KIP: last 2 are font_scale and thickness
        if with_bboxWH=="yes":
            cv2.rectangle(img,(x1, y1),(x1+x2,y1+y2),color,1) #KIP: modified, before was 
        else: #Coordinates of top left and bottom right
            cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
            #(img,(x1, y1),(x2,y2),color,3)
        #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1) #KIP commented out
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        
        # KIP: added rectangle to add frame_no:
        if frame_no is not None:
             cv2.rectangle(img,(1, 1),(5+200,5+30),[255,255,25],3)
             frame_label = "frame:"+str(frame_no)
             cv2.putText(img,frame_label,(5,5+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
               
    return img

def get_bboxes_fromFrame(frame_no):
    df=df_labels[df_labels[0]==frame_no][[2,3,4,5,1]] #2,3,4,5 are the bbox coords, 1 is the id    
    return (df.to_numpy())

def main(args):
    # Get bbox coords+id_labels
    print("args.LABELS_PATH:", args.labels_path)
    assert os.path.isfile(args.labels_path), "Error: path error for label csv file"
    global df_labels
    df_labels=pd.read_csv(args.labels_path,header=None)   
    
    # Take in video
    vdo = cv2.VideoCapture()    
    assert os.path.isfile(args.VIDEO_PATH), "Error: path error for video"
    vdo.open(args.VIDEO_PATH)
    im_width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert vdo.isOpened()
    fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(args.save_path, fourcc, args.fps, (im_width,im_height))
    
    
    #Go through each frame and add labels,output tracked video
    idx_frame = 0
    while vdo.grab():        
        _, ori_im = vdo.retrieve()        
        arr_bbox=get_bboxes_fromFrame(idx_frame)   
        if args.plot_frames=="yes":
            ori_im = draw_boxes(ori_im, arr_bbox[:,:4], idx_frame, arr_bbox[:,-1], with_bboxWH = args.detwith_bboxWH)
        else:
            ori_im = draw_boxes(ori_im, arr_bbox[:,:4], arr_bbox[:,-1], with_bboxWH = args.detwith_bboxWH)
        idx_frame += 1
        writer.write(ori_im)

    ##
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--labels_path", type=str, default="labels.csv")
    parser.add_argument("--save_path", type=str, default="video_labeled.avi")
    parser.add_argument("--fps", type=int, default=30) #KIP: Include fps of video
    parser.add_argument("--detwith_bboxWH", type=str, default="yes") #KIP: detection is with bboxes annotated with top-left coord and bbox width and height
    parser.add_argument("--plot_frames", type=str, default="yes") #KIP
    
    return parser.parse_args()
   

if __name__ == '__main__':
    args = parse_args()    
    start = time.time()
    main(args)
    end = time.time()
    print("time: {:.03f}s".format(end-start))
    