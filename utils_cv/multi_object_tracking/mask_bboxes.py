import numpy as np
import pandas as pd
import cv2
import multiprocessing
import time

# Load mask
arr_mask = cv2.imread("../data/Benfica_data/mask_terrain_Patrick.png")
arr_out = np.where(arr_mask[:,:,0]==0)#black
black_tuples = tuple(zip(arr_out[1],arr_out[0])) #xy format

# Load track text file with bbox coord in xywh format
df_labels=pd.read_csv("./demo/benfica_20sec_tracked_0409.txt",header=None) 


#filter out bboxes with bottom coords outside of terrain
def checkBoxCoords_filter(i, row):#Assuming in bbox_xywh format    
    bottomLeftCoord = (row[2+0],row[2+1]+row[2+3] )
    bottomRightCoord = (row[2+0]+row[2+2],row[2+1]+row[2+3])    
    print("Row i:",i)
    if (bottomLeftCoord in black_tuples and bottomRightCoord in black_tuples):
        return(i, "Bottom coords are outside, remove")
    else:
        return(i,"")
def checkBoxCoords_filter_WHlimits(i, row):#Assuming in bbox_xywh format    
    bottomLeftCoord = (row[2+0],row[2+1]+row[2+3] )
    bottomRightCoord = (row[2+0]+row[2+2],row[2+1]+row[2+3])  
    print("Row i:",i)
    if (row[2+2]>2*41 and row[2+3]>2*52 ): #KIP: twice the max W and H obtained from Lorenzo gt bboxes
            return([i, "Bbox too big, remove"])
    else:
        if (bottomLeftCoord in black_tuples and bottomRightCoord in black_tuples):
            return([i,"Bottom coords are outside, remove"])
        else:
            return([i,""])          

# Run multiprocessing
inputs=range(len(df_labels))
num_cores=multiprocessing.cpu_count()-1
result_list=[]
pool=multiprocessing.Pool()
start=time.time()
corr_results = [pool.apply_async(checkBoxCoords_filter_WHlimits, args=(i, df_labels.iloc[i,:])) for i in inputs]
corrResult_list=[p.get() for p in corr_results]
end=time.time()
print("Duration of inner loop:", end-start)
pool.close()

corrResult_list.to_csv("multicore_res_wWH.csv")

df_corrResult=pd.DataFrame(corrResult_list)
df_corrResult.columns=["rowNumber","check_bboxCoords"]
df_res=pd.merge(df_labels, df_corrResult, how = "outer", left_index=True, right_index=True)#join by row index

df_labels_filtered=df_res[df_res["check_bboxCoords"]==""].drop(columns=['rowNumber','check_bboxCoords'])
df_labels_filtered.to_csv("./demo/benfica_20sec_tracked_0409_filteredBottomOutWH_deepsort.txt", header=None, index=None)

# df_labels["checkBox_bottomOut"]=df_labels.apply(checkBoxCoords_filter,axis=1)
# df_labels_filtered=df_labels[df_labels["checkBox_bottomOut"]!="Bottom coords are outside, remove"].drop(columns=['checkBox_bottomOut'])
# df_labels_filtered.to_csv("./demo/benfica_20sec_tracked_0409_filteredBottomOut2.txt", header=None, index=None)


