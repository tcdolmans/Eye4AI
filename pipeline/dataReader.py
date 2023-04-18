import pandas as pd
from psychopy import visual, monitors, event
import numpy as np
import matplotlib.pyplot as plt
from titta import Titta, helpers_tobii as helpers
import h5py
import sys
import os as os

filename='C:\\Users\\admin\Documents\\Titta\\osieData\\'+str(sys.argv[1])+'.h5'

g = pd.read_hdf(filename, 'gaze')
msg = pd.read_hdf(filename, 'msg')
print(msg)

imgPath='C:\\Users\\admin\Documents\\Titta\\osie200\\'
im_names = []
for file in os.listdir(imgPath):
    if file.lower().endswith(".jpg") and not file.startswith('\.'):
        im_names.append(file)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


for i in im_names[0:100]:

    ts=msg[msg.msg.str.contains(i)]
    
    if not ts.empty:
        start=ts.system_time_stamp[ts.index[0]]
        stop=ts.system_time_stamp[ts.index[1]]
        trialTime=(stop-start)/1000
        
        gt=g[(g['system_time_stamp'] >= start) & (g['system_time_stamp'] <= stop)]

        gtI=gt[gt['right_gaze_point_on_display_area_y'].isna()]
        
        if gtI.shape[0]/gt.shape[0]<.25:

            
            plt.imshow(mpimg.imread('C:\\Users\\admin\Documents\\Titta\\osie200\\'+i));
            plt.scatter(y=[(1-gt.right_gaze_point_on_display_area_y)*600], x=[gt.right_gaze_point_on_display_area_x*1067], c='r', s=40);
            plt.text(10, -3, str(trialTime)+' ms', bbox=dict(fill=False, edgecolor='red', linewidth=2))
            plt.show();
        
   




# df_msg[1,1]
# type(df_msg)
# df.to_csv(sys.stdout, index=False)


# library("rhdf5")
# library(jpeg)
# library(png)
# library(raster)
# library(plyr)
# library(stringr)


# #read directly from h5 (for some reason, messages do not read. Data reads though)
# all<-h5ls("/Users/juklep/Desktop/osiePilot/pilot001b.h5")
# m<- h5read(file = '/Users/juklep/Desktop/osiePilot/pilot001b.h5', 
#              name = "/msg/")
# h <- h5read(file = '/Users/juklep/Desktop/osiePilot/pilot001b.h5', 
#             name = "/gaze/block1_items")
# d <- h5read(file = '/Users/juklep/Desktop/osiePilot/pilot001b.h5', 
#                      name = "/gaze/block1_values/")
# plot(d[1,],col='white')
# lines(d[1,],col='blue')
# lines(d[2,],col='red')


# #read from csv
# d<-read.csv("/Users/juklep/Desktop/osiePilot/pilot001b.csv")
# m<-read.csv("/Users/juklep/Desktop/osiePilot/pilot001bM.csv")
# m<-m[substr(m$msg,1,4)=='onse',]



# for (i in 1:nrow(m)){
  
#   image<-substr(m[i,2],nchar(m[i,2])-7,nchar(m[i,2]))
#   img <- readJPEG(paste("/Users/juklep/Desktop/osieOriginal/OSIE_100/",image,sep=''))  
#   plot(NULL, xlim=c(0,800), ylim=c(0,600))
#   rasterImage(img,1,1,800,600)

#     startT<-m[i,1]
  
#   points((d$left_gaze_point_on_display_area_x[d$system_time_stamp>startT&d$system_time_stamp<(startT+2950*1000)]*800)[],
#        ((1-d$left_gaze_point_on_display_area_y[d$system_time_stamp>startT&d$system_time_stamp<(startT+2950*1000)])*600)[], col='red', cex=1.5)}

