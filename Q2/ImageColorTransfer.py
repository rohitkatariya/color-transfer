#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook
from matplotlib import pyplot as plt
import os
import random
import pandas as pd
import numpy as np
import math
from numpy import linalg as LA


# In[2]:


from skimage import exposure
import datetime


# In[3]:


dir_color_transfer = 'data/color_transfer_image'
dir_input = dir_color_transfer+'/input/'
dir_output = dir_color_transfer+'/output/'


# In[4]:


image_names_dict = [['color_'+z.split('_')[1],'bw_'+z.split('_')[1]] for z in os.listdir(dir_input) if z.startswith('bw')]
[(k,v[0].split("_")[1].split('.')[0]) for k,v in enumerate(image_names_dict)]


# In[5]:


this_pair = image_names_dict[3]
this_pair


# In[6]:


def read_image_pair(this_pair):
    orig_image =cv2.imread(dir_input+this_pair[1],0)
#     origLAB = cv2.cvtColor(orig_image, cv2.COLOR_BGR2Lab)
    source_img =cv2.imread(dir_input+this_pair[0])
    sourceLAB = cv2.cvtColor(source_img, cv2.COLOR_BGR2Lab)
    return orig_image,sourceLAB


# In[7]:


# def getLabImage(source_img):
#     brightLAB = cv2.cvtColor(source_img, cv2.COLOR_BGR2Lab)
#     return brightLAB
# def getHistogram(intensity_values):
#     return pd.Series(intensity_values).value_counts()


# In[8]:


# def visualizeLab(brightLAB):
#     for i in range(3):
#         l = brightLAB[:,:,i]
#         plt.imshow(l)
#         plt.show()


# In[9]:


def applyLuminanceRemapping(orig_image,sourceLAB):
    #hist_source =  getHistogram(sourceLAB[:,:,0].reshape(-1))
    #hist_orig = getHistogram(orig_image.reshape(-1))
    #hist_source = hist_source.sort_index()
    #hist_orig = hist_orig.sort_index()
    # plt.bar(s.index,s)
    matched2 = exposure.match_histograms( sourceLAB[:,:,0],orig_image, multichannel=False)
    #m2 = getHistogram(matched2.reshape(-1))
    #m2 = m2.sort_index()
    #plt.bar(m2.index,m2)
    return matched2


# In[10]:


def get_jittered_sampling(source_image):
    m,n=source_image.shape()[0:1]


# In[ ]:





# In[11]:


orig_image,source_img_lab = read_image_pair(this_pair)


# In[12]:


sourceLAB = source_img_lab


# In[13]:


l_remaped = applyLuminanceRemapping(orig_image,sourceLAB)
sourceLAB[:,:,0] = l_remaped


# In[ ]:





# In[14]:


m,n=sourceLAB.shape[0:2]
square_side = math.floor(math.sqrt(m*n/1000))
x_start = 0
y_start = 0
gridded_image = sourceLAB
square_starts_x = []
square_starts_y = []
while y_start<m:
    # gridded_image = cv2.line(gridded_image, (0,y_start), (n-1,y_start), (0,0,255),1 )
    square_starts_y.append(y_start)
    y_start+=square_side

while x_start<n:
    #gridded_image = cv2.line(gridded_image, (x_start,0), (x_start,m-1 ), (0,0,255),1 )
    square_starts_x.append(x_start)
    x_start+=square_side


# In[15]:


# generating sample points
sample_points = []
for y_start in square_starts_y:
    for x_start in square_starts_x:
        random.randint(0, min(square_side,n-1-x_start))
        random.randint(0, min(square_side,m-1-y_start))
        random_pertubations = (random.randint(0, min(square_side,n-1-x_start)),random.randint(0, min(square_side,m-1-y_start)))
        this_sample = (x_start+random_pertubations[0],y_start+random_pertubations[1])
        sample_points.append(this_sample)
        #gridded_image = cv2.circle(gridded_image, this_sample[:2], radius=0, color=(255, 255, 255), thickness=-1)


# In[16]:


len(sample_points)


# In[17]:


def get_point_statistics(this_sample_point,sourceLAB):
    m,n=sourceLAB.shape[0:2]
    this_pt = sourceLAB[this_sample_point[1],this_sample_point[0]]
    if sourceLAB.shape[-1]==3:
        l_val,a_val,b_val = this_pt
    else:
        l_val= this_pt
        a_val,b_val =0,0
    frame_size = 11
    start_point = max(0,this_sample_point[0]-frame_size//2),max(0,this_sample_point[1]-frame_size//2)
    end_point = min(n-1,this_sample_point[0]+frame_size//2),min(m-1,this_sample_point[1]+frame_size//2)
    if sourceLAB.shape[-1]==3:
        nbhood = sourceLAB[start_point[1]:end_point[1]+1,start_point[0]:end_point[0]+1,0]
    else:
        nbhood = sourceLAB[start_point[1]:end_point[1]+1,start_point[0]:end_point[0]+1]
    var_this = np.std(nbhood)
    point_statistics = (int(l_val),a_val,b_val,var_this)
    return point_statistics
    


# In[18]:


all_point_stats=[]
for this_point in sample_points:
    this_point_stats = get_point_statistics(this_point,sourceLAB)
    all_point_stats.append(this_point_stats)


# In[19]:


m_orig,n_orig = orig_image.shape
transferedImage = np.zeros([m_orig,n_orig,3])
m_orig,n_orig


# In[20]:


def get_closest_chromatics(orig_point_stats,all_sample_point_stats):
    closest_d = float('inf')
    chromatics_closest = (-1,-1)
    for this_sample_point in all_sample_point_stats:
        l_diff = this_sample_point[0] - orig_point_stats[0]
        var_diff = this_sample_point[3] - orig_point_stats[3]
        d_this = l_diff**2+var_diff**2
        if d_this<closest_d:
            closest_d = d_this
            chromatics_closest = this_sample_point[1],this_sample_point[2]
    return chromatics_closest


# In[ ]:





# In[21]:


#iterating over image and applying color transfer
np_all_point_stats = np.array(all_point_stats)
print(orig_image.shape)
for i in range(m_orig):
    if i%100==0:
        print(i)
    for j in range(n_orig):
        transferedImage[i][j][0]=orig_image[i][j]
        # print((i,j),orig_image.shape)
        orig_point_stats =get_point_statistics((j,i),orig_image)
        diff = np_all_point_stats[:,[0,3]] - (orig_point_stats[0],orig_point_stats[3])
        idx_min_match = np.argmin(LA.norm(diff,axis=1))
        a_val,b_val = all_point_stats[idx_min_match][1:3]
        transferedImage[i][j][1] = a_val
        transferedImage[i][j][2] = b_val
#         break
#     break


# In[23]:


# for i in range(m_orig) :
#     for j in range(n_orig) :
#         for k in range(3):
#             transferedImage[i][j][k]= np.uint8(transferedImage[i][j][k])
# #             if transferedImage[i][j][k]>255:
# #                 transferedImage[i][j][k]=255
# #             if transferedImage[i][j][k]<0:
# #                 transferedImage[i][j][k]=0


# In[24]:


# transferedImage.shape


# In[25]:


transferedImage_rgb = cv2.cvtColor(transferedImage.astype('uint8'), cv2.COLOR_Lab2BGR)
cv2.imwrite(dir_output+'{}ti_{}'.format(datetime.datetime.now().strftime("%H_%M_%S"),this_pair[0].split("_")[1]), transferedImage_rgb)


# In[26]:


# orig_image = cv2.cvtColor(transferedImage_rgb, cv2.COLOR_BGR2RGB)
# plt.imshow(orig_image)
# plt.show()


# In[27]:


# plt.imshow(transferedImage.astype('uint8'))
# plt.show()


# In[ ]:





# In[ ]:




