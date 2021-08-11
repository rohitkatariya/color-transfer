#!/usr/bin/env python
# coding: utf-8
import cv2
from matplotlib import pyplot as plt
import os
import random
import pandas as pd
import numpy as np
import math
from numpy import linalg as LA

from skimage import exposure
import datetime
import pdb

dir_color_transfer = 'data/color_transfer_image'
dir_input = dir_color_transfer+'/input/'
dir_output = dir_color_transfer+'/output/'

class ColorTransfer:
    def __init__(self):
        self.box_color = (125, 125, 125) 
        self.box_color_red = (0, 0, 255) 
        self.box_thickness = 1
        self.current_index = 0
        #choose image to work on
        image_names_dict = [['color_'+z.split('_')[1],'bw_'+z.split('_')[1]] for z in os.listdir(dir_input) if z.startswith('bw')]
        print([(k,v[0].split("_")[1].split('.')[0]) for k,v in enumerate(image_names_dict)])
        image_id = input('which image to load?')
        this_pair = image_names_dict[int(image_id)]
        print("woriking on ",this_pair)
        self.this_pair = this_pair
        
        # read orig and source images
        self.orig_image,self.source_img = self.read_image_pair(this_pair)
        # source lab
        self.sourceLAB = cv2.cvtColor(self.source_img, cv2.COLOR_BGR2Lab)

        # Histogram Equalization
        l_remaped = self.applyLuminanceRemapping(self.orig_image,self.sourceLAB)
        self.sourceLAB[:,:,0] = l_remaped

        # setting up jittering grid
        self.m_source,self.n_source=self.sourceLAB.shape[0:2]
        approx_num_box=1000
        self.square_side = math.floor(math.sqrt(self.m_source*self.n_source/approx_num_box))
        print("jittering square_side",self.square_side)
        x_start = 0
        y_start = 0
        # gridded_image = self.sourceLAB
        self.square_starts_x = []
        self.square_starts_y = []
        while y_start<self.m_source:
            # gridded_image = cv2.line(gridded_image, (0,y_start), (n-1,y_start), (0,0,255),1 )
            self.square_starts_y.append(y_start)
            y_start+=self.square_side

        while x_start<self.n_source:
            #gridded_image = cv2.line(gridded_image, (x_start,0), (x_start,m-1 ), (0,0,255),1 )
            self.square_starts_x.append(x_start)
            x_start+=self.square_side
        
        # 1 sample point to choosen from each jittering grid box
        self.sample_points = []
        for y_start in self.square_starts_y:
            for x_start in self.square_starts_x:
                random_pertubations_x = random.randint(0, min(self.square_side,self.n_source-1-x_start))
                random_pertubations_y = random.randint(0, min(self.square_side,self.m_source-1-y_start))
                this_sample = (x_start+random_pertubations_x,y_start+random_pertubations_y)
                self.sample_points.append(this_sample)
                #gridded_image = cv2.circle(gridded_image, this_sample[:2], radius=0, color=(255, 255, 255), thickness=-1)
        
        # computing neighborhood sats for each sample point
        self.all_point_stats=[]
        for this_point in self.sample_points:
            this_point_stats = self.get_point_statistics(this_point,self.sourceLAB)
            self.all_point_stats.append(this_point_stats)

        # setting up tranfered image
        self.m_orig,self.n_orig = self.orig_image.shape
        self.transferedImage = np.zeros([self.m_orig,self.n_orig,3])
        
    def read_image_pair(self,this_pair):
        orig_image =cv2.imread(dir_input+this_pair[1],0)
        source_img =cv2.imread(dir_input+this_pair[0])
        return orig_image,source_img

    def applyLuminanceRemapping(self,orig_image,sourceLAB):
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

    def get_jittered_sampling(self,source_image):
        m,n=source_image.shape()[0:1]

    def get_point_statistics(self,this_sample_point,sourceLAB):
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
        std_this = np.std(nbhood)
        point_statistics = (int(l_val),a_val,b_val,std_this)
        return point_statistics


    def get_closest_chromatics(self,orig_point_stats,all_sample_point_stats):
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
    def get_current_start_end_points(self):
        this_start_point = self.all_swatch_grid_starts[self.current_index]
        this_end_point = (min(this_start_point[0]+self.swatch_size,self.n_orig-1),
                            min(this_start_point[1]+self.swatch_size,self.m_orig-1)) 
        return (this_start_point,this_end_point)
    
    def update_image(self,pt_x_y):
        x,y=pt_x_y
        relevant_sample_points = [] 
        for point_idx,(this_sample_point_x,this_sample_point_y) in enumerate(self.sample_points):
            relevant_point=True
            if this_sample_point_x< x-self.swatch_size//2 or this_sample_point_x > x+self.swatch_size//2:
                relevant_point=False
            if this_sample_point_y< y-self.swatch_size//2 or this_sample_point_y > y+self.swatch_size//2:
                relevant_point=False
            if relevant_point==True:
                relevant_sample_points.append( self.all_point_stats[point_idx] )
        
        if len(relevant_sample_points)==0:
            print( "ERROR: getting 0 sample points")
            return
        
        np_all_point_stats = np.array(relevant_sample_points)
        (this_start_point,this_end_point) =  self.get_current_start_end_points()
        for i in range(this_start_point[1],this_end_point[1]+1):
            for j in range(this_start_point[0],this_end_point[0]+1):
                self.transferedImage[i][j][0]=self.orig_image[i][j]
                print(j,i,self.m_orig,self.n_orig)
                orig_point_stats =self.get_point_statistics((j,i),self.orig_image)
                diff = np_all_point_stats[:,[0,3]] - (orig_point_stats[0],orig_point_stats[3])
                idx_min_match = np.argmin(LA.norm(diff,axis=1))
                a_val,b_val = relevant_sample_points[idx_min_match][1:3]
                self.transferedImage[i][j][1] = a_val
                self.transferedImage[i][j][2] = b_val
        
    def write_text_on_image(self,orig_image_copy):
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (0, 0, 0) 
        
        # Line thickness of 2 px 
        thickness = 2
        # org 
        org = (50, 50) 
        orig_image_copy= cv2.putText(orig_image_copy, 'Press Enter!',org, cv2.FONT_HERSHEY_SIMPLEX,  
        fontScale, color, thickness, cv2.LINE_AA) 
        return orig_image_copy

    def click_event(self,event, x, y, flags, params): 
        if event == cv2.EVENT_LBUTTONDOWN:
            print((x, y))
            if self.current_index>=len(self.all_swatch_grid_starts):
                orig_image_copy = self.orig_image.copy()
                orig_image_copy = self.write_text_on_image(orig_image_copy)
                cv2.imshow('bw_img', orig_image_copy)

                source_img_copy = self.source_img.copy()
                source_img_copy = self.write_text_on_image(source_img_copy)
                cv2.imshow('color_img', source_img_copy) 
                return

            self.update_image((x, y))

            # update orig image based on click
            self.plot_current_bw((x,y))
            # font = cv2.FONT_HERSHEY_SIMPLEX 
            # cv2.putText(bw_img, str(x) + ',' +
            #         str(y), (x,y), font, 
            #         1, (255, 0, 0), 2) 
            # #cv2.imshow('image', color_img) 
            # #bw_img = cv2.imread('data/color_transfer_image/input/bw_agri.jpg', 1) 
            # cv2.imshow('bw_img', bw_img) 


    def click_event_2(self,event, x, y, flags, params): 
        if event == cv2.EVENT_LBUTTONDOWN:
            print((x, y))
            swatch_start_x = max(x-self.swatch_size//2 , 0)
            swatch_start_y = max(y-self.swatch_size//2 , 0)
            swatch_end_x = min(x+self.swatch_size//2,self.n_source-1)
            swatch_end_y = min(y+self.swatch_size//2,self.m_source-1)
            this_swatch = [(swatch_start_x,swatch_start_y),(swatch_end_x,swatch_end_y)] 
            self.selectedSwatches.append(this_swatch)
            # draw rectangle on source image
            print(this_swatch)

            
            self.source_img_copy = cv2.rectangle(self.source_img_copy , this_swatch[0], this_swatch[1], self.box_color_red, self.box_thickness) 
            cv2.imshow('color_img', self.source_img_copy) 
    def plot_current_bw(self,point_clicked=None):
        # if point_clicked==None:
        self.current_index+=1
        orig_image_copy = self.orig_image.copy()
        if self.current_index>=len(self.all_swatch_grid_starts):
            pass
        else:
            
            this_start_point = self.all_swatch_grid_starts[self.current_index]
            print('this_start_point',this_start_point)
            this_end_point = (min(this_start_point[0]+self.swatch_size,self.n_orig-1),
                            min(this_start_point[1]+self.swatch_size,self.m_orig-1)) 
            orig_image_copy = cv2.rectangle(orig_image_copy, this_start_point, this_end_point, self.box_color, self.box_thickness) 
        
        cv2.imshow('bw_img', orig_image_copy) 
        transferedImage_rgb = cv2.cvtColor(self.transferedImage.astype('uint8'), cv2.COLOR_Lab2BGR)
        cv2.imshow('transfered_img', transferedImage_rgb) 
        


    def define_swatch_grid(self,swatch_size):
        x_start = 0
        y_start = 0
        # gridded_image = self.sourceLAB
        swatch_square_starts_x = []
        swatch_square_starts_y = []
        while y_start<self.m_orig:
            swatch_square_starts_y.append(y_start)
            y_start+=swatch_size

        while x_start<self.n_orig:
            #gridded_image = cv2.line(gridded_image, (x_start,0), (x_start,m-1 ), (0,0,255),1 )
            swatch_square_starts_x.append(x_start)
            x_start+=swatch_size

        all_swatch_grid_starts = []
        for y_start in swatch_square_starts_y:
            for x_start in swatch_square_starts_x:
                this_start_point = x_start,y_start
                all_swatch_grid_starts.append(this_start_point) 
        return all_swatch_grid_starts
    
     
    
    def run_this(self,alg_type):
        print("running {} algo".format(alg_type))
        if alg_type == "global":
            #iterating over image and applying color transfer
            np_all_point_stats = np.array(self.all_point_stats)
            print(self.orig_image.shape)
            # generating sample points
            for i in range(self.m_orig):
                if i%100==0:
                    print('row#',i)
                for j in range(self.n_orig):
                    self.transferedImage[i][j][0]=self.orig_image[i][j]
                    # print((i,j),orig_image.shape)
                    orig_point_stats =self.get_point_statistics((j,i),self.orig_image)
                    diff = np_all_point_stats[:,[0,3]] - (orig_point_stats[0],orig_point_stats[3])
                    idx_min_match = np.argmin(LA.norm(diff,axis=1))
                    a_val,b_val = self.all_point_stats[idx_min_match][1:3]
                    self.transferedImage[i][j][1] = a_val
                    self.transferedImage[i][j][2] = b_val
            transferedImage_rgb = cv2.cvtColor(self.transferedImage.astype('uint8'), cv2.COLOR_Lab2BGR)
            cv2.imwrite(dir_output+'{}ti_{}'.format(datetime.datetime.now().strftime("%H_%M_%S"),self.this_pair[0].split("_")[1]), transferedImage_rgb)
            cv2.imshow('transferedImage_rgb', transferedImage_rgb) 
            cv2.waitKey(0) 
        elif alg_type == 'swatches':
            self.swatch_size = 50
            self.all_swatch_grid_starts = self.define_swatch_grid(self.swatch_size)
            
            self.current_index=-1
            self.plot_current_bw()
            
            # displaying the image 
            cv2.imshow('color_img', self.source_img) 
            cv2.moveWindow("color_img", 1020,20)

            transferedImage_rgb = cv2.cvtColor(self.transferedImage.astype('uint8'), cv2.COLOR_Lab2BGR)
            cv2.imshow('transfered_img', transferedImage_rgb) 
            cv2.moveWindow("transfered_img", 20,420)

            cv2.setMouseCallback('color_img', self.click_event) 
            # wait for a key to be pressed to exit 
            cv2.waitKey(0) 

            # close the window 
            cv2.destroyAllWindows() 
            # pdb.set_trace()
            transferedImage_rgb = cv2.cvtColor(self.transferedImage.astype('uint8'), cv2.COLOR_Lab2BGR)
            cv2.imwrite(dir_output+'{}ti_{}'.format(datetime.datetime.now().strftime("%H_%M_%S"),self.this_pair[0].split("_")[1]), transferedImage_rgb)
        # elif alg_type == 'swatches':
        #     self.swatch_size = 70
        #     self.selectedSwatches = []
        #     self.source_img_copy = self.source_img.copy()
        #     # displaying the image 
        #     cv2.imshow('color_img', self.source_img) 
        #     cv2.moveWindow("color_img", 1020,20)

        #     cv2.setMouseCallback('color_img', self.click_event_2) 
        #     # wait for a key to be pressed to exit 
        #     cv2.waitKey(0) 

        #     # close the window 
        #     cv2.destroyAllWindows() 

            
if __name__=="__main__":
    alg_type_inp = input("entere 1 for global, 2 for swatches")
    if int(alg_type_inp)==1:
        alg_type = 'global'
    else:
        alg_type = 'swatches'
    c_obj = ColorTransfer()
    c_obj.run_this(alg_type)
# exit()

