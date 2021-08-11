import pandas as pd
from sklearn.neighbors import KDTree
import random
class PopularitySampling:
    def colors_and_counts(self,input_img):
        return pd.Series([tuple(x) for x in input_img.reshape(-1,3)]).value_counts()
    
    def getKdTree(self,topk_colors_list):
        tree = KDTree(topk_colors_list, leaf_size=5) 
        return tree
    
    def getClosestColor(self,this_color):
        dist, ind = self.KDSearchTree.query([this_color], k=1)                
        return self.topk_colors_list[ind[0][0]]
    
    def get_quantized_img(self,input_img):
        img_new = input_img.copy()
        m_orig,n_orig,_ = img_new.shape
        for i in range(m_orig):
            if i%100==0:
                print(i)
            for j in range(n_orig):
                img_new[i][j] = self.getClosestColor(input_img[i][j])
        return img_new
    
    def dither_popularity(self):

        '''
        Returns the dithered image by applying Floyd-Steinberg dithering algorithm 
        '''
        # "qimg" is a copy of the image and updates the pixels with closest color in palette
        print("#Applying Dithering")
        qimg = self.inputImage.copy()
        m_orig =qimg.shape[0]
        n_orig =qimg.shape[1]
        for i in range(m_orig):
            if i%100==0:
                print(i)
            for j in range(n_orig):     
                closest = self.getClosestColor(qimg[i][j])
                qimg[i][j] = closest
                # finding the quantization error
                err =    qimg[i][j] - closest 
                if i+1<m_orig:
                    qimg[i+1][j] = qimg[i+1][j] + err * 3/8
                if j+1<n_orig:
                    qimg[i][j+1] = qimg[i][j+1] + err * 3/8
                if i+1<m_orig and j+1<n_orig:
                    qimg[i+1][j+1] = qimg[i+1][j+1] + err * 1/4
        return qimg
    
    def __init__(self,input_img,k_popularity_algo= 512):
        top_colors = self.colors_and_counts(input_img)
        topk_colors= top_colors.iloc[0:k_popularity_algo]
        topk_colors_list = topk_colors.index.tolist()
        KDSearchTree = self.getKdTree(topk_colors_list)
        self.inputImage = input_img
        self.KDSearchTree = KDSearchTree
        self.topk_colors_list = topk_colors_list
        print('performing quantization')
        self.quantized_image = self.get_quantized_img(input_img)
        self.dithered_image = self.dither_popularity()
        
    