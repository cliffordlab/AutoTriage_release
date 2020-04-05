# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:59:48 2020

@author: chait
"""
import numpy as np

def get_hist(img):
    
    #img = cv2.imread(img)
    size1,size2,_ = img.shape
    R = img[:,:,0].reshape(size1*size2)
    G = img[:,:,1].reshape(size1*size2)
    B = img[:,:,2].reshape(size1*size2)
    #bins = [15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255]
    

    R_hist, R_bins = np.histogram(R, bins=6)
    G_hist, G_bins = np.histogram(G, bins=6)
    B_hist, B_bins = np.histogram(B, bins=6)

    R_hist = [a/(size1*size2) for a in R_hist]
    G_hist = [a/(size1*size2) for a in G_hist]
    B_hist = [a/(size1*size2) for a in B_hist]
    
    data = np.concatenate((R_hist,G_hist),axis=0)
    data = np.concatenate((data,B_hist),axis=0) 
    
    return data

def predict_cyanosis(clf, img, lip_coords):
    
    pred_all_ppl = []
    for i in range(10):
        if lip_coords[i][1][0]<lip_coords[i][0][0] and lip_coords[i][1][1]<lip_coords[i][0][1]:
            roi = img[lip_coords[i][1][1]:lip_coords[i][0][1], lip_coords[i][1][0]:lip_coords[i][0][0]]
        elif lip_coords[i][1][0]>lip_coords[i][0][0] and lip_coords[i][1][1]<lip_coords[i][0][1]:
            roi = img[lip_coords[i][1][1]:lip_coords[i][0][1], lip_coords[i][0][0]:lip_coords[i][1][0]]
        elif lip_coords[i][1][0]<lip_coords[i][0][0] and lip_coords[i][1][1]>lip_coords[i][0][1]:
            roi = img[lip_coords[i][0][1]:lip_coords[i][1][1], lip_coords[i][1][0]:lip_coords[i][0][0]] 
        else:
            roi = img[lip_coords[i][0][1]:lip_coords[i][1][1], lip_coords[i][0][0]:lip_coords[i][1][0]]
            
        s1,s2,_ = roi.shape
        if s1*s2 != 0:
            hist = get_hist(roi)
            pred = clf.predict([hist])
            pred_all_ppl.append(pred)
        else:
            pass

    return pred_all_ppl