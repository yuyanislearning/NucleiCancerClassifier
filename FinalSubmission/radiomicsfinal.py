# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:08:02 2020

@author: hwz62

Run instructions: The function createfeatures() is the primary one and calls generatemask()

Input: tilefile and maskfile both use the fullpath to the tile image and mask image, 
respectively. tilefile should point to a .png or a .tif. maskfile should point 
to a .csv, which is a N-by-4 array from Samir's segmentation algorithm.

Returns: a 816-by-1 ndarray of radiomics features.

Dependencies: SimpleITK, PyRadiomics, pandas, numpy
"""

#supressing the warnings
import logging
# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

#import os
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import numpy as np

def generatemask(maskpts,id):
    list=maskpts.loc[maskpts['mask_id']==id]
    mask=np.zeros([512,512],dtype='int')
    mask[list['x'],list['y']]=1
    return mask

def createfeatures(tilefile,maskfile):
    features=['original_shape2D_Elongation', 'original_shape2D_MajorAxisLength', 'original_shape2D_MaximumDiameter', 'original_shape2D_MeshSurface', 'original_shape2D_MinorAxisLength', 'original_shape2D_Perimeter', 'original_shape2D_PerimeterSurfaceRatio', 'original_shape2D_PixelSurface', 'original_shape2D_Sphericity', 'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength']
    featurelist=[]
    for i in features:
        featurelist.append(i+'_count')
        featurelist.append(i+'_mean')
        featurelist.append(i+'_std')
        featurelist.append(i+'_min')
        featurelist.append(i+'_25')
        featurelist.append(i+'_50')
        featurelist.append(i+'_75')
        featurelist.append(i+'_max')
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    #for f in range(1000):
    #tilefolder=os.path.join(os.getcwd(),'masks')
    #tilepathlist=os.listdir(tilefolder)
    #tilefile=os.path.join(os.getcwd(),'tiles_rois','normalized',tilepathlist[f][:-4]+'.png')
    #maskfile=os.path.join(os.getcwd(),'maskimg',tilepathlist[f][:-4]+'.png')
    #maskfolder=os.path.join(os.getcwd(),'masks')
    #maskpathlist=os.listdir(maskfolder)
    #maskfile=os.path.join(os.getcwd(),'masks',maskpathlist[f])
    maskpts=pd.read_csv(maskfile)
    tiledf=pd.DataFrame()
    for i in range(max(maskpts['mask_id'])):
        image = sitk.ReadImage(tilefile, sitk.sitkInt8)
        maskarr=generatemask(maskpts,i)
        #mask = sitk.ReadImage(maskfile, sitk.sitkInt8)
        mask = sitk.GetImageFromArray(maskarr)
        result = extractor.execute(image, mask)
        var=list(float(result[k]) for k in features)
        tiledf[i]=var
    tiledf=tiledf.transpose()
    tiledf.columns=features
    statdf=pd.DataFrame()
    for j in features:
        featvec=tiledf[j].describe()
        statdf[j]=featvec
    statdf=statdf.transpose()
    featflat=statdf.values.flatten()
    return featflat
