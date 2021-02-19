import pandas as pd
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from numpy import unique
from scipy.stats import entropy as scipy_entropy
from scipy import ndimage as ndi
import openslide
from openslide import OpenSlide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import cv2
import time

def get_subimages(img,cord):    
    subimages = []
    for r in cord:
        row_subimages = (img.read_region(r,0,(300,300)).convert('RGB'))
        row_subimages = cv2.cvtColor(np.array(row_subimages), cv2.COLOR_RGB2GRAY)
        subimages.append(row_subimages)
        
    return subimages

def shannon_entropy(image, base=2):
    _, counts = unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)

def compute_feats(image, kernels):
    accum = np.zeros_like(image)
    feats = np.zeros((len(kernels), 3), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        np.maximum(accum, filtered, accum)
        x = accum.mean()
        y = np.std(accum)
        z = shannon_entropy(accum)
        return x,y,z

def multiproc(iterable,krn_bank, ncores):
    import time
    from concurrent.futures import ProcessPoolExecutor
    import itertools
    cnt=0
    print("Starting ProcessPoolExecutor")
    begin = time.perf_counter()
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        results=[]
        generators=executor.map(compute_feats, iterable, itertools.repeat(krn_bank))
        while True:
            try:
                results.append(next(generators))
                #print(len(results))
            except (StopIteration, KeyboardInterrupt):
                break
    print(f"Ended in {time.perf_counter()-begin} seconds")
    return results

if __name__ == "__main__":
    image1=openslide.OpenSlide(r'ABCD3.svs')
    image2=openslide.OpenSlide(r'WLS.svs')

    df_full = pd.read_csv('Haralick_Features_QuPath_measurements_swap.tsv', sep='\t', header=0)
    df = df_full[['Centroid X µm','Centroid Y µm']].div(0.50119999999999998) 
    df = df.sub(150) 
    df = df.round(0).astype(int) 
    cord1= df.loc[0:28459]         
    cord2 = df.loc[28460::]

    cord1 = list(cord1[['Centroid X µm', 'Centroid Y µm']].itertuples(index=False, name=None))   
    cord2 = list(cord2[['Centroid X µm', 'Centroid Y µm']].itertuples(index=False, name=None))

    print("Getting ABCD3 subimages")
    ABCD3 = get_subimages(image1,cord1)
    print("Getting WLS subimages")
    WLS = get_subimages(image2,cord2)

    result = []

    kernel1 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 0.84, theta, 2, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel1.append(kernel)

    kernel2 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 1.68, theta, 4, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel2.append(kernel)
    
    kernel3 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 3.36, theta, 8, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel3.append(kernel)

    kernel4 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 7.72, theta, 16, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel4.append(kernel)

    kernel5 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 15.42, theta, 32, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel5.append(kernel)

    ncores=10
    featsABCD1=multiproc(ABCD3,kernel1,ncores)
    featsABCD2=multiproc(ABCD3,kernel2,ncores)
    featsABCD3=multiproc(ABCD3,kernel3,ncores)
    featsABCD4=multiproc(ABCD3,kernel4,ncores)
    featsABCD5=multiproc(ABCD3,kernel5,ncores)

    featsWLS1=multiproc(WLS,kernel1,ncores)
    featsWLS2=multiproc(WLS,kernel2,ncores)
    featsWLS3=multiproc(WLS,kernel3,ncores)
    featsWLS4=multiproc(WLS,kernel4,ncores)
    featsWLS5=multiproc(WLS,kernel5,ncores)
    print("Finished Gabor filtering")

    FeatWLS1 = pd.DataFrame(featsWLS1)
    FeatWLS2 = pd.DataFrame(featsWLS2)
    FeatWLS3 = pd.DataFrame(featsWLS3)
    FeatWLS4 = pd.DataFrame(featsWLS4)
    FeatWLS5 = pd.DataFrame(featsWLS5)
    FeatABCD1 = pd.DataFrame(featsABCD1)
    FeatABCD2 = pd.DataFrame(featsABCD2)
    FeatABCD3 = pd.DataFrame(featsABCD3)
    FeatABCD4 = pd.DataFrame(featsABCD4)
    FeatABCD5 = pd.DataFrame(featsABCD5)
    FeatWLS1 = pd.DataFrame(featsWLS1)
    FeatWLS2 = pd.DataFrame(featsWLS2)
    FeatWLS3 = pd.DataFrame(featsWLS3)
    FeatWLS4 = pd.DataFrame(featsWLS4)
    FeatWLS5 = pd.DataFrame(featsWLS5)

    FeaturesABCD = pd.concat([FeatABCD1,FeatABCD2,FeatABCD3,FeatABCD4,FeatABCD5],axis=1)
    FeaturesWLS = pd.concat([FeatWLS1,FeatWLS2,FeatWLS3,FeatWLS4,FeatWLS5],axis=1)
    FeaturesWLS.index += 28460

    Gabor2images = pd.concat([FeaturesABCD,FeaturesWLS],axis=0)

    Gabor2images.columns = Gabor2images.columns = ["Gabor sigma=0.84, wavelength=2, gamma=0.5 psi=0 Mean","Gabor sigma=0.84, wavelength=2, gamma=0.5 psi=0 SD","Gabor sigma=0.84, wavelength=2, gamma=0.5 psi=0 Entropy","Gabor sigma=1.68, wavelength=4, gamma=0.5 psi=0 Mean","Gabor sigma=1.68, wavelength=4, gamma=0.5 psi=0 SD","Gabor sigma=1.68, wavelength=4, gamma=0.5 psi=0 Entropy","Gabor sigma=3.36, wavelength=8, gamma=0.5 psi=0 Mean","Gabor sigma=3.36, wavelength=8, gamma=0.5 psi=0 SD","Gabor sigma=3.36, wavelength=8, gamma=0.5 psi=0 Entropy","Gabor sigma=7.72, wavelength=16, gamma=0.5 psi=0 Mean","Gabor sigma=7.72, wavelength=16, gamma=0.5 psi=0 SD","Gabor sigma=7.72, wavelength=16, gamma=0.5 psi=0 Entropy","Gabor sigma=15.42 wavelength=32, gamma=0.5 psi=0 Mean","Gabor sigma=15.42 wavelength=32, gamma=0.5 psi=0 SD","Gabor sigma=15.42 wavelength=32, gamma=0.5 psi=0 Entropy"]
    #Gabor2images.columns=[ "Gabor energy bandwidth =1.4 wavelength=2 Mean", "Gabor energy bandwidth =1.4 wavelength=2 SD", "Gabor energy bandwidth =1.4 wavelength=2 Entropy ",]
    dfswap = df_full.iloc[:,0:7]
    Gabor2images = pd.concat([dfswap,Gabor2images],axis=1) # add coordinates

    from sklearn.decomposition import PCA
    import skfuzzy as fuzz

    Features = Gabor2images.iloc[:,7::]

    scaler = StandardScaler()
    scaler.fit(Features)
    scaled_data = scaler.transform(Features)
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    scaled_data.shape
    x_pca.shape

    fpc = x_pca[:,0]
    spc = x_pca[:,1]
    #tpc = x_pca[:,2] and so on
    b = pd.DataFrame({'fpc':fpc, 'spc':spc})

    # plot the obtained two components to determine the number of clusters based on FPC parameter looping from 2 to 10 number of centers 
    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    allPCA = np.vstack((b['fpc'], b['spc']))
    fpcs = []

    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen'] 

    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            allPCA, ncenters, 2, error=0.005, maxiter=1000, init=None)

        # Store fpc values for later plots
        fpcs.append(fpc)

        # Plot assigned clusters, for each data point in training set
        cluster_memberships = np.argmax(u, axis=0)
        for j in range(ncenters):
            ax.plot(b['fpc'][cluster_memberships == j],
                    b['spc'][cluster_memberships == j],'.', color=colors[j])
        # Mark the center of each fuzzy cluster
        ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
        ax.axis('off')
    plt.savefig("figure1.png")

    # build a 7-cluster model assigning coefficients randomly
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        allPCA, 7, 2, error=0.002, maxiter=10000, seed=32)
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y']
    fig2, ax2 = plt.subplots()
    ax2.set_title('Trained model')
    for j in range(7):
        ax2.plot(allPCA[0, u_orig.argmax(axis=0) == j],
                allPCA[1, u_orig.argmax(axis=0) == j], 'o',
                label='series ' + str(j))
    ax2.legend()
    plt.savefig("figure2.png")

    # generate uniformly sampled data into the pre-existing model and assign a cluster 
    # corresponding to the highest membership value to the degree of being in a given cluster. 

    allPCA = np.column_stack((b['fpc'], b['spc']))

    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        allPCA.T, cntr, 2, error=0.002, maxiter=10000, seed = 32)

    cluster_membership = np.argmax(u, axis=0) 

    fig3, ax3 = plt.subplots()
    ax3.set_title('Random points classifed according to known centers')
    for j in range(7):
        ax3.plot(allPCA[cluster_membership == j, 0],
                allPCA[cluster_membership == j, 1], 'o',
                label='series ' + str(j))
    ax3.legend()
    plt.savefig("figure3.png")

    Gabor2images['Class'] = cluster_membership

    Gabor2images.to_csv('Gaborfilters', sep='\t', index=False) # save Gabor features, coordinates, class


    PCA1= Gabor2images.loc[0:28459] # for image 1
    PCA2 = Gabor2images.loc[28460:47737] # for image 2
    import seaborn as sns

    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00'] 

    fig, ax = plt.subplots(figsize=(20, 20))
    f = sns.scatterplot(x="Centroid X µm", y="Centroid Y µm",
                    hue = 'Class', palette=colors,
                    data=PCA1)
    ax.set_title('Clustering after PCA transformation')
    f.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    ax=plt.gca()                            
    ax.set_ylim(ax.get_ylim()[::-1])        
    ax.xaxis.tick_top()                     
    plt.savefig("figure4.png")

    fig, ax = plt.subplots(figsize=(20, 20))
    g = sns.scatterplot(x="Centroid X µm", y="Centroid Y µm",
                    hue = 'Class', palette=colors,
                    data=PCA2)

    g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis 
    plt.savefig("figure5.png") 

    # import the original coordinates for ABCD3 image for rotating the figure
    df_old = pd.read_csv('Haralick_Features_QuPath_measurements (1).tsv', sep='\t', header=0)
    df_old['Class'] = cluster_membership
    df_old.to_csv('Haralick_Features_QuPath_measurements.tsv', sep = '\t', index = False)


    PCA=df_old.loc[0:28459] # for image 1
    import seaborn as sns

    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00']                    

    fig, ax = plt.subplots(figsize=(20, 20))
    g = sns.scatterplot(x="Centroid X µm", y="Centroid Y µm",
                    hue = 'Class', palette=colors,
                    data=PCA)

    g.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis  
    plt.savefig("figure6.png")