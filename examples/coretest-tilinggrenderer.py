def singlerun(ncores,iter):
    import libglcmsw
    import openslide
    from skimage import io as si
    import numpy as np
    import shutil
    import time
    
    #Input data for testing function
    img = libglcmsw.io.openimg.load("./input.tif")
    prop='homogeneity'
    tmpdir=f"./ischdown30x-tilesz125-{iter}"
    WINDOWSZ=13
    TILESZ=125
    workers=ncores
    output="./outputdown30-tile125.tif"
    #End of input data
    
    w, h =libglcmsw.tiling.tilegen.tilegendisc(img,TILESZ,WINDOWSZ//2,tmpdir=tmpdir)
    listoftiles=libglcmsw.tiling.tilegen.gettileslistfull(img, TILESZ, WINDOWSZ//2)
    print(listoftiles)
    if not listoftiles:
        print("No tiles to render.")
    else:
        begin = time.perf_counter()
        libglcmsw.render.cpu.tilerenderlist(tmpdir,listoftiles,WINDOWSZ,ncores=workers, prop=prop)
        end = time.perf_counter()
        libglcmsw.tiling.reconstruct.fillblanks(tmpdir, img, TILESZ, WINDOWSZ // 2,
                                                libglcmsw.tiling.tilegen.gettileslistfull(img, TILESZ, WINDOWSZ // 2),
                                                prop=prop)
    final=libglcmsw.tiling.reconstruct.allconcat(tmpdir,w,h)
    si.imsave(output, final.astype(np.float32))
    shutil.rmtree(tmpdir)
    return end-begin

if __name__ == "__main__":
    import sys
    log = open("corestest","w")
    orig_stdout = sys.stdout
    sys.stdout = log
    times=[]
    #Number of iterations, minimum amount of cores, maximum amount of cores
    numiters=5
    coresmin=1
    coresmax=12 #inclusive
    #End of parameters for number of tests
    for ncores in range(coresmin,coresmax+1):
        tmp=[]
        print(f"Using {ncores} cores:")
        for i in range(numiters):
            tmp.append(singlerun(ncores,i))
        times.append(tmp)
    for i in range(len(times)):
        print(f"Times to complete for {i+coresmin} cores: {times[i]}")
        #log.write(f"Times to complete for {i+coresmin} cores: {times[i]}")
    log.close()
    sys.stdout = orig_stdout
