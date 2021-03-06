if __name__ == "__main__":
    import libglcmsw
    from skimage import io as si

    #Input data
    img = libglcmsw.io.openimg.load("./input.tif")
    mask = libglcmsw.io.openimg.load("./mask.tif")
    prop="homogeneity"
    tmpdirimg="./tilingtmp-img"
    tmpdirmask = "./tilingtmp-mask"
    WINDOWSZ=13
    TILESZ=125
    workers=5
    output=f"./outputmasked-colmajor.tif"
    angle=0
    distance=1
    #End of input data

    numtiles = libglcmsw.tiling.tilegen.tilegendisc(img,TILESZ,WINDOWSZ//2,tmpdir=tmpdirimg)
    libglcmsw.tiling.tilegen.tilegendisc(mask,TILESZ,WINDOWSZ//2,tmpdir=tmpdirmask)

    import numpy as np
    import os
    w, h = numtiles
    listoftilestmp=[]
    for i in range(w):
        for j in range(h):
            try:
                tmp = np.load(tmpdirmask+f"/{i}_{j}.npy")
                if 255 in tmp:
                    listoftilestmp.append((i,j))
                    os.remove(tmpdirmask+f"/{i}_{j}.npy")
                else:
                    os.remove(tmpdirimg+f"/{i}_{j}.npy")
                    os.remove(tmpdirmask+f"/{i}_{j}.npy")
            except:
                pass
    listoftiles=libglcmsw.io.crashrecovery.getunprocessedtiles(tmpdirimg,listoftilestmp)
    print(listoftiles)
    if not listoftiles:
        print("No tiles to render. Stopping")
    else:
        libglcmsw.render.cpu.tilerenderlist(tmpdirimg,listoftiles,WINDOWSZ,ncores=workers, prop=prop, angle=angle, distance=distance)
    libglcmsw.tiling.reconstruct.fillblanks(tmpdirimg, img, TILESZ, WINDOWSZ // 2,
                                            libglcmsw.tiling.tilegen.gettileslistfull(img, TILESZ, WINDOWSZ // 2),
                                            prop=prop, angle=angle, distance=distance)
    final=libglcmsw.tiling.reconstruct.allconcat(tmpdirimg,w,h)
    si.imsave(output, final.astype(np.float32))
    import shutil
    shutil.rmtree(tmpdirimg)
    shutil.rmtree(tmpdirmask)