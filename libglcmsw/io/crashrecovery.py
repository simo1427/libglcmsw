import os
import numpy as np
"""
func getunprocessedtiles:
    returns list of tuples - tiles which have not been previously processed for unknown reasons

    Arguments:
    dpath (string) - path to directory containing all tiles (processed and unprocessed)
    tileslist - list of tuples - the selection of tiles which are chosen to be processed, given by libglcmsw.tiling.tilegen.gettileslistdir or libglmcsw.tiling.tilegen.gettileslistfull

    Process:
    checks through all files in the specified directory;
    parses the id of the processed tiles through the means of its name;
    tries to remove the tiles which have been processed from the list of all tiles
"""
def getunprocessedtiles(dpath, tileslist):
    for fname in sorted(os.listdir(dpath)):
        if fname[0]=='g':
            tileid=(fname[1:-4]).split("_")
            try:
                tileslist.remove((int(tileid[0]),int(tileid[1])))
            except:
                pass
    return tileslist

"""
func rasterrecovery:
    returns a tuple, containing the recovered array (if no crashes have been present, it returns an empty array ready for sliding image generation) and the number of non-empty lines in the recovered array

    Arguments:
    recoveryfile - path to temporary file
    ri - number of rows in the image
    rj - number of column in the image

    Process:
    initialize the array, in which the recovery file will be stored
    try to load the recovery file
    if it is not found, do not change the contents of the array
    while there are rows different from all zeros:
        add 1 to the counter of non-empty rows in the array
    return the tuple

    Usage:
    libglcmsw.render.cpu.rasterrender
"""
def rasterrecovery(recoveryfile,ri,rj):
    glcm_hom=np.zeros((ri,rj))
    print("Recovering from any previous crashes:")
    try:
        glcm_hom = np.load(recoveryfile)
        print("Successfully recovered")
    except FileNotFoundError:
        print("No recovery file. Ignore if there has not been a previous crash.")
    print("Array file read!")
    k = 0#number of rows in temporary file
    while (glcm_hom[k]).all() != (np.zeros((ri))).all():
        k+=1
    return (glcm_hom, k)
# if __name__ == "__main__":
#     print("This is only one part of the program. Please run main.py. \nExiting")
#     exit()