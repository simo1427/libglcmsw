if __name__ == "__main__":
    from libglcmsw import SlidingWindow

    whole=SlidingWindow("./input.tif",13,"homogeneity",downscale=1)
    whole.render("tiling",tilesz=125)
    whole.savetif("outputclasstiling", type="float32")

    downscale=SlidingWindow("./input.tif",13,"homogeneity",downscale=2)
    downscale.render("raster",rowssave=80)
    downscale.savetif("outputclassraster", type="uint8")
