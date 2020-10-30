if __name__ == "__main__":
    import libglcmsw

    whole=libglcmsw.SlidingWindow("./input.tif",13,"homogeneity",downscale=1)
    whole.render("tiling",tilesz=125)
    whole.savetif("outputclassraster", type="uint8")

    downscale=libglcmsw.SlidingWindow("./input.tif",13,"homogeneity",downscale=2)
    downscale.render("raster",rowssave=80)
    downscale.savetif("outputclassraster", type="uint8")
