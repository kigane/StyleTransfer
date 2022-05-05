# -----------------------------change class to idx----------------------------

# for scribble only


def scribble2idx(nake):
    cls_lst = ["pineapple", "strawberry", "basketball", "chicken", "cookie",
               "cupcake", "moon", "orange", "soccer", "watermelon"]
    try:
        A_nake = cls_lst.index(nake)
    except ValueError:
        raise ValueError(
            "The input category name {} is not recognized by scribble.".format(
                nake)
        )
    return A_nake


# for sketcycoco only
def sketchycoco2idx(nake):
    indices = [2, 3, 4, 5, 10, 11, 17, 18, 19, 20, 21, 22, 24, 25]
    idx = indices.index(int(nake))
    return int(idx)


# ----------------------------------------------------------------------------

IMG_EXTENSIONS = [".jpg", ".JPG", 
                  ".jpeg", ".JPEG",
                  ".png", ".PNG", 
                  ".ppm", ".PPM", 
                  ".bmp", ".BMP",]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
