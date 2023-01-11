import face_recognition as fr
import rawpy
import imageio
from os import listdir
from os.path import exists
from os import mkdir
import cv2
from shutil import copy2

# make in rust
def img_to_classify_to_jpg(imgDir: str, target: str) -> None:
    '''imgDir takes in the directory of the folder where the photos are. the target is the directory
    to the folder that the temp files will be stored'''
    photo_dir = imgDir
    imgDir = [i for i in listdir(imgDir) if i[0] != "." and not i.endswith(".xmp")]
    if not exists(target):
        mkdir(target)

    for image in imgDir:
        if image.endswith(".jpg"): 
            copy2(src=f"{photo_dir}/{image}", dst=f"{target}/{image}")
            continue
        file_name = image.split(".")[0]
        with rawpy.imread(f"{photo_dir}/{image}") as raw:
            rgb = raw.postprocess()
        imageio.imsave(f"{target}/{file_name}.jpg", rgb)

        img = cv2.imread(f"{target}/{file_name}.jpg")
        width, height = img.shape[0], img.shape[1]

        width = int(width/1)
        height = int(height/1)
        divider = 1
        while (width/divider) > 2500 and (height/divider) > 2500:
            divider+=1
        
        width = int(width/divider)
        height = int(height/divider)

        new_img = cv2.resize(img, (height,width))
        cv2.imwrite(f"{target}/{file_name}.jpg" , new_img)
