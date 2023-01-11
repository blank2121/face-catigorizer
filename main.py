import face_recognition as fr
import cv2
from tkinter import Tk
from tkinter.filedialog import askdirectory
from os import listdir
from os.path import exists
import pickle
import FaceFilter


def pickle_dump(file_name: str, var) -> None:
    with open(f"{file_name}.pickle", "wb") as f:
        pickle.dump(var, f)
        
def pickle_load(file_name: str):
    with open(f"{file_name}.pickle", "rb") as f:
        data = pickle.load(f)
    return data

def known_face_encoding(people: list[str], dir: str, end_info = True) -> list[str]:
    encoded_faces = []
    if end_info:
        last_elem = people[-1]
        people.pop(-1)
        for img in people:
            target_img = fr.load_image_file(f"{dir}/{img}")
            target_encoding = fr.face_encodings(target_img)
            if target_encoding == []:
                encoded_faces.append([])
                continue
            encoded_faces.append(target_encoding[0])
        encoded_faces.append(last_elem)
        return encoded_faces
    else:
        for img in people:
            target_img = fr.load_image_file(f"{dir}/{img}")
            target_encoding = fr.face_encodings(target_img)
            if target_encoding == []:
                encoded_faces.append([])
                continue
            encoded_faces.append(target_encoding[0])
        return encoded_faces

Tk().withdraw()

# user prompting and checking if the pickled file is there or not
if exists("./ref.pickle"):
    inp = input("reference images already exist please enter, \"prompt\" to be prompted with the selection menu or just press enter/return to use the existing reference data.\n")
    if inp == "": pass
    else:
        dir = askdirectory()
        print(f"testing: {dir}")
        pic_list = listdir(dir)
        
        #ignoring dot files
        pic_list = [i for i in pic_list if i[0] != "."]
        pic_list.append(dir)
        # function to make photos to jpg in temp file
        pickle_dump("ref",known_face_encoding(pic_list, dir))
        print("new reference images have been saved and properly encoded\n")
else:
    print("there are no reference images for the program to use, please select a folder on reference images \n")
    dir = askdirectory()
    pic_list = listdir(dir)
    pic_list = [i for i in pic_list if i[0] != "."]
    pic_list.append(dir)
    # function to make photos to jpg in temp file
    pickle_dump("ref",known_face_encoding(pic_list, dir))
    
print("now please select the folder with the images you want to be classified by person\n")    

# prompting user to select folder with pictures
dir = askdirectory()

print("\n\n")
print("starting the computation")
print("\n\n")

print("start convert to jpg\n\n")

FaceFilter.img_to_classify_to_jpg(dir, "./temp")

print("finished the jpg convertion\n\n")

pic_list: list[str] = listdir("./temp")
pic_list: list[str] = [i for i in pic_list if i[0] != "." and not i.endswith(".xmp")]

print("finished making the pic_list\n\n")

print(pic_list)
#imgs = images to classify
imgs: list[str] = known_face_encoding(pic_list, "./temp", end_info=False)

print("finished encoding the images\n\n")

name_list = pickle_load("./ref")
name_list = listdir(name_list[-1])
ref_img = pickle_load("./ref")
ref_img.pop(-1)

print("starting the for loop\n\n")

for idx, img in enumerate(imgs):
    if img == []: continue
    print(f"{pic_list[idx]} {[person[1] for person in enumerate(name_list) if fr.compare_faces(ref_img, img)[person[0]]]}")


print("program finished\n\n")