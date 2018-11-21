import os
import cv2
import numpy as np

import config


cfg = config.Config()



def saveImg(img, fname, type="normal"):
    if type == "normal":
        s_path = os.path.join(cfg.stream, cfg.nrl)
        ns_path = os.path.join(cfg.nostream, cfg.nrl)
    elif type == "dursen":
        s_path = os.path.join(cfg.stream, cfg.drn)
        ns_path = os.path.join(cfg.nostream, cfg.drn)
    elif type == "type1":
        s_path = os.path.join(cfg.stream, cfg.t1)
        ns_path = os.path.join(cfg.nostream, cfg.t1)
    elif type == "type2":
        s_path = os.path.join(cfg.stream, cfg.t2)
        ns_path = os.path.join(cfg.nostream, cfg.t2)
    else:
        print("There is no such class")
    
    ext_name = fname.split(" ")[2]
    if ext_name[7] == "a":         # image with blood stream
        cv2.imwrite(os.path.join(s_path, fname), img)
    elif ext_name[7] == "b":
        cv2.imwrite(os.path.join(ns_path, fname), img)
        

def extImg(file):
    img = cv2.imread(file)
    img = img[566:975, 99:955]
    
    return img


if __name__ == "__main__":
    path = "F:/OCT/sent 11-12/type 2 CNV-11/"
    flist = os.listdir(path)
    for fname in flist:
        print(fname)
        file = os.path.join(path, fname)
        img = extImg(file)
        saveImg(img, fname, type="type2")
    