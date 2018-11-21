import os

import pandas as pd


def buildLabel(path):
    label = pd.DataFrame(columns=["id", "class"])
    dirlist = os.listdir(path)    #[drusen, normal, type1, type2]
    for dir in dirlist:
        dpath = os.path.join(path, dir)
        flist = os.listdir(dpath)
        for fname in flist:
            print(dir, fname)
            if dir == "normal":
                cls = 0
            elif dir == "drusen":
                cls = 1
            elif dir == "type1":
                cls = 2
            elif dir == "type2":
                cls = 3
            row = pd.DataFrame({"id":fname.split(".")[0], "class":cls}, index=["0"])
            label = label.append(row, ignore_index=True)
    
    label.to_csv(os.path.join(path, "label.csv"), header=True, index=False)
    

if __name__ == "__main__":
    path = "F:/OCT/classification/stream/"
    buildLabel(path)