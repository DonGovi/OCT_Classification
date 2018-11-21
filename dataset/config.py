import os


class Config(object):
    
    def __init__(self):
        self.root = "F:/OCT/classification/"
        self.stream = os.path.join(self.root, "stream")
        self.nostream = os.path.join(self.root, "non_stream")
        self.nrl = "normal/"
        self.drn = "drusen/"
        self.t1 = "type1/"
        self.t2 = "type2/"
        
