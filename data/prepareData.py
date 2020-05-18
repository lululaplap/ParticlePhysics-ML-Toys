import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

"""
Inputs: Image Size x (int),Image Size y (int),  classification (int), input directory (str), output filename (str)
"""
def main(argv):
    print(argv)
    myImgSize = [0,0]
    myImgSize[0] = int(argv[1])
    myImgSize[1] = int(argv[2])
    # N = int(argv[2])#29
    classification = int(argv[3])
    inputdir = argv[4]
    fname = argv[5]
    outfile = open(fname,'a')
    files = os.listdir(inputdir)
    xmax = myImgSize[0]
    ymax  = myImgSize[1]
   # for i in range(0,len(files)):
      #  pad = [0,0]
     #   if "ADC" in files[i] and ".csv" in files[i]:
    #        f = np.genfromtxt(inputdir + files[i],delimiter=",",dtype=int,skip_header=2,skip_footer=2)
   #         if f.shape[0] > xmax:
  #              xmax = f.shape[0]
 #           if f.shape[1] > ymax:
#                ymax = f.shape[1]

    for i in range(0,len(files)):
        pad = [0,0]
        if "ADC" in files[i] and ".csv" in files[i]:
            try:
                f = np.genfromtxt(inputdir + files[i],delimiter=",",dtype=int,skip_header=2,skip_footer=2)
                print(f.shape)
                # plt.imshow(f,cmap='Greys', interpolation='none')
                # plt.show()
                f = f[:,0:-3]
                pad[0] = max(xmax-f.shape[0],0)
                pad[1] = max(ymax-f.shape[1],0)

                pad = tuple(pad)
                #plt.imshow(f)
                # print(pad)
                f_p = np.pad(f,((0,pad[0]),(0,pad[1])),'constant',constant_values=0)
                fmax = np.max(f_p)
                fmin = np.min(f_p)
                ran = fmax-fmin

                f_ps = np.array(((f_p+abs(fmin))/ran)*255).astype(np.uint8)
                #plt.imshow(f_ps,cmap='Greys', interpolation='none')
                #plt.show()
                #plt.imshow(f_ps)
                # plt.show()
                if np.mean(f_ps)<200:#np.sum(f_ps==0)>500:
                    im = Image.fromarray(f_ps)
                    im.save(inputdir+"ADC{}.png".format(i))

                    outfile.write(inputdir+"ADC{}.png, {}\n".format(i,classification))
            except:
                print("fail")

main(sys.argv)
