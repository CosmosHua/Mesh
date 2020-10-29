# coding:utf-8
#!/usr/bin/python3
import random, os, cv2


################################################################################
def mesh_apart(mesh, clean, out): # sever mesh
    if not os.path.exists(out): os.makedirs(out)
    for im in os.listdir(mesh):
        net = os.path.join(out, im)
        cc = im[:im.rfind("_")]+".jpg"
        cc = cv2.imread(os.path.join(clean, cc))
        im = cv2.imread(os.path.join(mesh, im))
        cc = cc.astype(int)-im.astype(int)
        cv2.imwrite(net, 255-cc)

        im = cv2.imread(net, 0)
        '''for i in range(5,12):
            bk, C = 4*i+1, i # best: (45,6)
            cc = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bk, C)
            cv2.imwrite("_".join([net[:-4],str(bk),str(C),net[-4:]]), cc)
        '''
        cc = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 6)
        cv2.imwrite("_".join([net[:-4],net[-4:]]), cc)


################################################################################
if __name__ == '__main__':
    mesh_apart("Test", "Clean", "out")
