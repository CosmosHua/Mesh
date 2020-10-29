# coding:utf-8
#!/usr/bin/python3


##################################################################
# Intersection Over Union:
def IOU(rec1, rec2): # rec = (x,y,w,h,score)
    int_x = min(rec1[0]+rec1[2],rec2[0]+rec2[2]) - max(rec1[0],rec2[0])
    int_y = min(rec1[1]+rec1[3],rec2[1]+rec2[3]) - max(rec1[1],rec2[1])
    int_x = max(int_x, 0); int_y = max(int_y, 0)
    Intersect = int_x * int_y
    Union = (rec1[2]*rec1[3] + rec2[2]*rec2[3]) - Intersect
    return Intersect/Union


# Find max sum of sub_list
def MaxSub(ls): # ls = integer list
    mx = [ls[0],(0,1)]; N = len(ls)
    for i in range(N): # i=start
        for j in range(i+1,N+1): # j=end
            nw = sum(ls[i:j])
            if nw>mx[0]: mx = [nw,(i,j)]
            elif nw==mx[0]: mx += [(i,j)]
    return mx


##################################################################
if __name__ == '__main__':
    rec1 = (0,0,5,5,0); rec2 = (2,3,5,5,0)
    print(IOU(rec1, rec2))
    ss = [-11, -4, 5, -2, 3, -5, 14, -14, 1, -6, 14, 1]
    # MaxSub(ss); MaxSub(ss*2)
