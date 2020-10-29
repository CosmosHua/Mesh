#pragma once


#ifndef __MTCNN_NCNN_H__
#define __MTCNN_NCNN_H__


//#define NOMINMAX

#include <map>
#include <math.h>
#include <time.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <net.h>

using namespace std;


struct BBox
{
    int x1, y1;
    int x2, y2;
    float Area;
    float Score;
    float Pos[4];
    float kPoint[10];
};//end BBox


class MTCNN
{
public:
    MTCNN(const string &model_path);
    MTCNN(std::vector<std::string> params, std::vector<std::string> bins);
    ~MTCNN() { Pnet.clear(); Rnet.clear(); Onet.clear(); }

    void SetMinFace(int mi) { minsize = mi; }
    std::vector<BBox> Detect(ncnn::Mat &im);

private:
    ncnn::Mat img;
    ncnn::Net Pnet, Rnet, Onet;

    int img_w, img_h, minsize = 40;
    const int MIN_DET_SIZE = 12;
    const float facetor = 0.709f;
    const float threshold[3] = {0.8f, 0.8f, 0.6f};
    const float nms_threshold[3] = {0.5f, 0.7f, 0.7f};
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    std::vector<BBox> BBox1, BBox2, BBox3;

    void PNet(); void RNet(); void ONet();
    void NMS(vector<BBox> &Box, float IOU_threshold, string type="Union");
    void Refine(vector<BBox> &Box, int height, int width, bool square=true);
    void GenBBox(ncnn::Mat prob, ncnn::Mat location, vector<BBox> &Box, float scale);
};//end MTCNN


#endif //__MTCNN_NCNN_H__

