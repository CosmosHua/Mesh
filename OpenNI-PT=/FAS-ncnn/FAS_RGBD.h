#pragma once


#ifndef __FAS_RGBD_NCNN_H__
#define __FAS_RGBD_NCNN_H__


#include <net.h>
#include <time.h>
#include <string>
#include <iostream>
#include <opencv.hpp>

using namespace std;

#define USE_MEM_MODEL 1

#if USE_MEM_MODEL
#include "RGBD_id.h"
#define MODEL_NAME_ID RGBD_param_id
#endif //USE_MEM_MODEL


class FAS_RGBD
{
public:
    FAS_RGBD(int c, string model_dir);
    ~FAS_RGBD() { delete[] p; net.clear(); }
    //F=Face's depth|color(RGB/BGR) -> FAS result
    ncnn::Mat Discern(const cv::Mat& F, bool BGR=true);

    static uint16_t Min2(cv::Mat F); //F=depth->find 2nd_min
    static cv::Mat DepthClip(cv::Mat F); //F=depth->clip depth
    static float RatioN0(cv::Mat F); //F=depth->non-zero ratio
    
    //inc = 4 if (cmb in[4,13,31]) else (1 if cmb in[1,2] else 3)
    //auto inc = [](int c=cmb){ c = 4*(c>3)+(c==1||c==2); return c+3*(c==0); };
    static int inc(int c) { c = 4*(c>3)+(c==1||c==2); return c+3*(c==0); }
    //ncnn::Mat::{PIXEL_RGB=1, PIXEL_BGR=2, PIXEL_GRAY=3, PIXEL_RGBA=4}
    //auto typ = [](int c=cmb){ c = 4*(c>3)+3*(c==1||c==2); return c+(c==0); };
    static int typ(int c) { c = 4*(c>3)+3*(c==1||c==2); return c+(c==0); }
    
private:
    int cmb = 13, inp = 4, tp = 4;
    ncnn::Net net; float *p = NULL;

    //F=depth(CV_16UC1) -> color(CV_8UC3|RGB)
    static cv::Mat Encode(const cv::Mat& F, int md=1);
    //x=depth(uint16_t) -> color(Vec3b|RGB)
    static cv::Vec3b Encode(uint16_t x, int md=1);
    //F=color(Vec3b|RGB) -> depth(uint16_t)
    static uint16_t Decode(const cv::Vec3b& F, int md=1);
    //F=color(CV_8UC3|RGB) -> depth(CV_16UC1)
    static cv::Mat Decode(const cv::Mat& F, int md=1);

    //F=depth|color(RGB) -> combination(RGB|A)
    static cv::Mat Superpose(const cv::Mat& F, int c);
    float* MeanStd(int c); //substract_mean_normalize

    //Resize->(256,256), then Center-Crop->(wh,wh)
    static cv::Mat ResizedCrop(const cv::Mat& F, int wh=224);
};//end FAS_RGBD

int Max(const ncnn::Mat& F, float& mx); //1-dim
ncnn::Mat Softmax_(ncnn::Mat& F); //1-dim, inplace
void ShowMat(const ncnn::Mat& F, int cmb=5); //F:[C,H,W]


#endif //__FAS_RGBD_NCNN_H__
