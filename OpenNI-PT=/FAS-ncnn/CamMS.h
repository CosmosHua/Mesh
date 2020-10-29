#pragma once


#ifndef __CAM_RGBD_H__
#define __CAM_RGBD_H__


#include <ctime>
#include <string>
#include <direct.h>
#include <OpenNI.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


typedef enum
{
    OPENNI_INITIAL_FAILED =     1,
    DEVICE_OPEN_FAILED =        2,
    UNSUPPORTED_SENSOR_TYPE =   3,
    STREAM_COLOR_INIT_FAILED =  4,
    STREAM_COLOR_START_FAILED = 5,
    STREAM_DEPTH_INIT_FAILED =  6,
    STREAM_DEPTH_START_FAILED = 7,
    STREAM_IR_INIT_FAILED =     8,
    STREAM_IR_START_FAILED =    9
} CAM_RGBD_ERROR_CODE;


class Cam
{
private:
    void* m_device;
    void* m_stream_color;
    void* m_stream_depth;
    void* m_stream_ir;
    int OpenDevice();
    int CloseDeivce();
    void Destroy(void* &m_stream);

public:
    Cam();
    ~Cam();
    int TimeOut = 150; //ms
    int InitStream(const int CamType);
    int CloseStream(const int CamType);
    bool isValid(const int CamType)const;
    //function pointer, which transforms openni::VideoFrameRef to cv::Mat
    cv::Mat GetFrame(const int CamType, cv::Mat(*pf)(const openni::VideoFrameRef&)=NULL)const;
};//end Cam


cv::Mat IRFrame(const openni::VideoFrameRef& Frame);
cv::Mat RGBFrame(const openni::VideoFrameRef& Frame);
cv::Mat DepthFrame(const openni::VideoFrameRef& Frame);

const std::string path = "./";
void Superpose(cv::Mat& im, std::vector<cv::Mat> Img, int sup);
void SaveImage(const cv::Mat& im, int& tid, int& i, int f2p);
void SaveVideo(const cv::Mat& im, cv::VideoWriter& out);
void ExtractVideo(const std::string tid, int mod=1);


#endif //__CAM_RGBD_H__
