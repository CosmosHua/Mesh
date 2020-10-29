#include "MTCNN.h"
#include "FAS_RGBD.h"
#include "CamMS.h"


////////////////////////////////////////////////////////////////////////////////
//MTCNN: F=color(BGR) -> BBox of Faces
std::vector<BBox> GetFaceBox(const cv::Mat& F, string dir, int mi=50)
{
    const int tp = ncnn::Mat::PIXEL_BGR2RGB;
    ncnn::Mat in = ncnn::Mat::from_pixels(F.data, tp, F.cols, F.rows);
    MTCNN mtcnn(dir); mtcnn.SetMinFace(mi); return mtcnn.Detect(in);
}//end GetFaceBox


auto PT = [](float x, float y){ return cv::Point(int(x), int(y)); };
////////////////////////////////////////////////////////////////////////////////
//FAS_RGBD: F=color(BGR) -> Mark result inplace
void MarkFace_(cv::Mat& F, const BBox& bx, const cv::Scalar& BG, const string& info)
{
    const float* p = bx.kPoint; cv::Scalar TC(0,0,0), PC(0,255,0);
    int x = bx.x1, y = bx.y1, x2 = bx.x2, y2 = bx.y2, L = (int)info.size();
    cv::rectangle(F, cv::Point(x,y), cv::Point(x2,y2), BG, 1); //face box
    cv::rectangle(F, cv::Point(x,y), cv::Point(x+7*L,y+12), BG, -1); //info box
    cv::putText(F, info, cv::Point(x,y+11), 1, 0.8, TC); //info text
    for (int i = 0; i < 5; i++) cv::circle(F, PT(p[i],p[i+5]), 1, PC, 1);
}//end MarkFace_


const int cmb = 13; const string dir = "../weights";
enum FAS_TYPE { None = 0, Play = 1, Print = 2, Real = 3 };
auto dtms = [](clock_t x){ return 1000.f*(clock()-x)/CLOCKS_PER_SEC; };
////////////////////////////////////////////////////////////////////////////////
void main(int argc, char** argv)
{
    Cam ACam; clock_t t0 = clock();
    ACam.InitStream(openni::SENSOR_COLOR);
    ACam.InitStream(openni::SENSOR_DEPTH);

    cv::Mat Color, Depth, dp, im;
    int id, k = 0; float mx; char rt[9];
    FAS_RGBD FAS(cmb, dir); string info;
    std::vector<BBox> Box; cv::Scalar BG;
    while (k != 27)
    {
        Color = ACam.GetFrame(openni::SENSOR_COLOR);
        Depth = ACam.GetFrame(openni::SENSOR_DEPTH);
        if (Depth.empty() || Color.empty())
            throw runtime_error("Error: No RGBD!\n");
        t0 = clock(); Box = GetFaceBox(Color, dir);
        printf("Detect Faces: %.1fms\n", dtms(t0));

        for (auto &B : Box) //C++11
        {
            cv::Rect fc(B.x1, B.y1, B.x2-B.x1+1, B.y2-B.y1+1);
            dp = Depth(fc); cv::hconcat(dp, Color(fc), im);

            t0 = clock(); cout << "\t" << fc;
            cv::cvtColor(dp, dp, cv::COLOR_BGR2RGB);
            uint16_t dis = FAS_RGBD::Min2(dp); //RGB
            if (min(dp.rows, dp.cols) < 40)
            { info = "Small"; BG = cv::Scalar(99,99,99); }
            else if (FAS_RGBD::RatioN0(dp) < 0.5)
            { info = "Out/Block"; BG = cv::Scalar(99,99,99); }
            else if (dis < 350 || dis > 2000)
            { info = "Far/Near";  BG = cv::Scalar(99,99,99); }
            else //classify using CNN
            {
                t0 = clock(); id = Max(FAS.Discern(im), mx);
                switch (id) //Ref: FAS_TYPE
                {
                case 0: info = "None"; BG = cv::Scalar(0,222,222); break;
                case 1: info = "Play"; BG = cv::Scalar(0,0,255); break;
                case 2: info = "Print"; BG = cv::Scalar(0,0,255); break;
                case 3: info = "Real"; BG = cv::Scalar(0,255,0); break;
                }//end switch
                if (mx < 0.75) info = "X" + info; //BG=(255,0,255)
                sprintf_s(rt, ": %.3f", mx); info += rt;
            }//end if
            printf(" -> %s @ %.1fms\n", info, dtms(t0));
            
            cv::cvtColor(dp, dp, cv::COLOR_RGB2BGR);
            MarkFace_(Depth, B, BG, info); //stamp result
            MarkFace_(Color, B, BG, info); //stamp result
        }//end for
        cv::hconcat(Depth, Color, im);
        cv::imshow("Cam", im); k = cv::waitKey(10);
    }//end while
    cv::destroyAllWindows(); getchar();
}//end main

