#include "FAS_RGBD.h"


////////////////////////////////////////////////////////////////////////////////
FAS_RGBD::FAS_RGBD(int c, string model_dir)
{
    cmb = c; inp = inc(c);
    tp = typ(c); p = MeanStd(c);
    string md = model_dir + "/RGBD";
#if USE_MEM_MODEL
    net.load_param_bin((md+".param.bin").data());
#else
    net.load_param((md+".param").data());
#endif //USE_MEM_MODEL
    net.load_model((md+".bin").data());
}//end FAS_RGBD::FAS_RGBD


//F=Face's depth|color(RGB/BGR) -> FAS result
ncnn::Mat FAS_RGBD::Discern(const cv::Mat& F, bool BGR)
{
    cv::Mat im(F); //require: BGR->RGB
    if (BGR) cv::cvtColor(F, im, CV_BGR2RGB);
    im = Superpose(im, cmb); //RGB->RGB|A
    im = ResizedCrop(im, 224); //->(224,224)

    ncnn::Mat in, out; //from_pixels() reads continuous memory
    in = ncnn::Mat::from_pixels(im.data, tp, im.cols, im.rows);
    in.substract_mean_normalize(p, p+inp); //normalize

    ncnn::Extractor ex = net.create_extractor();
    //ex.set_light_mode(true); ex.set_num_threads(4);
#if USE_MEM_MODEL
    ex.input(MODEL_NAME_ID::BLOB_0, in);
    ex.extract(MODEL_NAME_ID::BLOB_282, out);
#else
    ex.input("0", in); ex.extract("282", out);
#endif //USE_MEM_MODEL
    return Softmax_(out);
}//end FAS_RGBD::Discern


////////////////////////////////////////////////////////////////////////////////
//F=depth(CV_16UC1) -> color(CV_8UC3|RGB)
cv::Mat FAS_RGBD::Encode(const cv::Mat& F, int md)
{
    if (md!=1) throw runtime_error("Unsupported!");
    const uint8_t cy = 22, r = 255/(cy-1);
    cv::Mat im = cv::Mat(F.rows, F.cols, CV_8UC3);
    const size_t Row = im.step[0], Col = im.step[1];
    //printf("%d\t%d\n", F.step[0], F.step[1]);
    for (int i = 0; i < F.rows; i++)
        for (int j = 0; j < F.cols; j++)
        {
            uint16_t x = *F.ptr<uint16_t>(i,j);
            uint8_t R = r*(x % cy); x /= cy;
            uint8_t G = r*(x % cy); x /= cy;
            uint8_t B = r*(x % cy); //x /= cy;

            im.data[i*Row + j*Col] = R; //Red
            im.data[i*Row + j*Col + 1] = G; //Green
            im.data[i*Row + j*Col + 2] = B; //Blue
        }//end for
    return im; //RGB
}//end FAS_RGBD::Encode


//x=depth(uint16_t) -> color(Vec3b|RGB)
cv::Vec3b FAS_RGBD::Encode(uint16_t x, int md)
{
    if (md!=1) throw runtime_error("Unsupported!");
    const uint8_t cy = 22, r = 255/(cy-1);
    cv::Vec3b RGB; //0 < x < 22^3
    RGB[0] = r*(x % cy); x /= cy; //Red
    RGB[1] = r*(x % cy); x /= cy; //Green
    RGB[2] = r*(x % cy); return RGB; //Blue
}//end FAS_RGBD::Encode


//F=color(Vec3b|RGB) -> depth(uint16_t)
uint16_t FAS_RGBD::Decode(const cv::Vec3b& F, int md)
{
    if (md!=1) throw runtime_error("Unsupported!");
    const uint8_t cy = 22, r = 255/(cy-1);
    uint16_t x = (F[0]/r) + (F[1]/r)*cy + (F[2]/r)*cy*cy;
    return x; //uint16_t
}//end FAS_RGBD::Decode


//F=color(CV_8UC3|RGB) -> depth(CV_16UC1)
cv::Mat FAS_RGBD::Decode(const cv::Mat& F, int md)
{
    if (md!=1) throw runtime_error("Unsupported!");
    const uint8_t cy = 22, r = 255/(cy-1);
    const size_t Row = F.step[0], Col = F.step[1];
    cv::Mat im = cv::Mat(F.rows, F.cols, CV_16UC1);
    //printf("%d\t%d\n", F.step[0], F.step[1]);
    for (int i = 0; i < F.rows; i++)
        for (int j = 0; j < F.cols; j++)
        {
            uint8_t R = F.data[i*Row + j*Col] / r; //Red
            uint8_t G = F.data[i*Row + j*Col + 1] / r; //Green
            uint8_t B = F.data[i*Row + j*Col + 2] / r; //Blue
            *im.ptr<uint16_t>(i,j) = R + G*cy + B*cy*cy;
        }//end for
    return im; //CV_16UC1
}//end FAS_RGBD::Decode


////////////////////////////////////////////////////////////////////////////////
//F=depth(CV_16UC1|RGB) -> find 2nd_min
uint16_t FAS_RGBD::Min2(cv::Mat F)
{
    if (F.channels()>1) F = Decode(F); //RGB
    double mi, mx; cv::minMaxIdx(F, &mi, &mx);
    uint16_t min = uint16_t(mi);
    if (mi==0 && mx>mi) //find 2nd_min
    {
        min = uint16_t(mx); //initial
        for (int i = 0; i < F.rows; i++)
            for (int j = 0; j < F.cols; j++)
            {
                uint16_t x = *F.ptr<uint16_t>(i,j);
                if (x>0 && x<min) min = x;
            }//end for
    }//end if
    return min; //uint16_t
}//end FAS_RGBD::Min2


//F=depth(CV_16UC1|RGB) -> clip depth(CV_8UC1)
cv::Mat FAS_RGBD::DepthClip(cv::Mat F)
{
    if (F.channels()>1) F = Decode(F); //RGB
    uint16_t ds = Min2(F); ds = ds-(ds>0); //offset
    cv::Mat im = cv::Mat(F.rows, F.cols, CV_8UC1);
    for (int i = 0; i < F.rows; i++)
        for (int j = 0; j < F.cols; j++)
        {
            uint16_t x = *F.ptr<uint16_t>(i,j);
            if (x>0) x -= ds; //relativize
            if (x>255) x = 0; //clip->0
            *im.ptr<uint8_t>(i,j) = (uint8_t)x;
        }//end for
    return im; //CV_8UC1
}//end FAS_RGBD::DepthClip


//F=depth(CV_16UC1|CV_8UC1|RGB) -> non-zero ratio
float FAS_RGBD::RatioN0(cv::Mat F)
{
    if (F.channels()>1) F = Decode(F); //RGB
    return cv::countNonZero(F) / (float)F.total();
}//end FAS_RGBD::RatioN0


////////////////////////////////////////////////////////////////////////////////
//F=depth|color(RGB) -> combination(RGB|A)
cv::Mat FAS_RGBD::Superpose(const cv::Mat& F, int c)
{
    const int h = F.rows, w = F.cols/2;
    cv::Rect dp(0, 0, w, h), co(w, 0, w, h);
    cv::Mat A(F(dp)), B(F(co)), im, CA[2]; //RGB
    switch (c)
    {
    case 0:  cv::vconcat(A, B, im); break; //CV_8UC3: A+B vertically
    case -1: cv::hconcat(A, B, im); break; //CV_8UC3: A+B horizontally
    case -2: cv::addWeighted(A, 1, B, 0.5, 0, im); break; //CV_8UC3: A+B
    case 1: im = DepthClip(Decode(A)); break; //CV_8UC1: clip A
    case 2: im = Decode(A); //CV_8UC1: not clip A
        double mi, mx; cv::minMaxIdx(im, &mi, &mx); //normalize
        im.convertTo(im, CV_8UC1, 255/mx, 0); break; //[0,255]
        //im.convertTo(im, CV_32FC1, 2/mx, -1); break; //[-1,1]
    case 3: im = A.clone(); break; //CV_8UC3: A
    case 4: //CV_8UC4: A(CV_8UC3) + A(CV_8UC1, clip)
        CA[0] = A; CA[1] = DepthClip(Decode(A));
        cv::merge(CA, 2, im); break;
    case 13: //CV_8UC4: B(CV_8UC3) + A(CV_8UC1, clip)
        im = Decode(A); B.at<cv::Vec3b>(0,0) = Encode(Min2(im));
        CA[0] = B; CA[1] = DepthClip(im); cv::merge(CA, 2, im); break;
    case 31: //CV_8UC4: A(CV_8UC3) + B(CV_8UC1, gray)
        CA[0] = A; cv::cvtColor(B, CA[1], CV_BGR2GRAY);
        cv::merge(CA, 2, im); break;
    }//end switch
    return im; //RGB|A
}//end FAS_RGBD::Superpose


//Ref: nxnn::Mat::substract_mean_normalize
//https://github.com/Tencent/ncnn/blob/master/src/mat.cpp
//https://github.com/Tencent/ncnn/blob/master/src/layer/scale.cpp
//Ref: DataAug.py->(MeanStd, TSFM_Test)->(ToTensor, Normalize)
//https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
//Deduce: A'=(A/255-Pm)/Ps=(A-m)*s -> m=255*Pm, s=1/(255*Ps)
float* FAS_RGBD::MeanStd(int c) //Ref: Superpose(RGB|A)
{
    double p1[] = { 0.15, 0.12 }; //GRAY: depth cliped
    double p2[] = { 0.50, 0.25 }; //GRAY: depth original
    double p0[] = { 0.485,0.458,0.407, 0.229,0.224,0.225 }; //RGB: color
    double p3[] = { 0.330,0.353,0.047, 0.250,0.250,0.047 }; //RGB: depth
    double p4[]  = { 0.330,0.353,0.047,0.15, 0.250,0.250,0.047,0.120 }; //RGBA: depth+depth
    double p13[] = { 0.485,0.458,0.407,0.15, 0.229,0.224,0.225,0.120 }; //RGBA: color+depth
    double p31[] = { 0.330,0.353,0.047,0.45, 0.250,0.250,0.047,0.226 }; //RGBA: depth+color

    double *pc = NULL;
    switch (c)
    {
    case 1: pc = p1; break;
    case 2: pc = p2; break;
    case 3: pc = p3; break;
    case 4: pc = p4; break;
    case 13: pc = p13; break;
    case 31: pc = p31; break;
    default: pc = p0; break;
    }//end switch

    const int N = inc(c);
    float *p = new float[2*N]();
    for (int i = 0; i < 2*N; i++)
    {
        p[i] = float(255 * pc[i]);
        if (i>=N) p[i] = 1.f / p[i];
    }//end for
    return p;
}//end FAS_RGBD::MeanStd


////////////////////////////////////////////////////////////////////////////////
//Resize->(256,256), then Center-Crop->(wh,wh)
//clone() is necessary for ROI, to make memory continuous,
//for ncnn::Mat::from_pixels() reads from continuous memory.
cv::Mat FAS_RGBD::ResizedCrop(const cv::Mat& F, int wh)
{
    int sz = 256, s0 = (sz-wh)/2; cv::Mat im;
    cv::resize(F, im, cv::Size(sz, sz)); //(sz, wh)
    cv::Rect ROI(s0, s0, wh, wh); //(s0, 0, wh, wh)
    return im(ROI).clone(); //make memory continuous
    //or else: from_pixels() may read data incorrectly.
}//end FAS_RGBD::ResizedCrop


////////////////////////////////////////////////////////////////////////////////
//F: 1-dim ncnn::Mat, find max's value and location
int Max(const ncnn::Mat& F, float& mx)
{
    int loc = 0; mx = F[0];
    for (int i = 1; i < F.w; i++)
        if (F[i] > mx) { loc = i; mx = F[i]; }
    return loc;
}//end Max


//F: 1-dim ncnn::Mat, softmax inplace
ncnn::Mat Softmax_(ncnn::Mat& F)
{
    float M = F[0], Sum = 0;
    const int N = F.c * F.h * F.w; //F.w
    for (int i = 1; i < N; i++) M = max(M, F[i]);
    for (int i = 0; i < N; i++)
    {
        F[i] = exp(F[i]-M); Sum += F[i];
    }//end for
    for (int i = 0; i < N; i++) F[i] /= Sum;
    return F;
}//end Softmax_


void ShowMat(const ncnn::Mat& F, int cmb)
{
    //printf("c=%d h=%d w=%d\t", F.c, F.h, F.w);
    //printf("B=%zd N=%zd\n", F.elemsize, F.cstep);
    if (cmb == CV_8U) //CV_8U: H->W->C
    {
        int tc = CV_MAKETYPE(cmb, FAS_RGBD::inc(cmb));
        cv::Mat x = cv::Mat::zeros(F.h, F.w, tc);
        F.to_pixels(x.data, FAS_RGBD::typ(cmb));
        cout << x << endl; //utilize cv::Mat
        /*for (int c = 0; c < F.c; c++) //C->H->W
        {
            ncnn::Mat fc = F.channel(c);
            cv::Mat x = cv::Mat::zeros(F.h, F.w, CV_8UC1);
            fc.to_pixels(x.data, ncnn::Mat::PIXEL_GRAY);
            cout << x << endl;
        }//end for*/
    }//end if
    else //universal: C->H->W
    {
        ncnn::Mat x = F.reshape(F.c*F.h*F.w);
        for (int i = 0, c = F.h*F.w; i < x.w; )
        {
            cout << x[i] << " "; i++;
            if (i%F.w == 0) cout << "\b,\n";
            if (i%c == 0 && i < x.w) cout << endl;
        }//end for
    }//end else
}//end ShowMat

