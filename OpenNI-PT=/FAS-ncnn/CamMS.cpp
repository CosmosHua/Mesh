#include "CamMS.h"


////////////////////////////////////////////////////////////////////////////////
Cam::Cam(): m_device(nullptr), m_stream_color(nullptr),
            m_stream_ir(nullptr), m_stream_depth(nullptr)
{}//end Cam


Cam::~Cam()
{
    Destroy(m_stream_color);
    Destroy(m_stream_depth);
    Destroy(m_stream_ir);
    CloseDeivce();
}//end ~Cam


void Cam::Destroy(void* &m_stream)
{
    if (nullptr != m_stream)
    {
        //VideoStream encapsulates a single video stream from a device
        openni::VideoStream* p_stream = (openni::VideoStream*)m_stream; //convert type
        p_stream->destroy(); delete p_stream; p_stream = nullptr; //destroy this stream
        m_stream = nullptr; //reference of the pointer
    }//end if
}//end Destroy


////////////////////////////////////////////////////////////////////////////////
int Cam::CloseDeivce()
{
    if (nullptr != m_device)
    {
        openni::Device* pDevice = (openni::Device*)m_device; //convert type
        pDevice->close(); delete pDevice; pDevice = nullptr; //close device
        m_device = nullptr; //nullptr=NULL
    }//end if

    //unload all drivers, close all streams and devices
    openni::OpenNI::shutdown(); //Shutdown OpenNI2
    return 0; //successfully close device
}//end CloseDeivce


int Cam::OpenDevice()
{
    if (nullptr != m_device) return 0; //already opened

    //load all available drivers, and see which devices are available
    openni::Status rc = openni::OpenNI::initialize(); //Initialize OpenNI2
    if (rc != openni::STATUS_OK)
    {
        //retrieve each calling thread's last extended error information
        printf("Init OpenNI2:\n%s\n", openni::OpenNI::getExtendedError());
        return OPENNI_INITIAL_FAILED;
    }//end if

    int DeviceNum = 0; //number of devices
    OniDeviceInfo* pDeviceList; //pointer of devices' list
    OniStatus sta = oniGetDeviceList(&pDeviceList, &DeviceNum);
    if (sta != ONI_STATUS_OK)
    {
        printf("OniGetDeviceList:\n%s\n", openni::OpenNI::getExtendedError());
        return OPENNI_INITIAL_FAILED;
    }//end if

    openni::Device* pDevice = new openni::Device; //create device
    //open the 1st device by passing its URI(Uniform Resource Identifier)
    //rc = pDevice->open(openni::ANY_DEVICE); //only for only 1 device
    rc = pDevice->open(pDeviceList[0].uri); //OR: openni::ANY_DEVICE
    if (rc != openni::STATUS_OK) //OR: pDevice->isValid()
    {
        printf("Device Open Failed:\n%s\n", openni::OpenNI::getExtendedError());
        delete pDevice; pDevice = nullptr; //free memory
        openni::OpenNI::shutdown(); //Shutdown OpenNI2
        return DEVICE_OPEN_FAILED;
    }//end if

    //should edit: OpenNI2/Drivers/orbbec.ini
    rc = pDevice->setDepthColorSyncEnabled(true); //useless
    if (rc == openni::STATUS_OK) printf("Frame_Sync: Enabled.\n");

    //Invalid Image_Registration below -> use orbbec.ini instead.
    //rc = pDevice->setImageRegistrationMode((openni::ImageRegistrationMode)1);
    //if (!pDevice->getImageRegistrationMode()) //STATUS_BAD_PARAMETER = 4
    //  printf("Image_Registration: Failed, %d.\n", rc);
    m_device = pDevice; return 0; //successfully open device
}//end OpenDevice


////////////////////////////////////////////////////////////////////////////////
//IR & Color Can't Init Concurrently!
int Cam::InitStream(const int CamType)
{
    char* hint = nullptr; //prompt message
    void** m_stream = nullptr; //stream pointer
    openni::PixelFormat DType; //pixel data type
    int INIT_FAILED, START_FAILED; //error code
    switch (CamType) //device's sensor type
    {
    case openni::SENSOR_COLOR:
        hint = "SENSOR_COLOR";
        m_stream = &m_stream_color;
        DType = openni::PIXEL_FORMAT_RGB888;
        INIT_FAILED = STREAM_COLOR_INIT_FAILED;
        START_FAILED = STREAM_COLOR_START_FAILED;
        break;
    case openni::SENSOR_DEPTH:
        hint = "SENSOR_DEPTH";
        m_stream = &m_stream_depth;
        DType = openni::PIXEL_FORMAT_DEPTH_1_MM;
        INIT_FAILED = STREAM_DEPTH_INIT_FAILED;
        START_FAILED = STREAM_DEPTH_START_FAILED;
        break;
    case openni::SENSOR_IR:
        hint = "SENSOR_IR";
        m_stream = &m_stream_ir;
        DType = openni::PIXEL_FORMAT_GRAY16;
        INIT_FAILED = STREAM_IR_INIT_FAILED;
        START_FAILED = STREAM_IR_START_FAILED;
        break;
    default: return UNSUPPORTED_SENSOR_TYPE;
    }//end switch

    if (nullptr != *m_stream) return 0; //stream already initialized

    int iret = OpenDevice(); //open device->m_device
    if (iret != 0) return iret; //check device
    if (nullptr == m_device) return DEVICE_OPEN_FAILED; //useless

    openni::Device* pDevice = (openni::Device*)m_device; //convert type
    openni::VideoStream* p_stream = new openni::VideoStream; //new VideoStream
    //initialize a stream of frames from a specific sensor of a specific device
    openni::Status rc = p_stream->create(*pDevice, (openni::SensorType)CamType);

    //CamType == p_stream->getSensorInfo().getSensorType()
    if (rc != openni::STATUS_OK) //check stream initialization
    {
        printf("Create %s Stream Failed:\n%s\n", hint, openni::OpenNI::getExtendedError());
        p_stream->destroy(); delete p_stream; p_stream = nullptr; //destroy this stream
        return INIT_FAILED;
    }//end if

    //int mi = p_stream->getMinPixelValue(), mx = p_stream->getMaxPixelValue();
    //printf("%s Pixel_Value = [%d,%d].\n", hint, mi, mx); //show extremum
 
    //store VideoStream's settings(e.g. fps, resolution, pixel format)
    openni::VideoMode viewMode = p_stream->getVideoMode();
    viewMode.setPixelFormat(DType); //change PixelFormat
    p_stream->setVideoMode(viewMode); //change stream's VideoMode

    rc = p_stream->start(); //start to generate data from stream
    if (rc != openni::STATUS_OK) //check start
    {
        printf("Start %s Stream Failed:\n%s\n", hint, openni::OpenNI::getExtendedError());
        p_stream->destroy(); delete p_stream; p_stream = nullptr; //destroy this stream
        return START_FAILED;
    }//end if

    *m_stream = p_stream; return 0; //successfully initialize
}//end InitStream


int Cam::CloseStream(const int CamType)
{
    if (CamType == openni::SENSOR_COLOR)
        Destroy(m_stream_color);
    else if (CamType == openni::SENSOR_DEPTH)
        Destroy(m_stream_depth);
    else if (CamType == openni::SENSOR_IR)
        Destroy(m_stream_ir);
    return 0; //successfully close stream
}//end CloseStream


bool Cam::isValid(const int CamType)const
{
    if (CamType == openni::SENSOR_COLOR)
        return m_stream_color != nullptr;
    else if (CamType == openni::SENSOR_DEPTH)
        return m_stream_depth != nullptr;
    else if (CamType == openni::SENSOR_IR)
        return m_stream_ir != nullptr;
    else return false; //UNSUPPORTED_SENSOR_TYPE
}//end isValid


////////////////////////////////////////////////////////////////////////////////
cv::Mat Cam::GetFrame(const int CamType, cv::Mat(*pf)(const openni::VideoFrameRef&))const
{
    cv::Mat NullMat; //empty image matrix
    char* hint = nullptr; //prompt message
    void *const * m_stream = nullptr; //stream pointer
    //openni::PixelFormat DType; //pixel data type
    switch (CamType) //device's sensor type
    {
    case openni::SENSOR_COLOR:
        hint = "SENSOR_COLOR";
        m_stream = &m_stream_color;
        if (!pf) pf = RGBFrame; //for RGB frame
        //DType = openni::PIXEL_FORMAT_RGB888;
        break;
    case openni::SENSOR_DEPTH:
        hint = "SENSOR_DEPTH";
        m_stream = &m_stream_depth;
        if (!pf) pf = DepthFrame; //for Depth frame
        //DType = openni::PIXEL_FORMAT_DEPTH_1_MM;
        break;
    case openni::SENSOR_IR:
        hint = "SENSOR_IR";
        m_stream = &m_stream_ir;
        if (!pf) pf = IRFrame; //for IR frame
        //DType = openni::PIXEL_FORMAT_GRAY16;
        break;
    default: return NullMat; //UNSUPPORTED_SENSOR_TYPE
    }//end switch

    if (nullptr == m_device) return NullMat; //check device
    openni::Device* pDevice = (openni::Device*)m_device;

    if (nullptr == *m_stream) return NullMat; //check sensor stream
    openni::VideoStream* p_stream = (openni::VideoStream*)(*m_stream);

    int StreamNum = 1, StreamIndex;
    //Wait for a new frame from p_stream, until TimeOut has passed.->prepare for readFrame()
    openni::Status rc = openni::OpenNI::waitForAnyStream(&p_stream, StreamNum, &StreamIndex, TimeOut);
    if (rc != openni::STATUS_OK) //check waitForAnyStream()
    {
        printf("%s: GetFrame Failed, over %d ms.\n", hint, TimeOut);
        return NullMat; //empty image matrix
    }//end if

    //VideoFrameRef encapsulates a single video frame.
    openni::VideoFrameRef Frame; //reference to a VideoStream's output
    //block until read the next/new frame from the video stream
    rc = p_stream->readFrame(&Frame); //VideoStream's output->Frame

    if (rc != openni::STATUS_OK) return NullMat; //check readFrame()
    return pf(Frame); //convert openni::Frame to cv::Mat
}//end GetFrame


////////////////////////////////////////////////////////////////////////////////
//convert openni::Frame(IR Format) to cv::Mat(CV_8UC1)
cv::Mat IRFrame(const openni::VideoFrameRef& Frame)
{
    //get VideoMode assigned to this frame -> get pixel data type
    //openni::PixelFormat DType = Frame.getVideoMode().getPixelFormat();
    //if (openni::PIXEL_FORMAT_GRAY16 != DType) return cv::Mat(); //empty

    //Grayscale16Pixel = uint16_t, which occupies 2-bytes, in [0,1023].
    openni::Grayscale16Pixel* pPixel = (openni::Grayscale16Pixel*)Frame.getData();
    int size = Frame.getDataSize() / 2; //FrameData measured in bytes
    cv::Mat Img = cv::Mat(Frame.getHeight(), Frame.getWidth(), CV_8UC1);

    for (int i = 0; i < size; i++) //convert pixel data
        Img.data[i] = (uint8_t)pPixel[i]; //loss data
    cv::cvtColor(Img, Img, cv::COLOR_GRAY2BGR);
    return Img; //IR image
}//end IRFrame


//convert openni::Frame(RGB Format) to cv::Mat(CV_8UC3)
cv::Mat RGBFrame(const openni::VideoFrameRef& Frame)
{
    //get VideoMode assigned to this frame -> get pixel data type
    openni::PixelFormat DType = Frame.getVideoMode().getPixelFormat();
    if (openni::PIXEL_FORMAT_RGB888 != DType) return cv::Mat(); //empty

    //RGB888Pixel = {uint8_t r, g, b}, uint8_t occupies 1-byte.
    openni::RGB888Pixel* pPixel = (openni::RGB888Pixel*)Frame.getData();
    int size = Frame.getDataSize() / 3; //FrameData measured in bytes
    cv::Mat Img = cv::Mat(Frame.getHeight(), Frame.getWidth(), CV_8UC3);

    for (int i = 0; i < size; i++) //convert pixel data
    {
        Img.data[3*i] = pPixel[i].b; //blue
        Img.data[3*i + 1] = pPixel[i].g; //green
        Img.data[3*i + 2] = pPixel[i].r; //red
    }//end for
    return Img; //BGR image
}//end RGBFrame


//convert openni::Frame(Depth Format) to cv::Mat(CV_8UC3)
cv::Mat DepthFrame(const openni::VideoFrameRef& Frame)
{
    //get VideoMode assigned to this frame -> get pixel data type
    //openni::PixelFormat DType = Frame.getVideoMode().getPixelFormat();
    //if (openni::PIXEL_FORMAT_DEPTH_100_UM != DType &&
    //  openni::PIXEL_FORMAT_DEPTH_1_MM != DType) return cv::Mat(); //empty

    //DepthPixel = uint16_t, occupies 2-bytes, measured in 1mm|100um.
    openni::DepthPixel* pPixel = (openni::DepthPixel*)Frame.getData();
    int size = Frame.getDataSize() / 2; //FrameData measured in bytes
    cv::Mat Img = cv::Mat(Frame.getHeight(), Frame.getWidth(), CV_8UC3);
 
    //Convert_Method_1:/*
    const int cy = 22, r = 255/(cy-1), h = 180/(cy-1);
    for (int i = 0; i < size; i++) //convert pixel data
    {
        uint16_t ui = pPixel[i]; //ui=[0,10000]
        uint8_t R = r*(ui % cy); ui /= cy; //r<->h
        uint8_t G = r*(ui % cy); ui /= cy;
        uint8_t B = r*(ui % cy); ui /= cy;

        Img.data[3*i] = B; //Blue|Hue
        Img.data[3*i + 1] = G; //Green|Saturation
        Img.data[3*i + 2] = R; //Red|Value
    }//end for
    //HSV->BGR will arise numerical inaccuracy
    //cv::cvtColor(Img, Img, cv::COLOR_HSV2BGR);
    return Img; //Depth image*/

    /*//Convert_Method_2:
    const int c1 = 40, c2 = 250, r1 = 255/c1, r2 = 255/c2;
    for (int i = 0; i < size; i++) //convert pixel data
    {
        uint16_t ui = pPixel[i]; //ui=[0,10000]
        uint8_t S = 166; //any fixed value
        uint8_t V = r1*(ui % c1); ui /= c1;
        uint8_t H = r2*(ui % c2); ui /= c2;

        Img.data[3*i] = H; //Hue
        Img.data[3*i + 1] = S; //Saturation
        Img.data[3*i + 2] = V; //Value
    }//end for
    cv::cvtColor(Img, Img, cv::COLOR_HSV2BGR);
    return Img; //Depth image*/
}//end DepthFrame


////////////////////////////////////////////////////////////////////////////////
void Superpose(cv::Mat& im, std::vector<cv::Mat> Img, int sup)
{
    if (sup == 0 || Img.size() < 2) return;
    cv::Mat Depth(Img[0]), IC(Img[1]); //IR|Color
    if (sup == 1) //1: superpose images
        im = Depth + 0.5*IC; //superimpose images
        //cv::addWeighted(Depth, 0.8, IC, 1, 0, im);
    else if (sup == 2) //2: IC->Alpha channel
    {
        cv::cvtColor(im, im, cv::COLOR_BGR2BGRA);
        cv::cvtColor(IC, IC, cv::COLOR_BGR2GRAY);
        cv::split(Depth, Img); Img.push_back(IC);
        cv::merge(Img, im); Img.clear();
    }//end elif
    else if(sup == 3) //3: Depth->Alpha channel
    {
        cv::cvtColor(im, im, cv::COLOR_BGR2BGRA);
        cv::cvtColor(Depth, Depth, cv::COLOR_BGR2GRAY);
        cv::split(IC, Img); Img.push_back(Depth);
        cv::merge(Img, im); Img.clear();
    }//end elif
}//end Superpose


void SaveImage(const cv::Mat& im, int& tid, int& i, int f2p)
{
    if (im.empty()) return; //check im
    char id[25]; sprintf_s(id, "%d/", f2p);
    std::string dir = path + id; //check folder
    if (_access(dir.data(), 0) != 0) _mkdir(dir.data());

    if (tid != time(0)) //name rule
    {
        tid = (int)time(0); i = 0;
    }//end if
    sprintf_s(id, "%d_%02d.png", tid, i);
    cv::imwrite(dir + id, im);
}//end SaveImage


//PS: Video decrease image quality->disturb depth info
void SaveVideo(const cv::Mat& im, cv::VideoWriter& out)
{
    if (im.empty()) return;
    if (!out.isOpened())
    {
        int codec = CV_FOURCC('X','V','I','D'); //MPEG-4, better
        //int codec = CV_FOURCC('D','I','V','X'); //MPEG-4, shameful
        //int codec = CV_FOURCC('P','I','M','1'); //MPEG-1, uncompress
        //int codec = CV_FOURCC('M','P','4','2'); //MPEG-4.2
        //int codec = CV_FOURCC('D','I','V','3'); //MPEG-4.3
        //int codec = CV_FOURCC('U','2','6','3'); //H263
        //int codec = CV_FOURCC('I','2','6','3'); //H263I
        std::string vid = path + std::to_string(time(0)) + ".mkv";
        out.open(vid, codec, 12, cv::Size(im.cols, im.rows));
        //out.set(cv::VIDEOWRITER_PROP_QUALITY, 1.0);
    }//end if
    out << im; //out.write(im)
}//end SaveVideo


//extract frames as png from video if given tid
void ExtractVideo(const std::string tid, int mod)
{
    std::string pre = path + tid; //check folder
    if (_access(pre.data(),0) != 0) _mkdir(pre.data());

    cv::VideoCapture vid(pre + ".mkv");
    int w = (int)vid.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)vid.get(cv::CAP_PROP_FRAME_HEIGHT);
    pre += "/" + tid + "_"; char id[9]; cv::Mat im;
    while (TRUE) //extract whole frames
    {
        sprintf_s(id, "%04d", (int)vid.get(cv::CAP_PROP_POS_FRAMES));
        if (!vid.read(im)) break; //vid >> im;
        if (!mod) //horizontally->vertically
        {
            cv::Range row(0,h), c1(0,w/2), c2(w/2,w);
            cv::vconcat(im(row,c1), im(row,c2), im);
        }//end if
        cv::imwrite(pre + id + ".png", im);
    }//end while
    vid.release();
}//end ExtractVideo
