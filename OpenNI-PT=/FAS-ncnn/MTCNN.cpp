#include "MTCNN.h"


////////////////////////////////////////////////////////////////////////////////
bool cmpScore(const BBox& A, const BBox& B)
{
    return A.Score < B.Score; //descent
}//end cmpScore


//sort ascent: return A.Score < B.Score;
bool operator<(const BBox& A, const BBox& B)
{
    return A.Score > B.Score; //descent
}//end operator<


////////////////////////////////////////////////////////////////////////////////
MTCNN::MTCNN(const string &model_path)
{
    std::vector<std::string> bins =
    { model_path + "/pnet.bin", model_path + "/rnet.bin", model_path + "/onet.bin" };
    std::vector<std::string> params = 
    { model_path + "/pnet.param", model_path + "/rnet.param", model_path + "/onet.param" };

    Pnet.load_param(params[0].data()); Pnet.load_model(bins[0].data());
    Rnet.load_param(params[1].data()); Rnet.load_model(bins[1].data());
    Onet.load_param(params[2].data()); Onet.load_model(bins[2].data());
}//end MTCNN::MTCNN


MTCNN::MTCNN(const std::vector<std::string> params, const std::vector<std::string> bins)
{
    Pnet.load_param(params[0].data()); Pnet.load_model(bins[0].data());
    Rnet.load_param(params[1].data()); Rnet.load_model(bins[1].data());
    Onet.load_param(params[2].data()); Onet.load_model(bins[2].data());
}//end MTCNN::MTCNN


////////////////////////////////////////////////////////////////////////////////
void MTCNN::GenBBox(ncnn::Mat prob, ncnn::Mat loc, std::vector<BBox> &Box, float scale)
{
    float *p = prob.channel(1);
    const int stride = 2, cellsize = 12;
    BBox box = { 0 }; scale = 1.f / scale;
    for (int row = 0; row < prob.h; row++)
        for (int col = 0; col < prob.w; col++)
        {
            if (*p > threshold[0])
            {
                box.Score = *p;
                box.x1 = lround((stride * col + 1) * scale);
                box.y1 = lround((stride * row + 1) * scale);
                box.x2 = lround((stride * col + 1 + cellsize) * scale);
                box.y2 = lround((stride * row + 1 + cellsize) * scale);
                box.Area = float(box.x2 - box.x1) * (box.y2 - box.y1);
                const int id = row * prob.w + col;
                for (int c = 0; c < 4; c++)
                    box.Pos[c] = loc.channel(c)[id];
                Box.push_back(box);
            }//end if
            p++;
        }//end for
}//end MTCNN::GenBBox


void MTCNN::NMS(std::vector<BBox> &Box, float IOU_threshold, string type)
{
    if (Box.empty()) return;
    sort(Box.begin(), Box.end(), cmpScore);

    float IOU = 0, cur, las;
    float maxX = 0, maxY = 0;
    float minX = 0, minY = 0;

    const size_t NB = Box.size();
    std::vector<int> vecPick(NB);
    std::multimap<float, int> vScores;
    for (int i = 0; i < NB; i++)
        vScores.insert(std::pair<float, int>(Box[i].Score, i));

    size_t pickCount = 0;
    while (!vScores.empty())
    {
        int last = vScores.rbegin()->second;
        vecPick[pickCount] = last;
        pickCount += 1;
        for (auto it = vScores.begin(); it != vScores.end();)
        {
            int it_idx = it->second;
            maxX = (float)max(Box.at(it_idx).x1, Box.at(last).x1);
            maxY = (float)max(Box.at(it_idx).y1, Box.at(last).y1);
            minX = (float)min(Box.at(it_idx).x2, Box.at(last).x2);
            minY = (float)min(Box.at(it_idx).y2, Box.at(last).y2);
            //reuse: maxX1, maxY1, IOU
            maxX = max(minX - maxX + 1, 0);
            maxY = max(minY - maxY + 1, 0);
            IOU = maxX * maxY;

            cur = Box.at(it_idx).Area; las = Box.at(last).Area;
            if (type == "Union")    IOU = IOU / (cur + las - IOU);
            else if (type == "Min") IOU = IOU / min(cur, las);

            if (IOU > IOU_threshold) it = vScores.erase(it);
            else it++;
        }//end for
    }//end while

    vecPick.resize(pickCount);
    std::vector<BBox> bx(pickCount);
    for (int i = 0; i < pickCount; i++)
        bx[i] = Box[vecPick[i]];
    Box = bx;
}//end MTCNN::NMS


void MTCNN::Refine(std::vector<BBox> &Box, int height, int width, bool square)
{
    if (Box.empty()) return;

    float h = 0, w = 0;
    float bbw = 0, bbh = 0, maxSide = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    for (auto &it : Box)
    {
        bbw = (float)it.x2 - it.x1 + 1;
        bbh = (float)it.y2 - it.y1 + 1;
        x1 = it.x1 + it.Pos[0] * bbw;
        y1 = it.y1 + it.Pos[1] * bbh;
        x2 = it.x2 + it.Pos[2] * bbw;
        y2 = it.y2 + it.Pos[3] * bbh;
        if (square)
        {
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h > w) ? h : w;
            x1 += (w - maxSide) * 0.5f;
            y1 += (h - maxSide) * 0.5f;
            it.x2 = lround(x1 + maxSide - 1);
            it.y2 = lround(y1 + maxSide - 1);
            it.x1 = lround(x1);
            it.y1 = lround(y1);
        }//end if
        //boundary check
        if (it.x1 < 0) it.x1 = 0;
        if (it.y1 < 0) it.y1 = 0;
        if (it.x2 > width)  it.x2 = width - 1;
        if (it.y2 > height) it.y2 = height - 1;
        it.Area = float(it.x2 - it.x1) * (it.y2 - it.y1);
    }//end for
}//end MTCNN::Refine


////////////////////////////////////////////////////////////////////////////////
void MTCNN::PNet()
{
    BBox1.clear();
    std::vector<float> scales;
    float minlen = (float) min(img_w, img_h);
    float rt = (float) MIN_DET_SIZE / minsize;
    minlen *= rt;
    while (minlen > MIN_DET_SIZE)
    {
        scales.push_back(rt);
        minlen *= facetor;
        rt *= facetor;
    }//end while
    for (float sc : scales) //C++11
    {
        int hs = (int) ceil(img_h * sc);
        int ws = (int) ceil(img_w * sc);
        ncnn::Mat in, prob, location;
        resize_bilinear(img, in, ws, hs);

        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ex.extract("prob1", prob);
        ex.extract("conv4-2", location);

        std::vector<BBox> box;
        GenBBox(prob, location, box, sc);
        NMS(box, nms_threshold[0]);
        BBox1.insert(BBox1.end(), box.begin(), box.end());
        box.clear();
    }//end for
}//end MTCNN::PNet


void MTCNN::RNet()
{
    BBox2.clear();
    for (auto &it : BBox1)
    {
        ncnn::Mat tmp, in, prob, box;
        copy_cut_border(img, tmp, it.y1, img_h-it.y2, it.x1, img_w-it.x2);
        resize_bilinear(tmp, in, 24, 24);

        ncnn::Extractor ex = Rnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ex.extract("prob1", prob);
        ex.extract("conv5-2", box);

        if (prob[1] > threshold[1])
        {
            for (int c = 0; c < 4; c++)
                it.Pos[c] = (float) box[c];
            it.Area = float(it.x2 - it.x1) * (it.y2 - it.y1);
            it.Score = prob.channel(1)[0];
            BBox2.push_back(it);
        }//end if
    }//end for
}//end MTCNN::RNet


void MTCNN::ONet()
{
    BBox3.clear();
    for (auto &it : BBox2)
    {
        ncnn::Mat tmp, in, prob, box, KPt;
        copy_cut_border(img, tmp, it.y1, img_h-it.y2, it.x1, img_w-it.x2);
        resize_bilinear(tmp, in, 48, 48);

        ncnn::Extractor ex = Onet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ex.extract("prob1", prob);
        ex.extract("conv6-2", box);
        ex.extract("conv6-3", KPt);

        if (prob[1] > threshold[2])
        {
            for (int c = 0; c < 4; c++)
                it.Pos[c] = (float) box[c];
            it.Area = float(it.x2 - it.x1) * (it.y2 - it.y1);
            it.Score = prob.channel(1)[0];
            for (int i = 0; i < 5; i++)
            {
                (it.kPoint)[i]     = it.x1 + (it.x2 - it.x1) * KPt[i];
                (it.kPoint)[i + 5] = it.y1 + (it.y2 - it.y1) * KPt[i + 5];
            }//end for
            BBox3.push_back(it);
        }//end if
    }//end for
}//end MTCNN::ONet


////////////////////////////////////////////////////////////////////////////////
void Adjust(BBox &box, float rt=0.85)
{
    int w = box.x2 - box.x1 + 1;
    int h = box.y2 - box.y1 + 1;
    if (w > h*rt)
    {
        w = lround((w-h*rt)/2);
        box.x1 += w; box.x2 -= w;
    }//end if
}//end Adjust


std::vector<BBox> MTCNN::Detect(ncnn::Mat &im)
{
    img = im; img_w = img.w; img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);

    PNet(); //1st stage
    if (BBox1.empty()) return BBox1;
    NMS(BBox1, nms_threshold[0]);
    Refine(BBox1, img_h, img_w, true);
    //printf("PNet.Box = %d\n", (int)BBox1.size());
    
    RNet(); //2nd stage
    if (BBox2.empty()) return BBox2;
    NMS(BBox2, nms_threshold[1]);
    Refine(BBox2, img_h, img_w, true);
    //printf("RNet.Box = %d\n", (int)BBox2.size());
    
    ONet(); //3rd stage 
    if (BBox3.empty()) return BBox3;
    Refine(BBox3, img_h, img_w, true);
    NMS(BBox3, nms_threshold[2], "Min");
    //printf("ONet.Box = %d\n", (int)BBox3.size());
    
    for (auto &it:BBox3) Adjust(it);
    return BBox3;
}//end MTCNN::Detect

