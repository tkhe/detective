#include <float.h>
#include <math.h>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>

#include "layer.h"
#include "net.h"

#define MAX_STRIDE 32

class HardSigmoid: public ncnn::Layer
{
public:
    HardSigmoid()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option& opt) const
    {
        int W = bottom_blob.w;
        int H = bottom_blob.h;
        int C = bottom_blob.c;

        top_blob.create(W, H, C, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    const float *ptr = bottom_blob.channel(c).row(h) + w;
                    float *outptr = top_blob.channel(c).row(h) + w;

                    *outptr = std::max(std::min((*ptr + 3.0) / 6.0, 1.0), 0.0);
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(HardSigmoid)

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
        {
            i++;
        }

        while (objects[j].prob < p)
        {
            j--;
        }

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j)
            {
                qsort_descent_inplace(objects, left, j);
            }
        }
        #pragma omp section
        {
            if (i < right)
            {
                qsort_descent_inplace(objects, i, right);
            }
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
    {
        return;
    }

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
            {
                continue;
            }

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
            {
                keep = 0;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat &cls_score, const ncnn::Mat &bbox_pred, int stride, const ncnn::Mat &in_pad, float prob_threshold, std::vector<Object> &objects)
{
    const int H = cls_score.h;
    const int W = cls_score.w;
    const int C = cls_score.c;

    const int reg_max = 17;

    const float prob_threshold_prev = -log((1 - prob_threshold) / (prob_threshold + 1e-5));

    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                float score = *(cls_score.channel(c).row(h) + w);
                if (score < prob_threshold_prev)
                {
                    continue;
                }
                score = sigmoid(score);

                ncnn::Mat bbox_pred_dis(reg_max, 4);
                for (int k = 0; k < 4; k++)
                {
                    for (int i = 0; i < reg_max; i++)
                    {
                        float *outptr = bbox_pred_dis.row(k) + i;
                        const float *ptr = bbox_pred.channel(k * reg_max + i).row(h) + w;
                        *outptr = *ptr;
                    }
                }
                {
                    ncnn::Layer *softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1);
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);
                    softmax->forward_inplace(bbox_pred_dis, opt);
                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float *dis_after_softmax = bbox_pred_dis.row(k);
                    for (int i = 0; i < reg_max; i++)
                    {
                        dis += i * dis_after_softmax[i];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float cx = (w + 0.5) * stride;
                float cy = (h + 0.5) * stride;

                float x0 = cx - pred_ltrb[0];
                float y0 = cy - pred_ltrb[1];
                float x1 = cx + pred_ltrb[2];
                float y1 = cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = (x1 - x0);
                obj.rect.height = (y1 - y0);
                obj.label = c;
                obj.prob = score;
                objects.push_back(obj);
            }
        }
    }
}

static int detect_ppyoloe(const cv::Mat &bgr, std::vector<Object> &objects)
{
    ncnn::Net ppyoloe;

    ppyoloe.opt.use_vulkan_compute = false;

    ppyoloe.register_custom_layer("mmcv.cnn.bricks.hsigmoid.HSigmoid", HardSigmoid_layer_creator);

    ppyoloe.load_param("../assets/ppyoloe-plus-small.param");
    ppyoloe.load_model("../assets/ppyoloe-plus-small.bin");

    const int target_size = 640;
    const float prob_threshold = 0.3f;
    const float nms_threshold = 0.45f;

    int width = bgr.cols;
    int height = bgr.rows;

    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = int(h * scale);
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = int(w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);

    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = ppyoloe.create_extractor();
    ex.input("in0", in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat cls_score_s8;
        ncnn::Mat bbox_pred_s8;

        ex.extract("out0", cls_score_s8);
        ex.extract("out3", bbox_pred_s8);

        std::vector<Object> objects8;
        generate_proposals(cls_score_s8, bbox_pred_s8, 8, in_pad, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat cls_score_s16;
        ncnn::Mat bbox_pred_s16;
        ex.extract("out1", cls_score_s16);
        ex.extract("out4", bbox_pred_s16);

        std::vector<Object> objects16;
        generate_proposals(cls_score_s16, bbox_pred_s16, 16, in_pad, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat cls_score_s32;
        ncnn::Mat bbox_pred_s32;
        ex.extract("out2", cls_score_s32);
        ex.extract("out5", bbox_pred_s32);

        std::vector<Object> objects32;
        generate_proposals(cls_score_s32, bbox_pred_s32, 32, in_pad, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold, true);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
        {
            y = 0;
        }
        if (x + label_size.width > image.cols)
        {
            x = image.cols - label_size.width;
        }

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imwrite("results.jpg", image);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_ppyoloe(m, objects);

    draw_objects(m, objects);

    return 0;
}
