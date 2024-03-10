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

#include "box_iou_rotated_utils.h"
#include "net.h"

#define MAX_STRIDE 32
#define PI 3.14159265

struct Object
{
    Point points[4];
    int label;
    float prob;
    float w;
    float h;
};

static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right)
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

static void qsort_descent_inplace(std::vector<Object> &objects)
{
    if (objects.empty())
    {
        return;
    }

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static float single_box_iou_rotated(const Object &obj1, const Object &obj2)
{
    float area1 = obj1.h * obj1.w;
    float area2 = obj2.h * obj2.w;
    if (area1 < 1e-14 || area2 < 1e-14)
    {
        return 0.f;
    }

    // fast rejection
    float a_xmin = FLT_MAX;
    float a_ymin = FLT_MAX;
    float a_xmax = -FLT_MAX;
    float a_ymax = -FLT_MAX;

    float b_xmin = FLT_MAX;
    float b_ymin = FLT_MAX;
    float b_xmax = -FLT_MAX;
    float b_ymax = -FLT_MAX;

    for (int i = 0; i < 4; i++)
    {
        a_xmin = std::min(obj1.points[i].x, a_xmin);
        a_ymin = std::min(obj1.points[i].y, a_ymin);
        a_xmax = std::max(obj1.points[i].x, a_xmax);
        a_ymax = std::max(obj1.points[i].y, a_ymax);
        b_xmin = std::min(obj2.points[i].x, b_xmin);
        b_ymin = std::min(obj2.points[i].y, b_ymin);
        b_xmax = std::max(obj2.points[i].x, b_xmax);
        b_ymax = std::max(obj2.points[i].y, b_ymax);
    }
    if (a_xmax <= b_xmin || a_ymax <= b_ymin || b_xmax <= a_xmin || b_ymax <= a_ymin)
    {
        return 0.f;
    }

    Point intersectPts[24];
    Point orderedPts[24];
    int num = get_intersection_points(obj1.points, obj2.points, intersectPts);
    if (num <= 2)
    {
        return 0.f;
    }
    int num_convex = convex_hull_graham(intersectPts, num, orderedPts, false);
    float area = polygon_area(orderedPts, num_convex);
    float iou = area / (area1 + area2 - area);
    return iou;
}

static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].w * objects[i].h;
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
            float IoU = single_box_iou_rotated(a, b);
            if (IoU > nms_threshold)
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

static void generate_proposals(
    const ncnn::Mat &cls_score,
    const ncnn::Mat &bbox_pred,
    const ncnn::Mat &angle,
    int stride,
    const ncnn::Mat &in_pad,
    float prob_threshold,
    std::vector<Object> &objects
)
{
    int H = cls_score.h;
    int W = cls_score.w;
    int C = cls_score.c;
    float prob_threshold_inv = -log((1 - prob_threshold) / (prob_threshold + 1e-5));
    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
        {
            float score = -FLT_MAX;
            int label = -1;
            for (int c = 0; c < C; c++)
            {
                float cur = *(cls_score.channel(c).row(h) + w);
                if (cur > score)
                {
                    score = cur;
                    label = c;
                }
            }

            if (score > prob_threshold_inv)
            {
                score = sigmoid(score);
                float rad = *((float *)angle.channel(0).row(h) + w);
                float sin_angle = sin(rad);
                float cos_angle = cos(rad);

                float l = *(bbox_pred.channel(0).row(h) + w) * stride;
                float t = *(bbox_pred.channel(1).row(h) + w) * stride;
                float r = *(bbox_pred.channel(2).row(h) + w) * stride;
                float b = *(bbox_pred.channel(3).row(h) + w) * stride;

                float rect_w = l + r;
                float rect_h = t + b;

                float offset_w = (r - l) / 2;
                float offset_h = (b - t) / 2;
                float offset_x = offset_w * cos_angle - offset_h * sin_angle;
                float offset_y = offset_w * sin_angle + offset_h * cos_angle;
                float ctr_x = w * stride + offset_x;
                float ctr_y = h * stride + offset_y;

                float hsin = sin_angle * 0.5;
                float hcos = cos_angle * 0.5;

                Object obj;
                obj.points[0].x = ctr_x - hsin * rect_h - hcos * rect_w;
                obj.points[0].y = ctr_y + hcos * rect_h - hsin * rect_w;
                obj.points[1].x = ctr_x + hsin * rect_h - hcos * rect_w;
                obj.points[1].y = ctr_y - hcos * rect_h - hsin * rect_w;
                obj.points[2].x = 2 * ctr_x - obj.points[0].x;
                obj.points[2].y = 2 * ctr_y - obj.points[0].y;
                obj.points[3].x = 2 * ctr_x - obj.points[1].x;
                obj.points[3].y = 2 * ctr_y - obj.points[1].y;

                obj.label = label;
                obj.prob = score;
                obj.h = rect_h;
                obj.w = rect_w;
                objects.push_back(obj);
            }
        }
    }
}

static int detect_rtmdet(const cv::Mat &bgr, std::vector<Object> &objects)
{
    ncnn::Net rtmdet;

    rtmdet.opt.use_vulkan_compute = false;

    rtmdet.load_param("../assets/rtmdet-r-tiny.param");
    rtmdet.load_model("../assets/rtmdet-r-tiny.bin");

    const int target_size = 1024;
    const float prob_threshold = 0.3f;
    const float nms_threshold = 0.3f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);

    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {1 / 57.375f, 1 / 57.12f, 1 / 58.395f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = rtmdet.create_extractor();
    ex.input("data", in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat cls_score_s8;
        ncnn::Mat bbox_pred_s8;
        ncnn::Mat angle_s8;
        ex.extract("cls_score_s8", cls_score_s8);
        ex.extract("bbox_pred_s8", bbox_pred_s8);
        ex.extract("angle_s8", angle_s8);

        std::vector<Object> objects8;
        generate_proposals(cls_score_s8, bbox_pred_s8, angle_s8, 8, in_pad, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }
    // stride 16
    {
        ncnn::Mat cls_score_s16;
        ncnn::Mat bbox_pred_s16;
        ncnn::Mat angle_s16;
        ex.extract("cls_score_s16", cls_score_s16);
        ex.extract("bbox_pred_s16", bbox_pred_s16);
        ex.extract("angle_s16", angle_s16);

        std::vector<Object> objects16;
        generate_proposals(cls_score_s16, bbox_pred_s16, angle_s16, 16, in_pad, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }
    // stride 32
    {
        ncnn::Mat cls_score_s32;
        ncnn::Mat bbox_pred_s32;
        ncnn::Mat angle_s32;
        ex.extract("cls_score_s32", cls_score_s32);
        ex.extract("bbox_pred_s32", bbox_pred_s32);
        ex.extract("angle_s32", angle_s32);

        std::vector<Object> objects32;
        generate_proposals(cls_score_s32, bbox_pred_s32, angle_s32, 32, in_pad, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // apply nms with nms_threshold
    std::vector<int> picked;

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    nms_sorted_bboxes(proposals, picked, nms_threshold, true);

    objects.resize(picked.size());
    for (int i = 0; i < picked.size(); i++)
    {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].points[0].x - (wpad / 2)) / scale;
        float y0 = (objects[i].points[0].y - (hpad / 2)) / scale;
        float x1 = (objects[i].points[1].x - (wpad / 2)) / scale;
        float y1 = (objects[i].points[1].y - (hpad / 2)) / scale;
        float x2 = (objects[i].points[2].x - (wpad / 2)) / scale;
        float y2 = (objects[i].points[2].y - (hpad / 2)) / scale;
        float x3 = (objects[i].points[3].x - (wpad / 2)) / scale;
        float y3 = (objects[i].points[3].y - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
        x2 = std::max(std::min(x2, (float)(img_w - 1)), 0.f);
        y2 = std::max(std::min(y2, (float)(img_h - 1)), 0.f);
        x3 = std::max(std::min(x3, (float)(img_w - 1)), 0.f);
        y3 = std::max(std::min(y3, (float)(img_h - 1)), 0.f);

        objects[i].points[0].x = x0;
        objects[i].points[0].y = y0;
        objects[i].points[1].x = x1;
        objects[i].points[1].y = y1;
        objects[i].points[2].x = x2;
        objects[i].points[2].y = y2;
        objects[i].points[3].x = x3;
        objects[i].points[3].y = y3;
    }
    return 0;
}

static void draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects)
{
    static const char* class_names[] = {
        "plane", "baseball-diamond", "bridge", "ground-track-field",
        "small-vehicle", "large-vehicle", "ship", "tennis-court",
        "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
        "harbor", "swimming-pool", "helicopter"
    };

    static const unsigned char colors[19][3] = {
        {54, 67, 244},
        {99, 30, 233},
        {176, 39, 156},
        {183, 58, 103},
        {181, 81, 63},
        {243, 150, 33},
        {244, 169, 3},
        {212, 188, 0},
        {136, 150, 0},
        {80, 175, 76},
        {74, 195, 139},
        {57, 220, 205},
        {59, 235, 255},
        {7, 193, 255},
        {0, 152, 255},
        {34, 87, 255},
        {72, 85, 121},
        {158, 158, 158},
        {139, 125, 96}
    };

    int color_index = 0;

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object &obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", obj.label, obj.prob, obj.points[0].x, obj.points[0].y, obj.points[1].x, obj.points[1].y, obj.points[2].x, obj.points[2].y, obj.points[3].x, obj.points[3].y);

        const unsigned char* color = colors[i % 19];

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::line(image, cv::Point2f(obj.points[0].x, obj.points[0].y), cv::Point2f(obj.points[1].x, obj.points[1].y), cc, 2);
        cv::line(image, cv::Point2f(obj.points[1].x, obj.points[1].y), cv::Point2f(obj.points[2].x, obj.points[2].y), cc, 2);
        cv::line(image, cv::Point2f(obj.points[2].x, obj.points[2].y), cv::Point2f(obj.points[3].x, obj.points[3].y), cc, 2);
        cv::line(image, cv::Point2f(obj.points[3].x, obj.points[3].y), cv::Point2f(obj.points[0].x, obj.points[0].y), cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.points[0].x;
        int y = obj.points[0].y - label_size.height - baseLine;
        if (y < 0)
        {
            y = 0;
        }
        if (x + label_size.width > image.cols)
        {
            x = image.cols - label_size.width;
        }

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
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

    detect_rtmdet(m, objects);

    draw_objects(m, objects);
    return 0;
}
