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
#define NUM_KEYPOINTS 17

class Focus: public ncnn::Layer
{
public:
    Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Focus)

struct Keypoint
{
    cv::Point2f p;
    float prob;
};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    Keypoint kpts[NUM_KEYPOINTS];
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

static void generate_proposals(
    const ncnn::Mat &cls_score,
    const ncnn::Mat &bbox_pred,
    const ncnn::Mat &objectness,
    const ncnn::Mat &kpts_offset,
    const ncnn::Mat &kpts_vis,
    int stride,
    const ncnn::Mat &in_pad,
    float prob_threshold,
    std::vector<Object> &objects
)
{
    const int H = cls_score.h;
    const int W = cls_score.w;
    const int C = cls_score.c;

    const float prob_threshold_prev = -log((1 - prob_threshold) / (prob_threshold + 1e-5));

    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
        {
            float obj_score = *(objectness.channel(0).row(h) + w);
            if (obj_score < prob_threshold_prev)
            {
                continue;
            }

            obj_score = sigmoid(obj_score);

            for (int c = 0; c < C; c++)
            {
                float score = obj_score * sigmoid(*(cls_score.channel(c).row(h) + w));
                if (score < prob_threshold)
                {
                    continue;
                }
                float pred_x = *(bbox_pred.channel(0).row(h) + w);
                float pred_y = *(bbox_pred.channel(1).row(h) + w);
                float pred_w = *(bbox_pred.channel(2).row(h) + w);
                float pred_h = *(bbox_pred.channel(3).row(h) + w);

                float cx = (pred_x + w) * stride;
                float cy = (pred_y + h) * stride;
                float width = exp(pred_w) * stride;
                float height = exp(pred_h) * stride;

                Object obj;
                obj.rect.x = cx - width / 2;
                obj.rect.y = cy - height / 2;
                obj.rect.width = width;
                obj.rect.height = height;
                obj.label = c;
                obj.prob = score;

                for (int i = 0; i < NUM_KEYPOINTS; i++)
                {
                    float coords_x = *(kpts_offset.channel(i * 2).row(h) + w);
                    float coords_y = *(kpts_offset.channel(i * 2 + 1).row(h) + w);
                    float vis = sigmoid(*(kpts_vis.channel(i).row(h) + w));

                    coords_x = (coords_x + w) * stride;
                    coords_y = (coords_y + h) * stride;

                    obj.kpts[i].p = cv::Point2f(coords_x, coords_y);
                    obj.kpts[i].prob = vis;
                }

                objects.push_back(obj);
            }
        }
    }
}

static int detect_yoloxpose(const cv::Mat &bgr, std::vector<Object> &objects)
{
    ncnn::Net yoloxpose;

    yoloxpose.opt.use_vulkan_compute = false;

    yoloxpose.register_custom_layer("mmdet.models.backbones.csp_darknet.Focus", Focus_layer_creator);

    yoloxpose.load_param("../assets/yolox-pose-tiny.param");
    yoloxpose.load_model("../assets/yolox-pose-tiny.bin");

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

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, width, height, w, h);

    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.f, 1.f, 1.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yoloxpose.create_extractor();
    ex.input("in0", in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat cls_score_s8;
        ncnn::Mat bbox_pred_s8;
        ncnn::Mat objectness_s8;
        ncnn::Mat kpts_offset_s8;
        ncnn::Mat kpts_vis_s8;

        ex.extract("out0", cls_score_s8);
        ex.extract("out3", bbox_pred_s8);
        ex.extract("out6", objectness_s8);
        ex.extract("out9", kpts_offset_s8);
        ex.extract("out12", kpts_vis_s8);

        std::vector<Object> objects8;
        generate_proposals(cls_score_s8, bbox_pred_s8, objectness_s8, kpts_offset_s8, kpts_vis_s8, 8, in_pad, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat cls_score_s16;
        ncnn::Mat bbox_pred_s16;
        ncnn::Mat objectness_s16;
        ncnn::Mat kpts_offset_s16;
        ncnn::Mat kpts_vis_s16;

        ex.extract("out1", cls_score_s16);
        ex.extract("out4", bbox_pred_s16);
        ex.extract("out7", objectness_s16);
        ex.extract("out10", kpts_offset_s16);
        ex.extract("out13", kpts_vis_s16);

        std::vector<Object> objects16;
        generate_proposals(cls_score_s16, bbox_pred_s16, objectness_s16, kpts_offset_s16, kpts_vis_s16, 16, in_pad, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat cls_score_s32;
        ncnn::Mat bbox_pred_s32;
        ncnn::Mat objectness_s32;
        ncnn::Mat kpts_offset_s32;
        ncnn::Mat kpts_vis_s32;

        ex.extract("out2", cls_score_s32);
        ex.extract("out5", bbox_pred_s32);
        ex.extract("out8", objectness_s32);
        ex.extract("out11", kpts_offset_s32);
        ex.extract("out14", kpts_vis_s32);

        std::vector<Object> objects32;
        generate_proposals(cls_score_s32, bbox_pred_s32, objectness_s32, kpts_offset_s32, kpts_vis_s32, 32, in_pad, prob_threshold, objects32);

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

        for (int k = 0; k < NUM_KEYPOINTS; k++)
        {
            float x = (objects[i].kpts[k].p.x - (wpad / 2)) / scale;
            float y = (objects[i].kpts[k].p.y - (wpad / 2)) / scale;
            x = std::max(std::min(x, x1), x0);
            y = std::max(std::min(y, y1), y0);
            objects[i].kpts[k].p = cv::Point2f(x, y);
        }

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

    static const int joint_pairs[14][2] = {
        {1, 3}, {2, 4}, {1, 0}, {0, 2}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
    };

    static const int joint_pairs_colors[14][3] = {
        {102, 204, 255}, {51, 153, 255}, {102, 0, 204}, {51, 102, 255}, {255, 128, 0}, {153, 255, 204}, {128, 229, 255}, {153, 255, 153},
        {102, 255, 224}, {255, 102, 0}, {255, 255, 77}, {153, 255, 204}, {191, 255, 128}, {255, 195, 77}
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        for (int k = 0; k < NUM_KEYPOINTS; k++)
        {
            const Keypoint &keypoint = obj.kpts[k];

            fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

            if (keypoint.prob < 0.2f)
            {
                continue;
            }

            cv::circle(image, keypoint.p, 3, cv::Scalar(0, 0, 255), -1);
        }

        for (int k = 0; k < 14; k++)
        {
            const Keypoint &p1 = obj.kpts[joint_pairs[k][0]];
            const Keypoint &p2 = obj.kpts[joint_pairs[k][1]];

            if (p1.prob < 0.2f || p2.prob < 0.2f)
            {
                continue;
            }

            cv::line(image, p1.p, p2.p, cv::Scalar(joint_pairs_colors[k][2], joint_pairs_colors[k][1], joint_pairs_colors[k][0]), 2);
        }

        float nose_x = obj.kpts[0].p.x;
        float nose_y = obj.kpts[0].p.y;
        float left_shoulder_x = obj.kpts[5].p.x;
        float left_shoulder_y = obj.kpts[5].p.y;
        float right_shoulder_x = obj.kpts[6].p.x;
        float right_shoulder_y = obj.kpts[6].p.y;
        float left_hip_x = obj.kpts[11].p.x;
        float left_hip_y = obj.kpts[11].p.y;
        float right_hip_x = obj.kpts[12].p.x;
        float right_hip_y = obj.kpts[12].p.y;

        cv::line(image, cv::Point2f(nose_x, nose_y), cv::Point2f((left_shoulder_x + right_shoulder_x) / 2, (left_shoulder_y + right_shoulder_y) / 2), cv::Scalar(0, 0, 255), 2);
        cv::line(image, cv::Point2f((left_hip_x + right_hip_x) / 2, (left_hip_y + right_hip_y) / 2), cv::Point2f((left_shoulder_x + right_shoulder_x) / 2, (left_shoulder_y + right_shoulder_y) / 2), cv::Scalar(0, 0, 255), 2);

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
    detect_yoloxpose(m, objects);

    draw_objects(m, objects);

    return 0;
}
