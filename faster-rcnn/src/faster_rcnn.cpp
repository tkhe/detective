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

#include "net.h"

#define MAX_STRIDE 32

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

static ncnn::Mat generate_anchors(int size, const ncnn::Mat ratios)
{
    int num_ratio = ratios.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio);

    int area = size * size;

    for (int i = 0; i < num_ratio; i++)
    {
        float ratio = ratios[i];

        float w = sqrt(area / ratio);
        float h = w * ratio;

        float *anchor = anchors.row(i);

        anchor[0] = -w / 2.0;
        anchor[1] = -h / 2.0;
        anchor[2] = w / 2.0;
        anchor[3] = h / 2.0;
    }

    return anchors;
}

static int generate_rpn_proposals(const ncnn::Mat &cls_score, const ncnn::Mat &bbox_pred, const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad, float prob_threshold, std::vector<Object> &objects)
{
    const int H = cls_score.h;
    const int W = cls_score.w;
    const int C = cls_score.c;

    for (int h = 0; h < H; h++)
    {
        for (int w = 0; w < W; w++)
        {
            for (int c = 0; c < C; c++)
            {
                float score = sigmoid(*(cls_score.channel(c).row(h) + w));
                if (score < prob_threshold)
                {
                    continue;
                }

                float anchor[4];
                anchor[0] = *(anchors.row(c)) + w * stride;
                anchor[1] = *(anchors.row(c) + 1) + h * stride;
                anchor[2] = *(anchors.row(c) + 2) + w * stride;
                anchor[3] = *(anchors.row(c) + 3) + h * stride;

                float anchor_w = anchor[2] - anchor[0];
                float anchor_h = anchor[3] - anchor[1];
                float anchor_cx = anchor[0] + 0.5 * anchor_w;
                float anchor_cy = anchor[1] + 0.5 * anchor_h;

                float dx = *(bbox_pred.channel(4 * c).row(h) + w) / 1.0;
                float dy = *(bbox_pred.channel(4 * c + 1).row(h) + w) / 1.0;
                float dw = std::min(*(bbox_pred.channel(4 * c + 2).row(h) + w) / 1.0, 4.13516);
                float dh = std::min(*(bbox_pred.channel(4 * c + 3).row(h) + w) / 1.0, 4.13516);

                float pred_cx = dx * anchor_w + anchor_cx;
                float pred_cy = dy * anchor_h + anchor_cy;
                float pred_w = exp(dw) * anchor_w;
                float pred_h = exp(dh) * anchor_h;

                Object obj;
                obj.rect.x = pred_cx - pred_w / 2;
                obj.rect.y = pred_cy - pred_h / 2;
                obj.rect.width = pred_w;
                obj.rect.height = pred_h;
                obj.label = 0;
                obj.prob = score;
                objects.push_back(obj);
            }
        }
    }

    return 0;
}

static int assign_proposal_to_levels(const Object &obj)
{
    float box_sizes = sqrt(obj.rect.width * obj.rect.height);
    int canonical_level = 4;
    int min_level = 2;
    int max_level = 5;
    int canonical_box_size = 224;

    int level_assignment = (int)floor(canonical_level + log2(box_sizes / canonical_box_size + 1e-8));
    level_assignment = std::max(min_level, std::min(max_level, level_assignment));

    return level_assignment - min_level;
}

static int detect_faster_rcnn(const cv::Mat &bgr, std::vector<Object> &objects)
{
    ncnn::Net rpn;

    rpn.opt.use_vulkan_compute = false;

    rpn.load_param("../assets/rpn.param");
    rpn.load_model("../assets/rpn.bin");

    ncnn::Net rcnn;

    rcnn.opt.use_vulkan_compute = false;

    rcnn.load_param("../assets/rcnn.param");
    rcnn.load_model("../assets/rcnn.bin");

    const int target_size = 640;
    const float rpn_prob_threshold = 0.05f;
    const float rpn_nms_threshold = 0.7f;
    const int rpn_pre_nms_topk = 1000;
    const int rpn_post_nms_topk = 1000;

    const int rcnn_prob_threshold = 0.3f;
    const float rcnn_nms_threshold = 0.45f;

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

    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {1.f, 1.f, 1.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = rpn.create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat fpn_s4;
    ncnn::Mat fpn_s8;
    ncnn::Mat fpn_s16;
    ncnn::Mat fpn_s32;

    ex.extract("out10", fpn_s4);
    ex.extract("out11", fpn_s8);
    ex.extract("out12", fpn_s16);
    ex.extract("out13", fpn_s32);

    std::vector<Object> proposals;

    // stride 4
    {
        ncnn::Mat cls_score_s4;
        ncnn::Mat bbox_pred_s4;

        ex.extract("out0", cls_score_s4);
        ex.extract("out5", bbox_pred_s4);

        ncnn::Mat aspect_ratios(3);
        aspect_ratios[0] = 0.5;
        aspect_ratios[1] = 1.0;
        aspect_ratios[2] = 2.0;

        const int size = 32;

        ncnn::Mat anchors = generate_anchors(size, aspect_ratios);

        std::vector<Object> objects4;
        generate_rpn_proposals(cls_score_s4, bbox_pred_s4, anchors, 4, in_pad, rpn_prob_threshold, objects4);

        if (rpn_pre_nms_topk > 0 && objects4.size() > rpn_pre_nms_topk)
        {
            objects4.resize(rpn_pre_nms_topk);
        }

        proposals.insert(proposals.end(), objects4.begin(), objects4.end());
    }

    // stride 8
    {
        ncnn::Mat cls_score_s8;
        ncnn::Mat bbox_pred_s8;

        ex.extract("out1", cls_score_s8);
        ex.extract("out6", bbox_pred_s8);

        ncnn::Mat aspect_ratios(3);
        aspect_ratios[0] = 0.5;
        aspect_ratios[1] = 1.0;
        aspect_ratios[2] = 2.0;

        const int size = 64;

        ncnn::Mat anchors = generate_anchors(size, aspect_ratios);

        std::vector<Object> objects8;
        generate_rpn_proposals(cls_score_s8, bbox_pred_s8, anchors, 8, in_pad, rpn_prob_threshold, objects8);

        if (rpn_pre_nms_topk > 0 && objects8.size() > rpn_pre_nms_topk)
        {
            objects8.resize(rpn_pre_nms_topk);
        }

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat cls_score_s16;
        ncnn::Mat bbox_pred_s16;

        ex.extract("out2", cls_score_s16);
        ex.extract("out7", bbox_pred_s16);

        ncnn::Mat aspect_ratios(3);
        aspect_ratios[0] = 0.5;
        aspect_ratios[1] = 1.0;
        aspect_ratios[2] = 2.0;

        const int size = 128;

        ncnn::Mat anchors = generate_anchors(size, aspect_ratios);

        std::vector<Object> objects16;
        generate_rpn_proposals(cls_score_s16, bbox_pred_s16, anchors, 16, in_pad, rpn_prob_threshold, objects16);

        if (rpn_pre_nms_topk > 0 && objects16.size() > rpn_pre_nms_topk)
        {
            objects16.resize(rpn_pre_nms_topk);
        }

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat cls_score_s32;
        ncnn::Mat bbox_pred_s32;

        ex.extract("out3", cls_score_s32);
        ex.extract("out8", bbox_pred_s32);

        ncnn::Mat aspect_ratios(3);
        aspect_ratios[0] = 0.5;
        aspect_ratios[1] = 1.0;
        aspect_ratios[2] = 2.0;

        const int size = 256;

        ncnn::Mat anchors = generate_anchors(size, aspect_ratios);

        std::vector<Object> objects32;
        generate_rpn_proposals(cls_score_s32, bbox_pred_s32, anchors, 32, in_pad, rpn_prob_threshold, objects32);

        if (rpn_pre_nms_topk > 0 && objects32.size() > rpn_pre_nms_topk)
        {
            objects32.resize(rpn_pre_nms_topk);
        }

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // stride 64
    {
        ncnn::Mat cls_score_s64;
        ncnn::Mat bbox_pred_s64;

        ex.extract("out4", cls_score_s64);
        ex.extract("out9", bbox_pred_s64);

        ncnn::Mat aspect_ratios(3);
        aspect_ratios[0] = 0.5;
        aspect_ratios[1] = 1.0;
        aspect_ratios[2] = 2.0;

        const int size = 512;

        ncnn::Mat anchors = generate_anchors(size, aspect_ratios);

        std::vector<Object> objects64;
        generate_rpn_proposals(cls_score_s64, bbox_pred_s64, anchors, 64, in_pad, rpn_prob_threshold, objects64);

        if (rpn_pre_nms_topk > 0 && objects64.size() > rpn_pre_nms_topk)
        {
            objects64.resize(rpn_pre_nms_topk);
        }

        proposals.insert(proposals.end(), objects64.begin(), objects64.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, rpn_nms_threshold, true);

    int count = std::min((int)picked.size(), rpn_post_nms_topk);

    // objects
    std::vector<Object> rcnn_objects;
    rcnn_objects.resize(count);

    std::vector<Object> rcnn_proposals;
    for (int i = 0; i < count; i++)
    {
        rcnn_objects[i] = proposals[picked[i]];

        ncnn::Mat cls_score;
        ncnn::Mat bbox_pred;

        std::vector<ncnn::Mat> roi_align_inputs;
        std::vector<ncnn::Mat> roi_align_outputs;

        roi_align_outputs.resize(1);

        ncnn::Mat rois(4);
        rois[0] = rcnn_objects[i].rect.x;
        rois[1] = rcnn_objects[i].rect.y;
        rois[2] = rcnn_objects[i].rect.x + rcnn_objects[i].rect.width;
        rois[3] = rcnn_objects[i].rect.y + rcnn_objects[i].rect.height;

        int level_assignment = assign_proposal_to_levels(rcnn_objects[i]);
        float spatial_scale = 0.f;
        if (level_assignment == 0)
        {
            roi_align_inputs.push_back(fpn_s4);
            roi_align_inputs.push_back(rois);
            spatial_scale = 1 / 4.0f;
        }
        else if (level_assignment == 1)
        {
            roi_align_inputs.push_back(fpn_s8);
            roi_align_inputs.push_back(rois);
            spatial_scale = 1 / 8.0f;
        }
        else if (level_assignment == 2)
        {
            roi_align_inputs.push_back(fpn_s16);
            roi_align_inputs.push_back(rois);
            spatial_scale = 1 / 16.0f;
        }
        else
        {
            roi_align_inputs.push_back(fpn_s32);
            roi_align_inputs.push_back(rois);
            spatial_scale = 1 / 32.0f;
        }

        {
            ncnn::Layer *roi_align = ncnn::create_layer("ROIAlign");

            ncnn::ParamDict pd;
            pd.set(0, 7);
            pd.set(1, 7);
            pd.set(2, spatial_scale);
            pd.set(3, 0);
            pd.set(4, true);
            pd.set(5, 1);
            roi_align->load_param(pd);

            ncnn::Option opt;
            opt.num_threads = 1;
            opt.use_packing_layout = false;

            roi_align->create_pipeline(opt);
            roi_align->forward(roi_align_inputs, roi_align_outputs, opt);
            roi_align->destroy_pipeline(opt);
        }

        ncnn::Mat roi_feat = roi_align_outputs[0];

        ncnn::Extractor ex2 = rcnn.create_extractor();
        ex2.input("in0", roi_feat);

        ex2.extract("out0", cls_score);
        ex2.extract("out1", bbox_pred);

        const int num_classes = cls_score.w - 1;
        int label = -1;
        float score = 0.f;
        for (int c = 0; c <= num_classes; c++)
        {
            const float cur = *(cls_score.row(0) + c);
            if (cur > score)
            {
                score = cur;
                label = c;
            }
        }
        if (label == num_classes || score < rcnn_prob_threshold)
        {
            continue;
        }

        float dx = *(bbox_pred.row(0) + label * 4) / 10.f;
        float dy = *(bbox_pred.row(0) + label * 4 + 1) / 10.f;
        float dw = *(bbox_pred.row(0) + label * 4 + 2) / 5.f;
        float dh = *(bbox_pred.row(0) + label * 4 + 3) / 5.f;
        dw = std::min(dw, 4.13516f);
        dh = std::min(dh, 4.13516f);

        float proposal_cx = rcnn_objects[i].rect.x + rcnn_objects[i].rect.width / 2.f;
        float proposal_cy = rcnn_objects[i].rect.y + rcnn_objects[i].rect.height / 2.f;
        float pred_cx = dx * rcnn_objects[i].rect.width + proposal_cx;
        float pred_cy = dy * rcnn_objects[i].rect.height + proposal_cy;
        float pred_w = exp(dw) * rcnn_objects[i].rect.width;
        float pred_h = exp(dh) * rcnn_objects[i].rect.height;

        rcnn_objects[i].label = label;
        rcnn_objects[i].prob = score;
        rcnn_objects[i].rect.x = pred_cx - pred_w / 2;
        rcnn_objects[i].rect.y = pred_cy - pred_h / 2;
        rcnn_objects[i].rect.width = pred_w;
        rcnn_objects[i].rect.height = pred_h;

        rcnn_proposals.push_back(rcnn_objects[i]);
    }

    qsort_descent_inplace(rcnn_proposals);

    nms_sorted_bboxes(rcnn_proposals, picked, rcnn_nms_threshold, true);

    count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = rcnn_proposals[picked[i]];

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
    detect_faster_rcnn(m, objects);

    draw_objects(m, objects);

    return 0;
}
