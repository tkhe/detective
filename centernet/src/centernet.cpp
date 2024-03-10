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

static void generate_proposals(const ncnn::Mat &heatmap, const ncnn::Mat &wh, const ncnn::Mat &offset, int stride, const ncnn::Mat &in_pad, float prob_threshold, std::vector<Object> &objects)
{
    const int H = heatmap.h;
    const int W = heatmap.w;
    const int C = heatmap.c;

    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                float score = *(heatmap.channel(c).row(h) + w);
                if (score < prob_threshold)
                {
                    continue;
                }

                float width = *(wh.channel(0).row(h) + w) * stride;
                float height = *(wh.channel(1).row(h) + w) * stride;
                float cx = w * stride + *(offset.channel(0).row(h) + w);
                float cy = h * stride + *(offset.channel(1).row(h) + w);
                float x0 = cx - width * 0.5;
                float y0 = cy - height * 0.5;

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = width;
                obj.rect.height = height;
                obj.label = c;
                obj.prob = score;
                objects.push_back(obj);
            }
        }
    }
}

static int detect_centernet(const cv::Mat &bgr, std::vector<Object> &objects)
{
    ncnn::Net centernet;

    centernet.opt.use_vulkan_compute = false;

    centernet.load_param("../assets/centernet-r18.param");
    centernet.load_model("../assets/centernet-r18.bin");

    const int target_size = 512;
    const float prob_threshold = 0.3f;
    const int topk = 100;

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

    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {1 / 58.395f, 1 / 57.12f, 1 / 57.375f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = centernet.create_extractor();
    ex.input("data", in_pad);

    std::vector<Object> proposals;

    {
        ncnn::Mat heatmap;
        ncnn::Mat wh;
        ncnn::Mat offset;

        ex.extract("heatmap", heatmap);
        ex.extract("wh", wh);
        ex.extract("offset", offset);

        std::vector<Object> objects;
        generate_proposals(heatmap, wh, offset, 4, in_pad, prob_threshold, objects);

        proposals.insert(proposals.end(), objects.begin(), objects.end());
    }
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    int count = std::min((int)proposals.size(), topk);

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[i];

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
    detect_centernet(m, objects);

    draw_objects(m, objects);

    return 0;
}
