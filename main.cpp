#include <iostream>

#include "yoloface.h"
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv){
    if (argc != 2){
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const float mean_vals[3] = {127.f, 127.f, 127.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

    YoloFace face_detector;
    face_detector.load("yolov5n", 640, mean_vals, norm_vals);
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    face_detector.detect(m, objects);
    face_detector.draw(m, objects);

    cv::imshow("output",m);
    cv::imwrite("output.jpg",m);
    cv::waitKey();

    return 0;
}