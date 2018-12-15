#ifndef OBJECT_DETECTION_UTILS_H
#define OBJECT_DETECTION_UTILS_H

#include <opencv2/opencv.hpp>


void AdaptiveFindThreshold(const cv::Mat& image, double* low, double* high, int aperture_size =3);

#endif