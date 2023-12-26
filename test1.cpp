#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/imgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaimgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaarithm.hpp>
#include <iostream>
#include <chrono>
#include </usr/local/include/opencv4/opencv2/core/cuda.hpp>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>

using namespace cv;
using namespace cv::cuda;

int main() {
    // Khởi tạo hình ảnh thực tế
    const char* img_path = "suong1.jpg";
    Mat in_img = imread(img_path);
    cuda::GpuMat src_gpu;
    src_gpu.upload(in_img);
    GpuMat src_gpu_gray;
    cv::cuda::cvtColor(src_gpu, src_gpu_gray, cv::COLOR_BGR2GRAY);
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_elapsed;
    // Xác định path size
    int pathSize = 7;

    // Tạo ma trận GPU để lưu trữ kết quả
    Mat dst(in_img.size(), CV_8UC1);
    start = std::chrono::high_resolution_clock::now();
    // Duyệt qua từng pixel trong hình ảnh
    for (int i = 0; i < in_img.rows; i++) {
        for (int j = 0; j < in_img.cols; j++) {
            // Xác định phạm vi patch xung quanh pixel hiện tại
            int rmin = std::max(0, i - pathSize);
            int rmax = std::min(i + pathSize, in_img.rows - 1);
            int cmin = std::max(0, j - pathSize);
            int cmax = std::min(j + pathSize, in_img.cols - 1);

            // Cắt patch từ hình ảnh và sao chép nó vào một ma trận GPU riêng biệt
            cv::cuda::GpuMat patch = src_gpu_gray(cv::Rect(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1));

            // Tìm giá trị tối thiểu trong patch
            double minVal, maxVal;
            //cv::Point minLoc, maxLoc;
            //cv::cuda::minMaxLoc(patch, &minVal, &maxVal, &minLoc, &maxLoc);
            cv::cuda::minMaxLoc(patch, &minVal, NULL, NULL, NULL);
            // // Lưu giá trị tối thiểu vào hình ảnh kết quả
            dst.ptr<uchar>(i)[j] = static_cast<uchar>(minVal);
            // uchar* dst_ptr = dst.ptr<uchar>(i);
            // dst_ptr[j] = minVal;
            // //dst.at<uchar>(i, j) = minVal;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    time_elapsed = end - start;
    std::cout << "Thời gian thực thi: " << time_elapsed.count() << " giây" << std::endl;
    // Lưu kết quả
    imwrite("dark_channel.jpg", dst);

    return 0;
}








