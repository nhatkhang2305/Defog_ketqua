#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/imgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaimgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaarithm.hpp>
#include <iostream>
#include <chrono>
#include </usr/local/include/opencv4/opencv2/core/cuda.hpp>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Giả sử cấu trúc Pixel được định nghĩa như sau:
struct Pixel {
    int x;
    int y;
    uchar value;
    Pixel(int i, int j, uchar v) : x(i), y(j), value(v) {}
};

// void get_dark_channel(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, int rows, int cols, int channels, int radius) {
    
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             int rmin = cv::max(0, i - radius);
//             int rmax = cv::min(i + radius, rows - 1);
//             int cmin = cv::max(0, j - radius);
//             int cmax = cv::min(j + radius, cols - 1);
//             double min_val = 255;
//             for (int x = rmin; x <= rmax; x++) {
//                 for (int y = cmin; y <= cmax; y++) {
//                     cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(x)[y];
//                     uchar b = tmp[0];
//                     uchar g = tmp[1];
//                     uchar r = tmp[2];
//                     uchar minpixel = b > g ? ((g>r) ? r : g) : ((b > r) ? r : b); //tim min rgb
//                     min_val = cv::min((double)minpixel, min_val); //so sanh mini vs min_val cap nhat min_val
//                 }
//             }
//             tmp_vec.push_back(Pixel(i, j, uchar(min_val)));
//         }
//     }
//     // Mã nguồn hàm get_dark_channel (giữ nguyên như bạn đã cung cấp)
// }

int main() {
    // Đọc hình ảnh
    cv::Mat src = cv::imread("suong1.jpg");
    //const cv::Mat *p_src ;
    if (src.empty()) {
        std::cerr << "Lỗi: Không thể đọc hình ảnh!" << std::endl;
        return -1;
    }

    // Xác định kích thước hình ảnh và bán kính patch
    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();
    int radius = 7; // Bạn có thể điều chỉnh bán kính ở đây
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_elapsed;
    // Tạo vector để lưu trữ kết quả
    std::vector<uchar> dark_channel(rows * cols);

    // Tính dark channel
    //get_dark_channel(&src, tmp_vec, rows, cols, channels, radius);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int rmin = cv::max(0, i - radius);
            int rmax = cv::min(i + radius, rows - 1);
            int cmin = cv::max(0, j - radius);
            int cmax = cv::min(j + radius, cols - 1);
            double min_val = 255;
            for (int x = rmin; x <= rmax; x++) {
                for (int y = cmin; y <= cmax; y++) {
                    cv::Vec3b tmp = src.ptr<cv::Vec3b>(x)[y];
                    uchar b = tmp[0];
                    uchar g = tmp[1];
                    uchar r = tmp[2];
                    uchar minpixel = b > g ? ((g>r) ? r : g) : ((b > r) ? r : b); //tim min rgb
                    min_val = cv::min((double)minpixel, min_val); //so sanh mini vs min_val cap nhat min_val
                }
            }
            dark_channel[i * cols + j] = uchar(min_val);
        }
    }

    // Tạo hình ảnh kết quả (dạng grayscale)
    cv::Mat dst = cv::Mat(rows, cols, CV_8UC1);

    // Gán các giá trị dark channel vào hình ảnh kết quả
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst.at<uchar>(i, j) = dark_channel[i * cols + j];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    time_elapsed = end - start;
    std::cout << "Thời gian thực thi: " << time_elapsed.count() << " giây" << std::endl;
    // Lưu hình ảnh kết quả
    cv::imwrite("dark_channel_cpu.jpg", dst);

    return 0;
}