#include "hazeremoval.h"
#include </usr/local/include/opencv4/opencv2/core/cuda.hpp>
#include <iostream>
#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/imgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaimgproc.hpp>
using namespace cv;
using namespace std;

CHazeRemoval::CHazeRemoval() {
	rows = 0;
	cols = 0;
	channels = 0;
} // ham khoi tao lop co cac bien thanh vien

CHazeRemoval::~CHazeRemoval() {

} // ham huy de giai phong bo nho nhung o day de trong

bool CHazeRemoval::InitProc(int width, int height, int nChannels) {
	bool ret = false;
	rows = height;
	cols = width;
	channels = nChannels;

	if (width > 0 && height > 0 && nChannels == 3) ret = true;
	return ret;
} //khoi tao gia tri bool ret cho biet tao co thanh cong khong // kieu bool l  kieu chi co true or false

bool CHazeRemoval::Process(const unsigned char* indata, unsigned char* outdata, int width, int height, int nChannels) {
    bool ret = true;
    if (!indata || !outdata) {
        ret = false;
    }

    rows = height;
    cols = width;
    channels = nChannels;
    int radius = 7;
    double omega = 0.95;
    double t0 = 0.1;
    Vec3d Alight_host;
    cout <<"hi" ;
    //Upload input image to GPU
    cuda::GpuMat src_gpu;
    src_gpu.upload(Mat(rows, cols, CV_8UC3, (void*)indata));

    // Allocate GPU memory for other matrices
    cuda::GpuMat dst_gpu(rows, cols, CV_64FC3);
    cuda::GpuMat tran_gpu(rows, cols, CV_64FC1);

 	cuda::GpuMat dark_channel_gpu;

    // // Perform operations on GPU
    get_dark_channel_gpu(&src_gpu, dark_channel_gpu, rows, cols, channels, radius);
    // get_air_light_gpu(&src_gpu, dark_channel_gpu, Alight_host, rows, cols, channels);
    // get_transmission_gpu(&src_gpu, &tran_gpu, &Alight_host, rows, cols, channels, radius, omega);
    // recover_gpu(&src_gpu, &tran_gpu, &dst_gpu, &Alight_host, rows, cols, channels, t0);
}
//     // Download result from GPU to CPU
//     Mat dst_host;
//     dst_gpu.download(dst_host);

//     // Copy data from Mat to outdata
//     for (int i = 0; i < rows * cols * channels; i++) {
//         outdata[i] = static_cast<unsigned char>(*((double*)(dst_host.data) + i));
//     }

//     return ret;
// }

// bool sort_fun(const Pixel&a, const Pixel&b) {
// 	return a.val > b.val;
// }// true neu a>b va nguoc lai

void get_dark_channel_gpu(const cv::cuda::GpuMat* p_src, cv::cuda::GpuMat& dark_channel_gpu, int rows, int cols, int channels, int radius) {
    // Convert the input image to grayscale on the GPU
    cv::cuda::GpuMat src_gray;
    cv::cuda::cvtColor(*p_src, src_gray, cv::COLOR_BGR2GRAY);

    // Allocate GPU memory for dark channel
    dark_channel_gpu.create(rows, cols, CV_8UC1);

    // Compute dark channel on GPU
    cv::cuda::GpuMat min_channel_gpu(rows, cols, CV_8UC1);
    cv::cuda::GpuMat temp_channel_gpu(rows, cols, CV_8UC1);
    
    // Loop over the image pixels
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         // Define the local neighborhood
    //         int rmin = cv::max(0, i - radius);
    //         int rmax = cv::min(i + radius, rows - 1);
    //         int cmin = cv::max(0, j - radius);
    //         int cmax = cv::min(j + radius, cols - 1);

    //         // Extract the corresponding region on the GPU
    //         cv::Rect roi(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1);
    //         cv::cuda::GpuMat region_gpu(src_gray, roi);

    //         // Compute the minimum value in the region on the GPU
    //         double min_val;
    //         double max_val;
    //         cv::Point min_loc;
    //         cv::Point max_loc;
    //         cv::minMaxLoc(region_gpu, &min_val, &max_val, &min_loc, &max_loc);

    //         // Set the result in the dark channel matrix
    //         dark_channel_gpu.ptr<uchar>(i)[j] = static_cast<uchar>(min_val);
    //     }
    // }
}

// void get_air_light_gpu(const cv::cuda::GpuMat* p_src, cv::cuda::GpuMat& dark_channel_gpu, Vec3d& Alight_host, int rows, int cols, int channels) {
//     // Download dark channel from GPU to CPU for sorting
//     Mat dark_channel_cpu;
//     dark_channel_gpu.download(dark_channel_cpu);

//     // Convert dark channel to a linear vector for sorting
//     std::vector<Pixel> tmp_vec;
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             tmp_vec.push_back(Pixel(i, j, dark_channel_cpu.at<uchar>(i, j)));
//         }
//     }

//     // Sort the vector
//     std::sort(tmp_vec.begin(), tmp_vec.end(), sort_fun);

//     // Calculate the average air light from the sorted dark channel
//     int num = static_cast<int>(rows * cols * 0.001);
//     double A_sum[3] = {0.0};
//     for (int cnt = 0; cnt < num; cnt++) {
//         cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(tmp_vec[cnt].i)[tmp_vec[cnt].j];
//         A_sum[0] += tmp[0];
//         A_sum[1] += tmp[1];
//         A_sum[2] += tmp[2];
//     }

//     for (int i = 0; i < 3; i++) {
//         Alight_host[i] = A_sum[i] / num;
//     }
// }

// void get_transmission_gpu(const cv::cuda::GpuMat* p_src, cv::cuda::GpuMat* p_tran_gpu, const Vec3d* p_Alight, int rows, int cols, int channels, int radius, double omega) {
//     p_tran_gpu->create(rows, cols, CV_64FC1);

//     // Download the source image from GPU to CPU for processing
//     Mat src_host;
//     p_src->download(src_host);

//     // Create a temporary GPU matrix for intermediate calculations
//     cv::cuda::GpuMat temp_gpu(rows, cols, CV_64FC1);

//     // Loop over each pixel in the image on the GPU
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             // Define the local neighborhood
//             int rmin = cv::max(0, i - radius);
//             int rmax = cv::min(i + radius, rows - 1);
//             int cmin = cv::max(0, j - radius);
//             int cmax = cv::min(j + radius, cols - 1);
//             double min_val = 255.0;

//             // Loop over the local neighborhood on the GPU
//             for (int x = rmin; x <= rmax; x++) {
//                 for (int y = cmin; y <= cmax; y++) {
//                     cv::Vec3b tmp = src_host.at<cv::Vec3b>(x, y);
//                     double b = static_cast<double>(tmp[0]) / (*p_Alight)[0];
//                     double g = static_cast<double>(tmp[1]) / (*p_Alight)[1];
//                     double r = static_cast<double>(tmp[2]) / (*p_Alight)[2];
//                     double minpixel = cv::min(cv::min(b, g), r);
//                     min_val = cv::min(minpixel, min_val);
//                 }
//             }

//             // Store the result back to the GPU matrix
//             temp_gpu.ptr<double>(i)[j] = 1.0 - omega * min_val;
//         }
//     }

//     // Upload the result back to the GPU matrix
//     temp_gpu.copyTo(*p_tran_gpu);
// }

// void recover_gpu(const cv::cuda::GpuMat *p_src, const cv::cuda::GpuMat *p_tran_gpu, cv::cuda::GpuMat *p_dst, const Vec3d* p_Alight, int rows, int cols, int channels, double t0) {
//     p_dst->create(rows, cols, CV_64FC3);

//     // Download data from GPU to CPU for processing
//     Mat src_host, tran_host;
//     p_src->download(src_host);
//     p_tran_gpu->download(tran_host);

//     // Create temporary CPU matrix for storing the result
//     Mat dst_host(rows, cols, CV_64FC3);

//     // Loop over each pixel in the image on the CPU
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             // Loop over each channel
//             for (int c = 0; c < channels; c++) {
//                 // Calculate the value on the CPU
//                 double val = (static_cast<double>(src_host.at<cv::Vec3b>(i, j)[c]) - (*p_Alight)[c]) /
//                              cv::max(t0, tran_host.ptr<double>(i)[j]) + (*p_Alight)[c];

//                 // Store the result in the temporary CPU matrix
//                 dst_host.at<cv::Vec3d>(i, j)[c] = cv::max(0.0, cv::min(255.0, val));
//             }
//         }
//     }

//     // Upload the result back to the GPU matrix
//     p_dst->upload(dst_host);
// }



// void assign_data(unsigned char *outdata, const cv::cuda::GpuMat *p_dst, int rows, int cols, int channels) {
//     // Download the result from GPU to CPU
//     Mat dst_host;
//     p_dst->download(dst_host);

//     // Copy data from Mat to outdata
//     for (int i = 0; i < rows * cols * channels; i++) {
//         *(outdata + i) = static_cast<unsigned char>(*((double*)(dst_host.data) + i));
//     }
// }
