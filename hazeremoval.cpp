#include "hazeremoval.h"
#include </usr/local/include/opencv4/opencv2/core/cuda.hpp>
#include <iostream>
#include <mutex>
#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/imgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaimgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaarithm.hpp>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
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
    cuda::GpuMat src_gpu;
    src_gpu.upload(Mat(rows, cols, CV_8UC3, (void*)indata));

    int radius = 7;
    double omega = 0.95;
    double t0 = 0.1;
    vector<Pixel> tmp_vec;
    // cv::cuda::GpuMat tmp_vec_gpu(tmp_vec);
    cv::cuda::GpuMat tmp_vec_gpu(rows * cols, 1, CV_8UC3);
    //Vec3d Alight_host;

    // Allocate GPU memory for other matrices
    cuda::GpuMat dst_gpu(rows, cols, CV_64FC3);
    cuda::GpuMat tran_gpu(rows, cols, CV_64FC1);

 	cuda::GpuMat dark_channel_gpu;

    // // Perform operations on GPU
    get_dark_channel_gpu(src_gpu, tmp_vec, rows, cols, channels, radius);
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

// void get_dark_channel_gpu(const cv::cuda::GpuMat* src_gpu, int rows, int cols, int channels, int radius) {
//     // Convert the input image to grayscale on the GPU
//     //cv::cuda::GpuMat src_gray;
//     //cv::cuda::cvtColor(*src_gpu, src_gray, cv::COLOR_BGR2GRAY);
//     //cv::cuda::GpuMat rmin_gpu, rmax_gpu, cmin_gpu, cmax_gpu;
//     cv::cuda::GpuMat tmp_vec_gpu(rows * cols, 1, CV_8UC3);
//     //cv::cuda::GpuMat::Pixel pixel;
//     // Allocate GPU memory for dark channel
//     //dark_channel_gpu.create(rows, cols, CV_8UC1);

//     // Compute dark channel on GPU
//     //cv::cuda::GpuMat min_channel_gpu(rows, cols, CV_8UC1);
//     //cv::cuda::GpuMat temp_channel_gpu(rows, cols, CV_8UC1);
//     //Loop over the image pixels
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             // Define the local neighborhood
//             int rmin = max(0, i - radius);
//             int rmax = min(i + radius, rows - 1);
//             int cmin = max(0, j - radius);
//             int cmax = min(j + radius, cols - 1);
//             double min_val = 255;
//             //cv::cuda::max(0, i - radius, rmin_gpu);
//             //cv::cuda::min(i + radius, rows - 1, rmax_gpu);
//             //cv::cuda::max(0, j - radius, cmin_gpu);
//             //cv::cuda::min(j + radius, cols - 1, cmax_gpu);
//             //int rmin = rmin_gpu.data[0];
//             //int rmax = rmax_gpu.data[0];
//             //int cmin = cmin_gpu.data[0];
//             //int cmax = cmax_gpu.data[0];
//             // for (int x = rmin; x <= rmax; x++) {
// 			// 	for (int y = cmin; y <= cmax; y++) {
// 			// 		//cv::Vec3b tmp = p_src->ptr<cv::Vec3b>(x)[y];
//             //         //cv::Vec3b tmp = src_gpu->ptr<cv::Vec3b>(x)[y];
//             //         cv::cuda::Vec3b tmp = src_gpu.at<cv::cuda::Vec3b>(x, y);
//             //         //cv::Vec3b tmp = src_gpu->ptr<cv::Vec3b>(x, y);
// 			// 		uchar b = tmp[0];
// 			// 		uchar g = tmp[1];
// 			// 		uchar r = tmp[2];
// 			// 		uchar minpixel = b > g ? ((g>r) ? r : g) : ((b > r) ? r : b); //tim min rgb
// 			// 		min_val = min((double)minpixel, min_val); //so sanh mini vs min_val cap nhat min_val
// 			// 	}
// 			// }
//             // uchar data[3];
//             // data[0] = (uchar)min_val;
//             // data[1] = (uchar)min_val;
//             // data[2] = (uchar)min_val;
//              for (int x = rmin; x <= rmax; x++) {
//                 // Access the entire row on the GPU
//                 const uchar3* row = src_gpu->ptr<uchar3>(x);

//                 for (int y = cmin; y <= cmax; y++) {
//                     // Access the pixel using the ptr method
//                     uchar3 tmp = row[y];
//                     uchar b = tmp.x;
//                     uchar g = tmp.y;
//                     uchar r = tmp.z;

//                     // uchar minpixel = b > g ? ((g > r) ? r : g) : ((b > r) ? r : b);
//                 //     min_val = std::min(static_cast<double>(minpixel), min_val);
//                 }
//             }
//             // cv::Vec3b* pixel = tmp_vec_gpu.ptr<cv::Vec3b>(i, j);
//             // *pixel = cv::Vec3b(data[0], data[1], data[2]);

//             //tmp_vec_gpu.push_back(cv::cuda::GpuMat::at<cv::Vec3b>(i, j, data));
//             //tmp_vec_gpu.push_back(cv::cuda::GpuMat::Pixel(i, j, (uchar)min_val));
//             // // Extract the corresponding region on the GPU
//             // cv::Rect roi(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1);
//             // cv::cuda::GpuMat region_gpu(src_gray, roi);

//             // // Compute the minimum value in the region on the GPU
//             // double min_val;
//             // double max_val;
//             // cv::Point min_loc;
//             // cv::Point max_loc;
//             // cv::minMaxLoc(region_gpu, &min_val, &max_val, &min_loc, &max_loc);

//             // // Set the result in the dark channel matrix
//             // dark_channel_gpu.ptr<uchar>(i)[j] = static_cast<uchar>(min_val);
//         }
//     }
// //cout <<"hi" ;
// }
void get_dark_channel_gpu(const cv::cuda::GpuMat& src_gpu, std::vector<Pixel>& tmp_vec, int rows, int cols, int channels, int radius) {
    tmp_vec.clear();

    cv::cuda::GpuMat tmp_gpu;
    cv::cuda::GpuMat min_vals_gpu(rows, cols, CV_8UC1);

    // Iterate over pixels
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int rmin = std::max(0, i - radius);
            int rmax = std::min(i + radius, rows - 1);
            int cmin = std::max(0, j - radius);
            int cmax = std::min(j + radius, cols - 1);

            // Extract the region of interest (ROI) on GPU
            cv::cuda::GpuMat roi = src_gpu(cv::Rect(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1));

            // Calculate the minimum value in the ROI on GPU
            cv::Mat min_vals_cpu;
            roi.download(min_vals_cpu);

            // Calculate the minimum value on CPU
            uchar min_val = *std::min_element(min_vals_cpu.ptr<uchar>(), min_vals_cpu.ptr<uchar>() + min_vals_cpu.cols);

            // Store the result
            tmp_vec.push_back(Pixel(i, j, min_val));

            // // Upload the result to CPU
            // min_vals_gpu.download(tmp_gpu);

            // // Find the minimum value across the channels on CPU
            // uchar min_val = *std::min_element(tmp_gpu.ptr<uchar>(), tmp_gpu.ptr<uchar>() + tmp_gpu.cols);
            
            // // Store the result
            // tmp_vec.push_back(Pixel(i, j, min_val));
        }
    }
}
// void get_dark_channel_gpu(const cv::cuda::GpuMat& src_gpu, std::vector<Pixel>& tmp_vec, int rows, int cols, int channels, int radius) {
//     tmp_vec.clear();

//     cv::cuda::GpuMat tmp_gpu;
//     cv::cuda::GpuMat min_vals_gpu(rows, cols, CV_8UC1);

//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             int rmin = std::max(0, i - radius);
//             int rmax = std::min(i + radius, rows - 1);
//             int cmin = std::max(0, j - radius);
//             int cmax = std::min(j + radius, cols - 1);

//             uchar min_val = 255;

//             for (int x = rmin; x <= rmax; x++) {
//                 const uchar3* row = src_gpu.ptr<uchar3>(x);

//                 for (int y = cmin; y <= cmax; y++) {
//                     uchar3 tmp = row[y];
//                     uchar b = tmp.x;
//                     uchar g = tmp.y;
//                     uchar r = tmp.z;

//                     uchar minpixel = b > g ? ((g > r) ? r : g) : ((b > r) ? r : b);
//                     std::lock_guard<std::mutex> lock(min_val);
//                     min_val = std::min(minpixel, min_val);
//                 }
//             }

//             tmp_vec.push_back(Pixel(i, j, min_val));
//         }
//     }
// }
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
