#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/imgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaimgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaarithm.hpp>
#include <iostream>
#include <chrono>
#include </usr/local/include/opencv4/opencv2/core/cuda.hpp>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>


__global__ void darkChannelKernel(const cv::cuda::PtrStepSz<uchar> src, cv::cuda::PtrStepSz<uchar> dst, int padSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < src.rows && col < src.cols) {
        // Xác định phạm vi patch xung quanh pixel hiện tại
        int rmin = std::max(0, row - padSize);
        int rmax = std::min(row + padSize, src.rows - 1);
        int cmin = std::max(0, col - padSize);
        int cmax = std::min(col + padSize, src.cols - 1);

        // Tìm giá trị tối thiểu trong patch
        uchar minValue = 255;
        for (int i = rmin; i <= rmax; i++) {
            for (int j = cmin; j <= cmax; j++) {
                minValue = std::min(minValue, src(i, j));
            }
        }

        // Lưu giá trị tối thiểu vào hình ảnh kết quả
        dst(row, col) = minValue;
    }
}

cv::Mat darkChannelGPU(const cv::Mat& input_image, int padSize) {
    cv::cuda::GpuMat src_gpu, dst_gpu;
    src_gpu.upload(input_image);

    dst_gpu.create(src_gpu.size(), CV_8UC1);

    dim3 block_size(32, 32);
    dim3 grid_size((src_gpu.cols + block_size.x - 1) / block_size.x, (src_gpu.rows + block_size.y - 1) / block_size.y);

    darkChannelKernel<<<grid_size, block_size>>>(src_gpu, dst_gpu, padSize);
    cudaDeviceSynchronize();

    cv::Mat dst;
    dst_gpu.download(dst);

    return dst;
}

int main() {
    // Khởi tạo hình ảnh thực tế
    const char* img_path = "suong1.jpg";
    cv::Mat in_img = cv::imread(img_path);

    if (in_img.empty()) {
        std::cerr << "Could not open or find the image." << std::endl;
        return -1;
    }

    // Xác định pad size
    int padSize = 7;

    // Thực hiện trích xuất kênh tối trên GPU và đo thời gian thực thi
    auto start_time = std::chrono::high_resolution_clock::now();
    cv::Mat dark_channel_result_gpu = darkChannelGPU(in_img, padSize);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Thời gian thực thi trên GPU: " << elapsed_time.count() << " giây" << std::endl;

    // Lưu kết quả
    cv::imwrite("dark_channel_gpu.jpg", dark_channel_result_gpu);

    return 0;
}


