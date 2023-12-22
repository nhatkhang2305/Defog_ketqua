#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/cudaimgproc.hpp>
#include </usr/local/include/opencv4/opencv2/core/cuda.hpp>

int main() {
    // Read an image from file
    cv::Mat input_image = cv::imread("22.jpg");

    if (input_image.empty()) {
        std::cerr << "Could not open or find the image." << std::endl;
        return -1;
    }

    // Create a GPU Mat for CUDA operations
    cv::cuda::GpuMat gpu_input_image;
    gpu_input_image.upload(input_image);

    // Convert color using CUDA
    cv::cuda::GpuMat gpu_output_image;
    cv::cuda::cvtColor(gpu_input_image, gpu_output_image, cv::COLOR_BGR2GRAY);

    // Download the result back to the CPU
    cv::Mat output_image;
    gpu_output_image.download(output_image);

    // Display the original and converted images
    cv::imshow("Original Image", input_image);
    cv::imshow("Converted Image", output_image);
    cv::waitKey(0);

    return 0;
}
