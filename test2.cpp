#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include </usr/local/include/opencv4/opencv2/imgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaimgproc.hpp>
#include </usr/local/include/opencv4/opencv2/cudaarithm.hpp>
#include <iostream>
#include </usr/local/include/opencv4/opencv2/core/cuda.hpp>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>

cv::cuda::GpuMat padArrayReplicate(const cv::cuda::GpuMat& input, int padRows, int padCols) 
{
    cv::cuda::GpuMat output;
    cv::cuda::copyMakeBorder(input, output, padRows, padRows, padCols, padCols, cv::BORDER_REPLICATE);
    return output;
}
int main() {
        // Read an image from file
        cv::Mat input_image = cv::imread("suong1.jpg");
        int rows = input_image.rows;
        int cols = input_image.cols;
        int P_S = 15;
        int p = (P_S - 1) / 2;
        // Create a GPU Mat for CUDA operations
        cv::cuda::GpuMat gpu_input_image;
        gpu_input_image.upload(input_image);
        cv::cuda::GpuMat gpu_output_image;
        cv::cuda::cvtColor(gpu_input_image, gpu_output_image, cv::COLOR_BGR2GRAY);
        cv::cuda::GpuMat gpu_input_image_padded = padArrayReplicate(gpu_output_image, p, p);
        cv::cuda::GpuMat J_DARK(rows, cols, CV_32F, cv::Scalar(0));
        cv::cuda::GpuMat local_patch_gpu(P_S, P_S, 1);  // Assuming grayscale image
        cv::cuda::GpuMat minimum_value_gpu(1, 1, CV_32F);  // Store single minimum value


        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
            // Extract patch directly on GPU
            cv::Range range_row(i, i + P_S);
            cv::Range range_col(j, j + P_S);
            gpu_input_image_padded(range_row, range_col).copyTo(local_patch_gpu);

            // Find minimum value in the patch on GPU
            
            cv::cuda::GpuMat minimum_value_mat;
            cv::cuda::minMaxLoc(local_patch_gpu, NULL, NULL, NULL, NULL, minimum_value_mat);

            //cv::cuda::GpuMat minimum_value_gpu = minimum_value_mat.clone();
            // Store minimum value in J_DARK_gpu
            float* J_DARK_ptr = J_DARK.ptr<float>(i);
            //float* minimum_value_ptr = minimum_value_gpu.ptr<float>(0);
            J_DARK_ptr[j] = minimum_value_mat[0];
            // float* J_DARK_ptr = J_DARK.ptr<float>(i);
            // J_DARK_ptr[j] = minimum_value_gpu.ptr<float>(0)[0];
    //      J_DARK.at<float>(i, j) = minimum_value_gpu.at<float>(0, 0);
            }
        }


    // // Convert color using CUDA
    // cv::cuda::GpuMat gpu_output_image;
    // cv::cuda::cvtColor(gpu_input_image, gpu_output_image, cv::COLOR_BGR2GRAY);

    // Download the result back to the CPU
    //cv::Mat output_image;
    //gpu_output_image.download(output_image);
    //cv::imwrite("output.jpg", output_image);
    // Display the original and converted images
    //cv::imshow("Original Image", input_image);
    //cv::imshow("Converted Image", output_image);
    //cv::waitKey(0);

    return 0;
}