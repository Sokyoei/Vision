// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp>  // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>    // Include OpenCV API

int main(int argc, char* argv[]) {
    try {
        // Declare depth colorizer for pretty visualization of depth data
        rs2::colorizer color_map;
        // Declare RealSense pipeline, encapsulating the actual device and sensors
        rs2::pipeline pipe;
        // Start streaming with default recommended configuration
        pipe.start();

        const auto window_name = "Display Image";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

        while (cv::waitKey(1) < 0 && cv::getWindowProperty(window_name, cv::WND_PROP_AUTOSIZE) >= 0) {
            rs2::frameset data = pipe.wait_for_frames();  // Wait for next set of frames from the camera
            rs2::frame depth = data.get_depth_frame().apply_filter(color_map);

            // Query frame size (width and height)
            const int w = depth.as<rs2::video_frame>().get_width();
            const int h = depth.as<rs2::video_frame>().get_height();

            // Create OpenCV matrix of size (w,h) from the colorized depth data
            cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

            // Update the window with new data
            cv::imshow(window_name, image);
        }

        return EXIT_SUCCESS;
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    "
                  << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
