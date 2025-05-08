#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <omp.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    cout << "Program started" << endl;

    // Constants
    int threshold_value = 25;           // threshold to ignore small noise

    // Get number of available threads
    int num_threads = omp_get_max_threads(); // reads from OMP_NUM_THREADS
    cout << "Running with " << num_threads << " threads.\n";

    // // File paths
    string video_path = argv[1];
    string foreground_output_path = "Output_Video/foreground_output_OpenMP.mp4";
    string background_output_path = "Output_Video/final_background_OpenMP.png";

    // Open input video
    VideoCapture cap;
    if (!cap.open(video_path)) {
        cout << "Error: Could not open video file" << endl;
        return -1;
    }
    cout << "Video opened successfully" << endl;

    // Get video properties
    double fps = cap.get(CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // Initialize writers
    VideoWriter foreground_writer(foreground_output_path,
                                VideoWriter::fourcc('m', 'p', '4', 'v'),
                                fps,
                                Size(width, height),
                                false);  // false: grayscale

    if (!foreground_writer.isOpened()) {
        cerr << "Error: Could not open output video files" << endl;
        return -1;
    }

    vector<Mat> gray_frames;
    Mat frame, gray, background, diff, fg_mask;

    double start_time = omp_get_wtime();

    // Read frames and store grayscale versions
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert to grayscale
        Mat gray(frame.rows, frame.cols, CV_8UC1);

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                Vec3b bgr = frame.at<Vec3b>(i, j);
                uchar blue = bgr[0];
                uchar green = bgr[1];
                uchar red = bgr[2];
                gray.at<uchar>(i, j) = static_cast<uchar>(0.114 * blue + 0.587 * green + 0.299 * red);
            }
        }

        gray_frames.push_back(gray.clone()); // clone to ensure we have a separate copy
    }

    cap.release();  // Release video capture after reading all frames
    cout << "Read " << gray_frames.size() << " frames" << endl;

    if (gray_frames.empty()) {
        cerr << "Error: No frames were read from the video" << endl;
        return -1;
    }

    // Compute the median background
    background=Mat(gray_frames[0].rows, gray_frames[0].cols, CV_8UC1);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < background.rows; ++i) {
        for (int j = 0; j < background.cols; ++j) {
            vector<uchar> pixel_values(gray_frames.size());
            for (size_t f = 0; f < gray_frames.size(); f++) {
                pixel_values[f] = gray_frames[f].at<uchar>(i, j);
            }
            nth_element(pixel_values.begin(), pixel_values.begin() + pixel_values.size() / 2, pixel_values.end());
            background.at<uchar>(i, j) = pixel_values[pixel_values.size() / 2];
        }
    }

    cout << "The median background Computed" << endl;

    // Process each frame and compute foreground
    for (size_t idx = 0; idx < gray_frames.size(); ++idx) {
        const Mat& current_frame = gray_frames[idx];

        // Compute the absolute difference between current frame and background
        Mat diff(current_frame.size(), current_frame.type());
        fg_mask.create(diff.size(), diff.type());

        // Compute absolute difference and threshold
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < background.rows; ++i) {
            for (int j = 0; j < background.cols; ++j) {
                // Compute absolute difference
                diff.at<uchar>(i, j) = static_cast<uchar>(abs(current_frame.at<uchar>(i, j) - background.at<uchar>(i, j)));
            }
        }

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < background.rows; ++i) {
            for (int j = 0; j < background.cols; ++j) {
                // Apply threshold in the same loop
                fg_mask.at<uchar>(i, j) = (diff.at<uchar>(i, j) > threshold_value) ? 255 : 0;
            }
        }

        // Write the foreground mask to output video
        foreground_writer.write(fg_mask);
    }

    double end_time = omp_get_wtime();
    double duration_time = end_time - start_time;
    cout << "The duration of this process is: " << duration_time << " seconds" << endl;

    // Save the final background image
    imwrite(background_output_path, background);
    cout << "Final background image saved!" << endl;

    // Release resources
    foreground_writer.release();

    cout << "Foreground video saved: " << foreground_output_path << endl;
    cout << "Background Photo saved: " << background_output_path << endl;

    cout << "Program finished successfully" << endl;

    return 0;
}