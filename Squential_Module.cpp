#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>  // For abs()

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char* argv[]) {
    cout << "Program started" << endl;

    // Constants
    int threshold_value = 25;           // threshold to ignore small noise

    // File paths
    string video_path = argv[1];
    string foreground_output_path = "Output_Video/foreground_output_Sequential.mp4";
    string background_output_path = "Output_Video/final_background_Sequential.png";

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
                                false);  // true: BGR

    if (!foreground_writer.isOpened()) {
        cerr << "Error: Could not open output video files" << endl;
        return -1;
    }

    Mat frame, gray, background, diff, fg_mask;
    vector<Mat> gray_frames;

    auto start = high_resolution_clock::now();


    // Read frames and store grayscale versions
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert to grayscale
        Mat gray(frame.rows, frame.cols, CV_8UC1);
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                Vec3b bgr = frame.at<Vec3b>(i, j);
                uchar blue = bgr[0];
                uchar green = bgr[1];
                uchar red = bgr[2];
                gray.at<uchar>(i, j) = static_cast<uchar>(0.114 * blue + 0.587 * green + 0.299 * red);
            }
        }
        gray_frames.push_back(gray);
    }

    cap.release();  // Release video capture after reading all frames
    cout << "Read " << gray_frames.size() << " frames" << endl;

    // Compute the median background
    background=Mat(gray_frames[0].rows, gray_frames[0].cols, CV_8UC1);
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
        Mat frame = gray_frames[idx];

        // Compute the absolute difference between current frame and background
        Mat diff(frame.size(), frame.type());
        fg_mask.create(diff.size(), diff.type());

        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                diff.at<uchar>(i, j) = static_cast<uchar>(
                    abs(frame.at<uchar>(i, j) - background.at<uchar>(i, j)));
            }
        }

        // Threshold to create foreground mask
        fg_mask.create(diff.size(), diff.type());
        for (int i = 0; i < diff.rows; ++i) {
            for (int j = 0; j < diff.cols; ++j) {
                fg_mask.at<uchar>(i, j) = (diff.at<uchar>(i, j) > threshold_value) ? 255 : 0;
            }
        }

        // Write the foreground mask to output video
        foreground_writer.write(fg_mask);
    }
    auto end = high_resolution_clock::now();
    duration<double> duration = end - start;
    cout << "\033[36m"; // Set text color to cyan
    cout << "The duration of this process is: " << duration.count() << " seconds";
    cout << "\033[0m" << endl; // Reset to default color

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