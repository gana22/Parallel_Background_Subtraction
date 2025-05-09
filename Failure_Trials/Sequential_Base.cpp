#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>  // For abs()

using namespace std;
using namespace cv;
using namespace chrono;

int main() {
    cout << "Program started" << endl;

    // Constants
    double alpha = 0.7;                // learning rate
    int threshold_value = 25;           // threshold to ignore small noise

    // File paths
        string video_path = "../Input_Video/input_vtest.avi";
        string foreground_output_path = "../Failure_Trials/Output_Video/foreground_output_Base.mp4";
        string background_output_path = "../Failure_Trials/Output_Video/final_background_Base.png";

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
    bool background_initialized = false;

    cout << "Starting processing..." << endl;
    auto start = high_resolution_clock::now();


    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert frame to grayscale
        Mat gray(frame.rows, frame.cols, CV_8UC1);

        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                Vec3b bgr = frame.at<Vec3b>(i, j);
                uchar blue = bgr[0];
                uchar green = bgr[1];
                uchar red = bgr[2];
                gray.at<uchar>(i, j) = static_cast<uchar>(0.114 * blue + 0.587 * green + 0.299*red);
            }
        }

        // Initialize background (first frame)
        if (!background_initialized) {
            gray.convertTo(background, CV_32F);  // Convert to float for better accumulation precision
            background_initialized = true;
            continue;
        }

        // Update background model manually (without accumulateWeighted)
        for (int i = 0; i < gray.rows; ++i) {
            for (int j = 0; j < gray.cols; ++j) {
                background.at<float>(i, j) = alpha * gray.at<uchar>(i, j) + (1 - alpha) * background.at<float>(i, j);
            }
        }

        // Convert background to uint8 for saving
        Mat background_u8;
        background.convertTo(background_u8, CV_8U);

        // Compute the absolute difference manually (without absdiff)
        diff.create(gray.size(), gray.type());
        for (int i = 0; i < gray.rows; ++i) {
            for (int j = 0; j < gray.cols; ++j) {
                diff.at<uchar>(i, j) = static_cast<uchar>(abs(gray.at<uchar>(i, j) - background_u8.at<uchar>(i, j)));
            }
        }

        // Threshold manually (without threshold function)
        fg_mask.create(diff.size(), diff.type());
        for (int i = 0; i < diff.rows; ++i) {
            for (int j = 0; j < diff.cols; ++j) {
                fg_mask.at<uchar>(i, j) = (diff.at<uchar>(i, j) > threshold_value) ? 255 : 0;
            }
        }

        // Save the foreground mask (converted to BGR for saving)
        foreground_writer.write(fg_mask);
    }

    auto end = high_resolution_clock::now();
    duration<double> duration = end - start;
    cout << "\033[36m"; // Set text color to cyan
    cout << "The duration of this process is: " << duration.count() << " seconds";
    cout << "\033[0m" << endl; // Reset to default color

    // After processing: Save final background image
    Mat final_background;
    background.convertTo(final_background, CV_8U);
    imwrite(background_output_path, final_background);
    cout << "Final background image saved!" << endl;

    // Release resources
    cap.release();
    foreground_writer.release();

    cout << "Foreground video saved: " << foreground_output_path << endl;
    cout << "Background Photo saved: " << background_output_path << endl;

    cout << "Program finished successfully" << endl;

    return 0;
}
