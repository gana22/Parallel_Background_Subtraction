 #include <iostream>
 #include <opencv2/opencv.hpp>
 #include <vector>
 #include <chrono>  // For timing the operation

 using namespace std;
 using namespace cv;
 using namespace chrono;

 int main() {
     cout << "Program started" << endl;

     // Constants
     double threshold = 30;

     // File paths
     string video_path = "../Input_Video/input_vtest.avi";
     string foreground_output_path = "../Failure_Trials/Output_Video/foreground_output_Mean.mp4";
     string background_output_path = "../Failure_Trials/Output_Video/final_background_Mean.png";

     // Open the video file
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
                                 false);  // grayscale output

     if (!foreground_writer.isOpened()) {
         cerr << "Error: Could not open output video files" << endl;
         return -1;
     }

     Mat frame, gray, background, diff, fg_mask;
     vector<Mat> gray_frames;

     auto start = high_resolution_clock::now();

     // Read all frames to create background model
     while (true) {
         cap >> frame;
         if (frame.empty()) break;

         // Convert to grayscale (manually)
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
         gray_frames.push_back(gray.clone());
     }

     // Calculate the background model (mean of M frames)
     cout << "Calculating background model..." << endl;
     background = Mat::zeros(height, width, CV_8UC1);

     // Sum up all frames
     for (int i = 0; i < gray_frames.size(); i++) {
         add(background, gray_frames[i], background, noArray(), CV_32F);
     }

     // Divide by the number of frames to get the mean
     background = background / gray_frames.size();
     background.convertTo(background, CV_8UC1);


     // Reset video to beginning
     cap.set(CAP_PROP_POS_FRAMES, 0);

     // Process all frames
     int frame_count = 0;
     Mat foreground_mask, foreground, output_frame;

     while (true) {
         cap >> frame;
         if (frame.empty()) {
             break;
         }

         frame_count++;

         // Convert frame to grayscale (manually)
         Mat gray_frame(frame.rows, frame.cols, CV_8UC1);
         for (int i = 0; i < frame.rows; ++i) {
             for (int j = 0; j < frame.cols; ++j) {
                 Vec3b bgr = frame.at<Vec3b>(i, j);
                 uchar blue = bgr[0];
                 uchar green = bgr[1];
                 uchar red = bgr[2];
                 gray_frame.at<uchar>(i, j) = static_cast<uchar>(0.114 * blue + 0.587 * green + 0.299 * red);
             }
         }

         // Calculate foreground mask by subtracting background and thresholding
         foreground_mask = Mat::zeros(gray_frame.size(), CV_8UC1);

         for (int y = 0; y < gray_frame.rows; y++) {
             for (int x = 0; x < gray_frame.cols; x++) {
                 int diff = abs(background.at<uchar>(y, x) - gray_frame.at<uchar>(y, x));
                 if (diff > threshold) {
                     foreground_mask.at<uchar>(y, x) = 255;
                 }
             }
         }

         // Get foreground by applying the mask to the original frame
         output_frame = Mat::zeros(frame.size(), frame.type());
         frame.copyTo(output_frame, foreground_mask);


         // Also write to the foreground output
         foreground_writer.write(foreground_mask);

         // Exit on ESC key
         char c = (char)waitKey(25);
         if (c == 27) {
             break;
         }
     }
     // Release resources
     cap.release();
     foreground_writer.release();

     auto end = high_resolution_clock::now();
     duration<double> duration = end - start;
     cout << "\033[36m"; // Set text color to cyan
     cout << "The duration of this process is: " << duration.count() << " seconds";
     cout << "\033[0m" << endl; // Reset to default color

     // Save the final background image
     imwrite(background_output_path, background);
     cout << "Final background image saved!" << endl;

     cout << "Foreground video saved: " << foreground_output_path << endl;
     cout << "Background Photo saved: " << background_output_path << endl;

     cout << "Program finished successfully" << endl;

     return 0;
 }
