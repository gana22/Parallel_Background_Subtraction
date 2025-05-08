#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>


using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   Mat diff,fg;
    const int threshold_value = 25;
     string video_path = argv[1];
    const string output_path = "Output_Video/foreground_output_mpi.mp4";
    const string background_img_path = "Output_Video/final_background_mpi.png";

    Mat background;

    int frame_count = 0, width = 0, height = 0;
    vector<Mat> all_gray_frames;
    vector<uchar>bg_vector;

    double fps = 0.0;

    if (rank == 0) {
        VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open video." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cout << "Video opened successfully" << endl;

        width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        frame_count = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
        fps = cap.get(CAP_PROP_FPS);



        for (int i = 0; i < frame_count; ++i) {
            Mat frame, gray(height, width, CV_8UC1);
            cap >> frame;
            if (frame.empty()) break;
            for (int r = 0; r < height; ++r) {
                for (int c = 0; c < width; ++c) {
                    Vec3b bgr = frame.at<Vec3b>(r, c);
                    gray.at<uchar>(r, c) = static_cast<uchar>(
                        0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]);
                }
            }
            all_gray_frames.push_back(gray);
        }
        frame_count = all_gray_frames.size();
        cap.release();
        cout << "Read " << frame_count << " frames" << endl;

         background=Mat(height, width, CV_8UC1, Scalar(0));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                vector<uchar> pixels(frame_count);
                for (int f = 0; f < frame_count; ++f)
                    pixels[f] = all_gray_frames[f].at<uchar>(i, j);
                nth_element(pixels.begin(), pixels.begin() + pixels.size() / 2, pixels.end());
                background.at<uchar>(i, j) = pixels[pixels.size() / 2];

            }
        }

        cout << "The median background Computed" << endl;

    }

    // Broadcast metadata
    MPI_Bcast(&frame_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        bg_vector.assign(background.begin<uchar>(), background.end<uchar>());
    } else {
        bg_vector.resize( width * height);
    }

    MPI_Bcast(bg_vector.data(), height * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        background = Mat(height, width, CV_8UC1);
        memcpy(background.data, bg_vector.data(), height * width);
    }
    int frame_pixels = width * height;
    int frames_per_proc = frame_count / size;
    int rem = frame_count % size;
    int local_count = frames_per_proc + (rank < rem ? 1 : 0);

    // Allocate local buffers
    vector<uchar> input_chunk(local_count * frame_pixels);
    vector<Mat> local_input(local_count);
    vector<uchar>local_output(local_count*frame_pixels);

    // Prepare flattening and scattering
    vector<uchar> flat_input;
    vector<int> sendcounts(size), displs(size);
    if (rank == 0) {
        flat_input.reserve(frame_count * frame_pixels);
        for (const auto& frame : all_gray_frames)
            flat_input.insert(flat_input.end(), frame.begin<uchar>(), frame.end<uchar>());

        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int count = frames_per_proc + (i < rem ? 1 : 0);
            sendcounts[i] = count * frame_pixels;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Scatterv(flat_input.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                 input_chunk.data(), input_chunk.size(), MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    // Compute local foreground masks
    for (int f = 0; f < local_count; ++f) {
        uchar* data = &input_chunk[f * frame_pixels];
        local_input[f]=Mat(height, width, CV_8UC1, data);
    }
    diff=Mat(height, width, CV_8UC1);
    fg=Mat(height, width, CV_8UC1);
    for (int i = 0; i < local_count; ++i) {
        Mat local_frame=local_input[i];
        for (int f = 0; f < height; ++f) {
            for (int r = 0; r < width; ++r) {
                diff.at<uchar>(f, r) = static_cast<uchar>(
                         abs(local_frame.at<uchar>(f, r) - background.at<uchar>(f, r)));
                fg.at<uchar>(f, r) = (diff.at<uchar>(f, r) > threshold_value) ? 255 : 0;
            }

        }
       // local_output_frame[i] = Mat(height, width, CV_8UC1, fg.data);
        memcpy(&local_output[i * frame_pixels], fg.data, frame_pixels);

    }


    // Gather all grayscale frames for background computation
    vector<uchar> all_gray_flat;
    if (rank == 0) all_gray_flat.resize(frame_count * frame_pixels);

    MPI_Gatherv(local_output.data(), local_output.size(), MPI_UNSIGNED_CHAR,
                all_gray_flat.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // Compute background and foreground (master)
    if (rank == 0) {
        vector<Mat> gray_frames;
        for (int i = 0; i < frame_count; ++i)
            gray_frames.emplace_back(height, width, CV_8UC1, &all_gray_flat[i * frame_pixels]);



        // Compute and write foreground video
        VideoWriter writer(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, Size(width, height), false);
        for (int f = 0; f < frame_count; ++f) {
            Mat frame = Mat(height, width, CV_8UC1, gray_frames[f].data);
            writer.write(frame);
        }

        // Save background
        imwrite(background_img_path, background);
        cout << "Saved foreground video and background image." << endl;

        double end_time = MPI_Wtime();
        cout << "\033[36m"; // Set text color to cyan
        cout << "The duration of this process is: " << (end_time - start_time) << " seconds";
        cout << "\033[0m" << endl; // Reset to default color

        cout << "Foreground video saved: " << output_path << endl;
        cout << "Background Photo saved: " << background_img_path << endl;
    }

    MPI_Finalize();
    return 0;
}