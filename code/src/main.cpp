#include "EllipseDetector.h"
#include "util.h"
#include "EdgeDetector.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <numeric>

namespace
{
double getMean(const std::vector<double> &v)
{
  if (v.empty()) {
    return -1;
  }

  size_t size = v.size();

  double sum = std::accumulate(v.begin(), v.end(), 0.0);

  return sum / (static_cast<double>(size));
}

double getMedian(const std::vector<double> &v)
{
  if (v.empty()) {
    return -1;
  }

  std::vector<double> v_copy = v;
  size_t size = v_copy.size();

  size_t n = size / 2;
  std::nth_element(v_copy.begin(), v_copy.begin() + n, v_copy.end());
  double val_n = v_copy[n];

  if ((size % 2) == 1) {
    return val_n;
  }
  else {
    std::nth_element(v_copy.begin(), v_copy.begin() + (n - 1), v_copy.end());
    return 0.5 * (val_n + v_copy[n - 1]);
  }
}
}

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // std::cout << "cv::getBuildInformation()\n" << cv::getBuildInformation() << std::endl;

    auto start = clock();
    std::string input = "a.jpg";
    bool save = false;
    std::string output_save = "";
    int roi0 = 0, roi1 = 0, roi2 = 0, roi3 = 0;
    if (argc > 1)
    {
        input = argv[1];

        if (argc > 2 && (std::string(argv[2]) == "-s" || std::string(argv[2]) == "--save"))
        {
            save = true;
            output_save = std::string(argv[3]);
        }

        if (argc > 2 && (std::string(argv[4]) == "--roi"))
        {
            roi0 = atoi(argv[5]);
            roi1 = atoi(argv[6]);
            roi2 = atoi(argv[7]);
            roi3 = atoi(argv[8]);
        }
    }
    std::cout << "roi0=" << roi0 << " ; roi1=" << roi1 << " ; roi2=" << roi2 << " ; roi3=" << roi3 << std::endl;

    cv::VideoCapture capture(input);
    if (!capture.isOpened())
    {
        std::cout << "Cannot open: " << input << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat img_raw;
    int iter = 0;
    std::vector<double> times_vec;
    for (;;)
    {
        // cv::Mat img_raw = cv::imread(input);
        capture >> img_raw;
        if (img_raw.empty())
        {
            std::cerr << "Cannot retrieve image" << std::endl;
            break;
        }

        double start = cv::getTickCount();
        cv::Mat img;
        if (false /*img_raw.cols > 1000 || img_raw.rows > 1000*/)
        {
            cv::resize(img_raw, img, cv::Size(), 0.5, 0.5);
        }
        else
        {
            img = img_raw;
        }
        // 241 821 129 124
        // cropped_frame_original = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        // Mat crop = img(cv::Range(821, 821+124), cv::Range(241, 241+129));

        int scale = 2;
        // Mat crop = img(cv::Range(scale*817, scale*(817+127)), cv::Range(scale*239, scale*(239+142)));
        Mat crop = img(cv::Range(scale*roi1, scale*(roi1+roi3)), cv::Range(scale*roi0, scale*(roi0+roi2)));

        img = crop.clone();
        // std::cout << "input: " << input << std::endl;
        // std::cout << "img_raw: " << img_raw.cols << "x" << img_raw.rows << std::endl;
        // std::cout << "img: " << img.cols << "x" << img.rows << std::endl;
        // std::cout << "crop: " << crop.cols << "x" << crop.rows << std::endl;

        EllipseDetector ellipse_Detector;
        std::vector<Ellipse> ellipses = ellipse_Detector.DetectImage(img);
        cv::Mat3b img0 = ellipse_Detector.image();
        draw_ellipses_all(ellipses, img0);
        // printf("Time: %.2f ms\n", (clock() - start) * 1000.0 / CLOCKS_PER_SEC);

        double end = cv::getTickCount();
        double elapsed_time = (end - start) / cv::getTickFrequency();
        times_vec.push_back(elapsed_time);

        if (save)
        {
            // std::filesystem::path p(input);
            // // std::filesystem::path output_path = p.parent_path() / (p.stem().string() + "_results.jpg");
            // std::filesystem::path output_path = (p.stem().string() + "_results.jpg");
            // std::cout << "output_path: " << output_path << std::endl;
            // cv::imwrite(output_path, img0);

            char buffer[256];
            sprintf(buffer, output_save.c_str(), iter);
            std::string output_filename = buffer;
            // std::cout << "output_save: " << output_save << std::endl;
            // std::cout << "output_filename: " << output_filename << std::endl;
            cv::imwrite(output_filename, img0);
        }

        iter++;

        cv::imshow("Result", img0);
        char key = cv::waitKey(30);
        if (key == 27)
        {
            break;
        }
    }

    std::cout << "Computation time, mean: " << getMean(times_vec) << " s ; median: " << getMedian(times_vec) << " s" << std::endl;

    return 0;
}