#include "EllipseDetector.h"
#include "util.h"
#include "EdgeDetector.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    auto start = clock();
    std::string input = "a.jpg";
    bool save = false;
    if (argc > 1)
    {
        input = argv[1];

        if (argc > 2 && (std::string(argv[2]) == "-s" || std::string(argv[2]) == "--save"))
        {
            save = true;
        }
    }
    cv::Mat img_raw = cv::imread(input);
    cv::Mat img;
    if (img_raw.cols > 1000 || img_raw.rows > 1000)
    {
        cv::resize(img_raw, img, cv::Size(), 0.5, 0.5);
    }
    else
    {
        img = img_raw;
    }
    std::cout << "input: " << input << std::endl;
    std::cout << "img_raw: " << img_raw.cols << "x" << img_raw.rows << std::endl;
    std::cout << "img: " << img.cols << "x" << img.rows << std::endl;

    EllipseDetector ellipse_Detector;
    std::vector<Ellipse> ellipses = ellipse_Detector.DetectImage(img);
    cv::Mat3b img0 = ellipse_Detector.image();
    draw_ellipses_all(ellipses, img0);
    printf("Time: %.2f ms\n", (clock() - start) * 1000.0 / CLOCKS_PER_SEC);

    if (save)
    {
        std::filesystem::path p(input);
        // std::filesystem::path output_path = p.parent_path() / (p.stem().string() + "_results.jpg");
        std::filesystem::path output_path = (p.stem().string() + "_results.jpg");
        std::cout << "output_path: " << output_path << std::endl;
        cv::imwrite(output_path, img0);
    }

    cv::imshow("Result", img0);
    cv::waitKey();

    return 0;
}