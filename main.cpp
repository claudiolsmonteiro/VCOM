// opencv_p01 - 2011/09/18 - JAS
// Read and display an image
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
int main()
{
	//read an image
	Mat image = cv::imread("img.jpg");
	//create image window named "My image"
	namedWindow("My image"); //not necessary ... imshow() is enough...!
							 //show the image on window
	imshow("My image", image);
	//wait key for 5000 ms
	waitKey(5000); // waitKey(0); // waits for key press
	return 0;
}