#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

Mat image, regionOfInterest;
vector<Point> roi;
void CallBackFunction(int event, int x, int y, int flags, void* point)
{

	if (event == EVENT_LBUTTONDOWN)
	{
		Point p;
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		p.x = x;
		p.y = y;
		roi.push_back(p);
	}

}

int main(int argc, char** argv)
{
	Point p;
	string imagename;
	cout << "Please enter the name of the file: ";
	getline(cin, imagename);
	cout << "The value you entered is " << imagename;
	image = imread(imagename, CV_LOAD_IMAGE_COLOR); // Read the file

	while(!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		cout << "Please enter the name of the file: ";
		getline(cin, imagename);
		cout << "The value you entered is " << imagename;
		image = imread(imagename, CV_LOAD_IMAGE_COLOR); // Read the file
	}

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	setMouseCallback("Display window", CallBackFunction, NULL);
	imshow("Display window", image); // Show our image inside it.
	while (cv::waitKey(1) != 27);
	cout << roi[0].x << " " << roi[0].y << " " << roi[1].x << " " << roi[1].y;
	regionOfInterest = image(Rect(roi[0].x, roi[0].y, roi[1].x - roi[0].x, roi[1].y - roi[0].y));
	destroyWindow("Display Window");
	namedWindow("Region of Interest", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Region of Interest", regionOfInterest); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}