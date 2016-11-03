#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
string windowname = "Region of Interest";
Mat image, imageEmpty, regionOfInterest, regionOfInterestEmpty, greyroi, greyroiempty, detected_edges;
vector<Point> roi, rect;
int size[2];
struct contour_sorter // 'less' for contours
{
	bool operator ()(const vector<Point>& a, const vector<Point> & b)
	{
		Rect ra(boundingRect(a));
		Rect rb(boundingRect(b));
		// scale factor for y should be larger than img.width
		return ((ra.x) < (rb.x));
	}
};


void matchTemplate(Point start, Point finish) {
	cv::Mat input = regionOfInterestEmpty;
	//cv::Mat input = regionOfInterest;
	cv::Mat gray;
	cv::cvtColor(input, gray, CV_BGR2GRAY);

	cv::Mat templ = regionOfInterest(Rect(start, finish));

	cv::Mat img = input;
	cv::Mat result;
	/// Create the result matrix
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	int match_method = TM_SQDIFF_NORMED;

	/// Do the Matching and Normalize
	matchTemplate(img, templ, result, match_method);
	//normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
	cv::Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
		cout << "MIN VAL " << minVal << endl;
		cout << "MAX VAL " << maxVal << endl;
		if (minVal <= 0.05) {
			cv::rectangle(input, start, finish, CV_RGB(0, 255, 0), 2, 8, 0);
			cv::rectangle(result, start, finish, CV_RGB(0, 255, 0), 2, 8, 0);
		}
		else {
			cv::rectangle(input, start, finish, CV_RGB(255, 0, 0), 2, 8, 0);
			cv::rectangle(result, start, finish, CV_RGB(255, 0, 0), 2, 8, 0);
		}
	}
}

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
/*
static double distanceBtwPoints(const cv::Point a, const cv::Point b)
{
	double xDiff = a.x - b.x;
	double yDiff = a.y - b.y;

	return std::sqrt((xDiff * xDiff) + (yDiff * yDiff));
}
*/
/*
void detectparksize() {
	for (int i = 0; i <= detected_edges.size().width; i++) {
		if (detected_edges.at<uchar>(i, 0) == 255)
			cout << "OLA " << i << " " << endl;
	}
}
*/
int main(int argc, char** argv)
{
	Point p;
	string imagename;
	cout << "Please enter the name of the file: ";
	getline(cin, imagename);
	cout << "The value you entered is " << imagename;
	image = imread(imagename, CV_LOAD_IMAGE_COLOR); // Read the file

	while (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		cout << "Please enter the name of the file: ";
		getline(cin, imagename);
		cout << "The value you entered is " << imagename;
		image = imread(imagename, CV_LOAD_IMAGE_COLOR); // Read the file
	}
	imageEmpty = imread("pls.jpg");
	namedWindow("Parking Lot", WINDOW_AUTOSIZE);// Create a window for display.
	namedWindow("Empty Parking Lot", WINDOW_AUTOSIZE);// Create a window for display.
	setMouseCallback("Empty Parking Lot", CallBackFunction, NULL);
	imshow("Parking Lot", image); // Show our image inside it.
	imshow("Empty Parking Lot", imageEmpty); // Show our image inside it.
	cout << "Select two points to create the region of interest and press ESC" << endl;
	while (cv::waitKey(1) != 27);

	cout << roi[0].x << " " << roi[0].y << " " << roi[1].x << " " << roi[1].y;

	regionOfInterest = image(Rect(roi[0].x, roi[0].y, roi[1].x - roi[0].x, (roi[1].y - roi[0].y) * 2));
	regionOfInterestEmpty = imageEmpty(Rect(roi[0].x, roi[0].y, roi[1].x - roi[0].x, (roi[1].y - roi[0].y) * 2));
	cvtColor(regionOfInterestEmpty, greyroiempty, COLOR_RGB2GRAY);
	GaussianBlur(greyroiempty, detected_edges, Size(3, 3), 3);
	Mat ignore;
	double otsu_thresh_val = threshold(detected_edges, ignore, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	double high_thresh_val = otsu_thresh_val, lower_thresh_val = otsu_thresh_val * 0.5;
	// Canny com valores definidos pelo otsu stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
	Canny(detected_edges, detected_edges, lower_thresh_val, high_thresh_val);
	//threshold(detected_edges, detected_edges, 0, 255, THRESH_BINARY | THRESH_OTSU);
	dilate(detected_edges, detected_edges, MORPH_RECT);

	//detectparksize();
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(detected_edges, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0));

	/// Draw contours
	RNG rng(12345);
	Mat drawing = Mat::zeros(detected_edges.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		//cout << "countor " << contours[0][0].x << " " << contours[0][0].y << endl;
		//cout << "countor " << contours[0][1].x << " " << contours[0][1].y << endl;
		//cout << "countor " << contours.size() << " " << endl;
	}
	//cout << "contours" << contours[7][0].x << " " << contours[7][0].y << endl;
	/*sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
		return contourArea(c1, false) < contourArea(c2, false);
	});*/

	std::sort(contours.begin(), contours.end(), contour_sorter());


	rectangle(regionOfInterestEmpty, Point(3, 0), Point(contours[0][0].x - 5, detected_edges.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
	rectangle(regionOfInterest, Point(3, 0), Point(contours[0][0].x - 5, detected_edges.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
	matchTemplate(Point(3, 0), Point(contours[0][0].x - 5, detected_edges.size().height / 2));






	rect.push_back(Point(3, 0));
	rect.push_back(Point(contours[0][0].x - 5, detected_edges.size().height / 2));

	for (int i = 0; i < contours.size() - 1; i++)
	{
		if ((contours[i + 1][0].x - contours[i][0].x) <= 10)
			continue;
		else {
			rectangle(regionOfInterestEmpty, Point(contours[i][0].x + 3, 0), Point(contours[i + 1][0].x - 3, detected_edges.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
			rectangle(regionOfInterest, Point(contours[i][0].x + 3, 0), Point(contours[i + 1][0].x - 3, detected_edges.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
			rect.push_back(Point(contours[i][0].x + 3, 0));
			rect.push_back(Point(contours[i + 1][0].x - 3, detected_edges.size().height / 2));

			matchTemplate(Point(contours[i][0].x + 3, 0), Point(contours[i + 1][0].x - 3, detected_edges.size().height / 2));
		}
		//rectangle(detected_edges, Point pt1, Point pt2, const Scalar& color, int thickness = 1, int lineType = 8, int shift = 0)
	}
	rectangle(regionOfInterestEmpty, Point(contours[contours.size() - 1][0].x + 3, 0), Point(detected_edges.size().width - 3, detected_edges.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
	rectangle(regionOfInterest, Point(contours[contours.size() - 1][0].x + 3, 0), Point(detected_edges.size().width - 3, detected_edges.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
	rect.push_back(Point(contours[contours.size() - 1][0].x + 3, 0));
	rect.push_back(Point(detected_edges.size().width - 3, detected_edges.size().height / 2));

	matchTemplate(Point(contours[contours.size() - 1][0].x + 3, 0), Point(detected_edges.size().width - 3, detected_edges.size().height / 2));
	/// Show in a window
	/*for (int i = 0; i < rect.size(); i++) {
		cout << "I: " << i << "X: " << rect[i].x << "Y: " << rect[i].y << endl;
	}*/









	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);


	destroyWindow("Display Window");

	//namedWindow("ROI GREY", WINDOW_AUTOSIZE);// Create a window for display.

								 //namedWindow("ROY CANNY", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("ROI CANNY", detected_edges); // Show our image inside it.
	imshow("REGION OF INTEREST EMPTY", regionOfInterestEmpty);
	imshow("REGION OF INTEREST", regionOfInterest); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
