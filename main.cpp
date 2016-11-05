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
int nRois = 1;
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

void detect_edges(Mat &imageToDetect) {
	GaussianBlur(greyroiempty, imageToDetect, Size(3, 3), 3);
	Mat ignore;
	double otsu_thresh_val = threshold(imageToDetect, ignore, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double high_thresh_val = otsu_thresh_val, lower_thresh_val = otsu_thresh_val * 0.5;
	// Canny com valores definidos pelo otsu stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
	Canny(imageToDetect, imageToDetect, lower_thresh_val, high_thresh_val);
	dilate(imageToDetect, imageToDetect, MORPH_RECT);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(imageToDetect, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0));

	/// Draw contours
	RNG rng(12345);
	Mat drawing = Mat::zeros(imageToDetect.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	std::sort(contours.begin(), contours.end(), contour_sorter());

	if (contours[0][0].x - 5 >= 0) {
		rectangle(regionOfInterestEmpty, Point(3, 0), Point(contours[0][0].x - 5, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
		rectangle(regionOfInterest, Point(3, 0), Point(contours[0][0].x - 5, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
		matchTemplate(Point(3, 0), Point(contours[0][0].x - 5, imageToDetect.size().height / 2));
		rect.push_back(Point(3, 0));
		rect.push_back(Point(contours[0][0].x - 5, imageToDetect.size().height / 2));
	}
	else {
		rectangle(regionOfInterestEmpty, Point(3, 0), Point(0, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
		rectangle(regionOfInterest, Point(3, 0), Point(0, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
		matchTemplate(Point(3, 0), Point(0, imageToDetect.size().height / 2));
		rect.push_back(Point(3, 0));
		rect.push_back(Point(0, imageToDetect.size().height / 2));
	}



	for (int i = 0; i < contours.size() - 1; i++)
	{
		if ((contours[i + 1][0].x - contours[i][0].x) <= 15)
			continue;
		else {
			if (contours[i + 1][0].x - 3 >= 0) {
				rectangle(regionOfInterestEmpty, Point(contours[i][0].x + 3, 0), Point(contours[i + 1][0].x - 3, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
				rectangle(regionOfInterest, Point(contours[i][0].x + 3, 0), Point(contours[i + 1][0].x - 3, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
				rect.push_back(Point(contours[i][0].x + 3, 0));
				rect.push_back(Point(contours[i + 1][0].x - 3, imageToDetect.size().height / 2));

				matchTemplate(Point(contours[i][0].x + 3, 0), Point(contours[i + 1][0].x - 3, imageToDetect.size().height / 2));
			}
			else {
				rectangle(regionOfInterestEmpty, Point(contours[i][0].x + 3, 0), Point(0, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
				rectangle(regionOfInterest, Point(contours[i][0].x + 3, 0), Point(0, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
				rect.push_back(Point(contours[i][0].x + 3, 0));
				rect.push_back(Point(0, imageToDetect.size().height / 2));

				matchTemplate(Point(contours[i][0].x + 3, 0), Point(0, imageToDetect.size().height / 2));
			}
		}
	}
	rectangle(regionOfInterestEmpty, Point(contours[contours.size() - 1][0].x + 3, 0), Point(imageToDetect.size().width - 3, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
	rectangle(regionOfInterest, Point(contours[contours.size() - 1][0].x + 3, 0), Point(imageToDetect.size().width - 3, imageToDetect.size().height / 2), CV_RGB(255, 255, 255), 1, 8, 0);
	rect.push_back(Point(contours[contours.size() - 1][0].x + 3, 0));
	rect.push_back(Point(imageToDetect.size().width - 3, imageToDetect.size().height / 2));

	matchTemplate(Point(contours[contours.size() - 1][0].x + 3, 0), Point(imageToDetect.size().width - 3, imageToDetect.size().height / 2));

	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);

}

int main(int argc, char** argv)
{
	Point p;
	string imagename;
	cout << "Please enter the name of the file: ";
	getline(cin, imagename);
	cout << "The value you entered is " << imagename << "\n\n";
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
	setMouseCallback("Parking Lot", CallBackFunction, NULL);
	imshow("Parking Lot", image); // Show our image inside it.
	imshow("Empty Parking Lot", imageEmpty); // Show our image inside it.
	cout << "Select two points to create the region of interest and press SPACE" << endl;

	while (nRois <= 5) {

		while (cv::waitKey(1) != 32);

		if (nRois > 1) {
			destroyWindow("Contours");
			destroyWindow("ROI CANNY");
			destroyWindow("REGION OF INTEREST EMPTY");
			destroyWindow("REGION OF INTEREST");
			regionOfInterest.release(); 
			regionOfInterestEmpty.release();
			greyroi.release();
			greyroiempty.release();
			detected_edges.release();
		}

		regionOfInterest = image(Rect(roi[0].x, roi[0].y, roi[1].x - roi[0].x, (roi[1].y - roi[0].y) * 2));
		regionOfInterestEmpty = imageEmpty(Rect(roi[0].x, roi[0].y, roi[1].x - roi[0].x, (roi[1].y - roi[0].y) * 2));
		cvtColor(regionOfInterestEmpty, greyroiempty, COLOR_RGB2GRAY);

		roi.clear();

		detect_edges(detected_edges);

		namedWindow("ROI CANNY", WINDOW_AUTOSIZE);
		namedWindow("REGION OF INTEREST EMPTY", WINDOW_AUTOSIZE);
		namedWindow("REGION OF INTEREST", WINDOW_AUTOSIZE);

		imshow("ROI CANNY", detected_edges); // Show our image inside it.
		imshow("REGION OF INTEREST EMPTY", regionOfInterestEmpty);
		imshow("REGION OF INTEREST", regionOfInterest); // Show our image inside it.

		cout << "ROI created\n\n";
		
		nRois++;

	}

	

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
