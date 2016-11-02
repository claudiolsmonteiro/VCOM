#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;


string windowname = "Region of Interest";
Mat image, regionOfInterest, greyroi, detected_edges;
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
    
    cout << "Select two points to create the region of interest and press ESC" << endl;
	while (cv::waitKey(1) != 27);
	
    cout << roi[0].x << " " << roi[0].y << " " << roi[1].x << " " << roi[1].y;
	
    regionOfInterest = image(Rect(roi[0].x, roi[0].y, roi[1].x - roi[0].x, roi[1].y - roi[0].y));
    
    cvtColor(regionOfInterest, greyroi, COLOR_RGB2GRAY);
    GaussianBlur( greyroi, detected_edges, Size(3,3), 0.5,0.5);
    Mat ignore;
    double otsu_thresh_val = threshold(detected_edges,ignore, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    double high_thresh_val  = otsu_thresh_val,lower_thresh_val = otsu_thresh_val * 0.5;
    // Canny com valores definidos pelo otsu stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    Canny( detected_edges, detected_edges, lower_thresh_val, high_thresh_val );
    imshow("otsu", detected_edges);
    // Canny com valores hardcoded
    //Canny( detected_edges, detected_edges, 30, 30*3, 3 );
    //imshow("canny",detected_edges);
    /// Find contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( detected_edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    /// Draw contours
    RNG rng(12345);

    Mat drawing = Mat::zeros( detected_edges.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }
    
    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    
    vector<vector<Point> >hull( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {   convexHull( Mat(contours[i]), hull[i], false ); }
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        drawContours( drawing, hull, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
    }
    namedWindow( "Hull demo", WINDOW_AUTOSIZE );
    imshow( "Hull demo", drawing );

    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }
    /// Draw polygonal contour + bonding rects + circles
    drawing = Mat::zeros( detected_edges.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    }
    /// Show in a window
    namedWindow( "POLY", CV_WINDOW_AUTOSIZE );
    imshow( "POLY", drawing );
    
    
    destroyWindow("Display Window");
	
    //namedWindow("ROI GREY", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("ROI GREY", greyroi); // Show our image inside it.
    
    //namedWindow("ROY CANNY", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("ROI CANNY", detected_edges); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
