#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "harrisDetector.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{


	Mat feup1,feup2;
	feup1 = imread("feup1.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (!feup1.data)
	{
		printf(" No image data \n ");
		return -1;
	}
	feup2 = imread("feup2.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (!feup2.data)
	{
		printf(" No image data \n ");
		return -1;
	}
	/*
	namedWindow("FEUP 1", CV_WINDOW_AUTOSIZE);
	namedWindow("FEUP 2", CV_WINDOW_AUTOSIZE);

	imshow("FEUP 1", feup1);
	imshow("FEUP 2", feup2);

	std::vector< cv::Point2f > corners1,corners2;
	int maxCorners = 10;
	double qualityLevel = 0.01;
	double minDistance = 20.;
	cv::Mat mask;
	int blockSize = 3;
	bool useHarrisDetector = true;
	double k = 0.04;

	cv::goodFeaturesToTrack(feup1, corners1, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	cv::goodFeaturesToTrack(feup2, corners2, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
	for (size_t i = 0; i < corners1.size(); i++)
	{
		cv::circle(feup1, corners1[i], 10, cv::Scalar(0, 0, 0, 0), -1);
		cv::circle(feup2, corners2[i], 10, cv::Scalar(0, 0, 0, 0), -1);
	}

	cv::namedWindow("FEUP 1", CV_WINDOW_NORMAL);
	cv::imshow("FEUP 1", feup1);
	cv::namedWindow("FEUP 2", CV_WINDOW_NORMAL);
	cv::imshow("FEUP 2", feup2);

	*/

	cv::namedWindow("FEUP 1", CV_WINDOW_NORMAL);
	cv::imshow("FEUP 1", feup1);
	cv::namedWindow("FEUP 2", CV_WINDOW_NORMAL);
	cv::imshow("FEUP 2", feup2);
	// Detect Harris Corners
	cv::Mat cornerStrength, cornerStrength2;
	cv::cornerHarris(feup1, cornerStrength,
		3,     // neighborhood size
		3,     // aperture size
		0.01); // Harris parameter
	cv::cornerHarris(feup2, cornerStrength2,
		3,     // neighborhood size
		3,     // aperture size
		0.01); // Harris parameter
	// threshold the corner strengths
	cv::Mat harrisCorners, harrisCorners2;
	double threshold = 0.0001;
	cv::threshold(cornerStrength, harrisCorners,
		threshold, 255, cv::THRESH_BINARY_INV);
	cv::threshold(cornerStrength2, harrisCorners2,
		threshold, 255, cv::THRESH_BINARY_INV);
	// Display the corners
	cv::namedWindow("Harris Corner Map");
	cv::imshow("Harris Corner Map", harrisCorners);
	cv::namedWindow("Harris Corner Map 2");
	cv::imshow("Harris Corner Map 2", harrisCorners2);
	// Create Harris detector instance
	HarrisDetector harris, harris2;
	// Compute Harris values
	harris.detect(feup1);
	harris2.detect(feup1);
	// Detect Harris corners
	std::vector<cv::Point> pts,pts2;
	harris.getCorners(pts, 0.01);
	harris2.getCorners(pts2, 0.01);
	// Draw Harris corners
	harris.drawOnImage(feup1, pts);
	harris2.drawOnImage(feup2, pts2);
	// Display the corners
	cv::namedWindow("Harris Corners");
	cv::imshow("Harris Corners", feup1);
	cv::namedWindow("Harris Corners 2");
	cv::imshow("Harris Corners 2", feup2);
	feup1 = imread("feup1.png", CV_LOAD_IMAGE_GRAYSCALE);
	feup2 = imread("feup2.png", CV_LOAD_IMAGE_GRAYSCALE);
	// vector of keypoints
	std::vector<cv::KeyPoint> keypoints, keypoints2;
	// Construction of the Good Feature to Track detector 
	cv::GoodFeaturesToTrackDetector gftt(
		500,	// maximum number of corners to be returned
		0.01,	// quality level
		10,3,true);	// minimum allowed distance between points
				// point detection using FeatureDetector method
	gftt.detect(feup1, keypoints);
	gftt.detect(feup2, keypoints2);

	cv::drawKeypoints(feup1,		// original image
		keypoints,					// vector of keypoints
		feup1,						// the resulting image
		cv::Scalar(255, 255, 255),	// color of the points
		cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag

												 // Display the corners
	cv::namedWindow("Good Features to Track Detector");
	cv::imshow("Good Features to Track Detector", feup1);

	cv::drawKeypoints(feup2,		// original image
		keypoints2,					// vector of keypoints
		feup2,						// the resulting image
		cv::Scalar(255, 255, 255),	// color of the points
		cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag

												 // Display the corners
	cv::namedWindow("Good Features to Track Detector 2");
	cv::imshow("Good Features to Track Detector 2", feup2);
	// Read input image
	feup1 = imread("feup1.png", CV_LOAD_IMAGE_GRAYSCALE);
	feup2 = imread("feup2.png", CV_LOAD_IMAGE_GRAYSCALE);
	keypoints.clear();
	keypoints2.clear();
	cv::FastFeatureDetector fast(40);
	fast.detect(feup1, keypoints);
	fast.detect(feup2, keypoints2);

	cv::drawKeypoints(feup1, keypoints, feup1, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
	cv::drawKeypoints(feup2, keypoints2, feup2, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

	// Display the corners
	cv::namedWindow("FAST Features");
	cv::imshow("FAST Features", feup1);
	cv::namedWindow("FAST Features 2");
	cv::imshow("FAST Features 2", feup2);
	cv::waitKey(0);

	waitKey(0);

	return 0;
}
