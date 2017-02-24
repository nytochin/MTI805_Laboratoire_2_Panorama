// MTI805_Laboratoire_2_Panorama.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main()
{

	Mat rightImg = imread("right1.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat leftImg = imread("left1.png", CV_LOAD_IMAGE_UNCHANGED);

	if (!rightImg.data || !leftImg.data)
	{
		cout << "Error : Images are not loaded..!!" << endl;
		return -1;
	}

	// Convert to gray - scale.

	Mat gray_rightImg;
	Mat gray_leftImg;
	Mat gray_rightImg_keyPoints;
	Mat gray_leftImg_keyPoints;
	cvtColor(rightImg, gray_rightImg, CV_RGB2GRAY);
	cvtColor(leftImg, gray_leftImg, CV_RGB2GRAY);

	// Detect key - points / features.

	int minHessian = 400;
	Ptr<SIFT> detector = SIFT::create(minHessian);
	std::vector< KeyPoint > keypoints_right, keypoints_left;
	detector->detect(gray_rightImg, keypoints_right);
	detector->detect(gray_leftImg, keypoints_left);

	// Display keypoints
	drawKeypoints(gray_rightImg, keypoints_right, gray_rightImg_keyPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("keyPoints_right", gray_rightImg_keyPoints);
	imwrite("sift_keypoints_right.jpg", gray_rightImg_keyPoints);

	drawKeypoints(gray_leftImg, keypoints_left, gray_leftImg_keyPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("keyPoints_left", gray_leftImg_keyPoints);
	imwrite("sift_keypoints_left.jpg", gray_leftImg_keyPoints);

	// Extract features.

	Ptr<SIFT> extractor = SIFT::create();
	cv::Mat descriptors_right, descriptors_left;
	extractor->compute(gray_rightImg, keypoints_right, descriptors_right);
	extractor->compute(gray_leftImg, keypoints_left, descriptors_left);

	// Match feature using Flann matcher
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_right, descriptors_left, matches);
	double max_dist = 0;
	double min_dist = 100;

	//  Recomputation of max distance & min distance
	for (int i = 0; i < descriptors_right.rows; i++) {
		double dist = matches[i].distance;
		if (dist > max_dist) max_dist = dist;
		if (dist < min_dist) min_dist = dist;
	}

	// Keep only the good matches (whose distance is less than 3*min_dist)
	vector<DMatch> good_matches;
	for (int i = 0; i < descriptors_right.rows; i++) {
		if (matches[i].distance < 3 * min_dist) {
			good_matches.push_back(matches[i]);
		}
	}

	// Draw matches
	Mat img_matches;
	drawMatches( gray_rightImg, keypoints_right, gray_leftImg, keypoints_left,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// Show detected matches
	imshow("Good Matches", img_matches);
	imwrite("matches.jpg", img_matches);

	vector<Point2f> obj;
	vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++) {
		obj.push_back(keypoints_right[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_left[good_matches[i].trainIdx].pt);
	}

	// Warp the image using estimated homography and blend the images
	Mat H = findHomography(obj, scene, CV_RANSAC);

	Mat result;
	warpPerspective(rightImg, result, H, cv::Size(rightImg.cols + leftImg.cols, rightImg.rows));
	Mat half(result, Rect(0, 0, leftImg.cols, leftImg.rows));
	leftImg.copyTo(half);
	imshow("Panorama", result);

	waitKey(0);
}

