#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

std::vector<cv::Point> vec_match_locs;
//determine whether there are point in pts is near in_pt(less than distance)
bool HaveNeighborPoint(cv::Point in_pt, std::vector<cv::Point> pts, float distance){
	if (pts.empty()){
		return false;
	}
	for (int i = 0; i < pts.size(); i++){
		float d = std::sqrt( (pts.at(i).x - in_pt.x)*(pts.at(i).x - in_pt.x) + (pts.at(i).y - in_pt.y)*(pts.at(i).y - in_pt.y) );
		if (d < distance) return true;
	}
	return false;
}
void match_img(cv::Mat& img, cv::Mat templ, cv::Mat& mask, cv::Mat& img_display, int match_method,float thres){

	// TM_CCORR_NORMED TM_CCOEFF TM_CCOEFF_NORMED TM_SQDIFF; TM_CCORR not work
	bool use_mask = false;
	if (!mask.empty()){
		use_mask = true;
	}
	cv::Mat result;
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);

	bool method_accepts_mask = (CV_TM_SQDIFF == match_method || match_method == CV_TM_CCORR_NORMED);
	if (use_mask && method_accepts_mask)
	{
	//	cv::matchTemplate(img, templ, result, match_method, mask);
	}
	else
	{
		matchTemplate(img, templ, result, match_method);
	}
	normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	cv::Point matchLoc;
	float min_distance = std::min(templ.rows/2,templ.cols/2);
	if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED)
	{
		 thres = 1-thres;
		for (int i = 0; i < result.rows; i++){
			for (int j = 0; j < result.cols; i++){
				if (result.at<float>(i, j) <thres){
					matchLoc = cv::Point(j, i);
					if (!HaveNeighborPoint(matchLoc, vec_match_locs, min_distance)){
						vec_match_locs.push_back(matchLoc);
						rectangle(img_display, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
					}
					
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < result.rows; i++){
			for (int j = 0; j < result.cols; j++){
				if (result.at<float>(i, j) >thres){
					matchLoc = cv::Point(j, i);
					if (!HaveNeighborPoint(matchLoc, vec_match_locs, min_distance)){
						vec_match_locs.push_back(matchLoc);
						rectangle(img_display, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
					}
				}
			}
		}

	}


	//imshow("frame", img_display);
	//imshow("result", result);
	//cv::waitKey(0);
}

//match homography
void match_homography_img(cv::Mat img,cv::Mat templ){
	cv::Mat img_object = templ;
	cv::Mat img_scene = img;

	if (!img_object.data || !img_scene.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return;
	}
	if (img_object.type() != CV_8UC1 || img_object.type() != CV_8UC1){
		img_object.convertTo(img_object,CV_8UC1);
		img_scene.convertTo(img_scene, CV_8UC1);
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	
	cv::SurfFeatureDetector detector(minHessian);

	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;

	detector.detect(img_object, keypoints_object);
	detector.detect(img_scene, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)
	cv::SurfDescriptorExtractor extractor;

	cv::Mat descriptors_object, descriptors_scene;

	extractor.compute(img_object, keypoints_object, descriptors_object);
	extractor.compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector< cv::DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< cv::DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	cv::Mat img_matches;
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
	std::vector<cv::Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + cv::Point2f(img_object.cols, 0), scene_corners[1] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + cv::Point2f(img_object.cols, 0), scene_corners[2] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + cv::Point2f(img_object.cols, 0), scene_corners[3] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + cv::Point2f(img_object.cols, 0), scene_corners[0] + cv::Point2f(img_object.cols, 0), cv::Scalar(0, 255, 0), 4);

	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);

	cv::waitKey(0);
}
void rotate_img(cv::Mat& org_img,cv::Mat& rotated_img,float angle){
	if (org_img.empty())
	{
		std::cout << "read image failure" << std::endl;
		return ;
	}
	IplImage* src = new IplImage(org_img);
	//cvNamedWindow( "src", 1 );
	cv::Mat mat1 = cv::cvarrToMat(src);
	cv::imshow(" src img", mat1);
	float anglerad = CV_PI*angle / 180.0;
	//输入图像的大小
	int w = src->width;
	int h = src->height;
	//旋转后图像的大小
	int w_dst = int(fabs(h*sin(anglerad)) + fabs(w*cos(anglerad)));
	int h_dst = int(fabs(w * sin(anglerad)) + fabs(h * cos(anglerad)));
	w_dst = std::max(w_dst, src->width);
	h_dst = std::max(h_dst, src->height);
	CvSize rect;
	//rect.height = std::max(h_dst,src->height);
	//rect.width = std::max(w_dst, src->width);
	rect.height = h_dst;
	rect.width = w_dst;
	//中间变量
	IplImage *des = cvCreateImage(rect, src->depth, src->nChannels);
	//旋转后的图像
	IplImage *des_rot = cvCreateImage(rect, src->depth, src->nChannels);
	//用0填充
	//cvFillImage(des,0);
	cvSet(des,cv::Scalar(255,255,255));

	//设置roi区域，将原图copy到roi区域
	CvRect roi;
	roi.x = (w_dst - w) / 2;
	roi.y = (h_dst - h) / 2;
	roi.height = h;
	roi.width = w;
	cvSetImageROI(des, roi);
	cvCopy(src, des, NULL);
	
	cvResetImageROI(des);
	//旋转矩阵
	float m[6];
	CvMat M = cvMat(2, 3, CV_32F, m);

	m[0] = (float)cos(-anglerad);
	m[1] = (float)sin(-anglerad);
	m[3] = -m[1];
	m[4] = m[0];
	// 将旋转中心移至图像中间
	m[2] = w_dst*0.5f;
	m[5] = h_dst*0.5f;
	cvGetQuadrangleSubPix(des, des_rot, &M);
	//cvNamedWindow( "dst", 1 );




	cv::Mat mat = cv::cvarrToMat(des_rot);
	rotated_img = mat;
	//cv::imshow("rotated img", mat);
	//cv::waitKey(0);



}
int main(){
	int match_method = cv::TM_CCOEFF;
	cv::Mat test_img = cv::imread("Data/image1.png");
	cv::Mat templ = cv::imread("Data/template.png");
	cv::Mat mask;
	//match img
	// TM_CCORR_NORMED TM_CCOEFF TM_CCOEFF_NORMED TM_SQDIFF; TM_CCORR not work
	cv::Mat rotated_img, img_display;
	test_img.copyTo(img_display);
	for (int i = 0; i < 36; i++){
		
		rotate_img(templ, rotated_img, i*10);
		match_img(test_img, rotated_img, mask, img_display, match_method,0.9);
		
		
	}
	imshow("frame", img_display);
	cv::waitKey(0);
	//video
	cv::VideoCapture cap("Data/video1.avi");
	if (!cap.isOpened() || templ.empty()) {
		return 0;
	}
	int n_cnt = 0;
	bool use_mask = false;
	while (true){
		cv::Mat img;
		bool b_ret= cap.read(img);
		if (!b_ret)
			break;
		std::cout << "frame " << n_cnt << std::endl;
		
		match_homography_img(img, templ);

		n_cnt += 1;
	}


	return 0;
}