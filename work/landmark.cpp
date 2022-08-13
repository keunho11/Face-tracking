#include <iostream>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dlib;

#define COLOR_DETECT Scalar(0, 255, 0)

// connect line
void connectLine(cv::Mat & img, full_object_detection landmarks, int iStart, int iEnd, bool isClosed = false)
{
	std::vector<cv::Point> points;
	for (int i = iStart ; i < iEnd ; i++)
	{
		points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));
	}
	cv::polylines(img, points, isClosed, COLOR_DETECT, 2, 16);
}

// draw polygon
void drawPolygon(cv::Mat & img, full_object_detection landmarks)
{
	// shape_predictor_68_face_landmarks
	// facial landmark number: face.png
	connectLine(img, landmarks, 0, 16); // 턱
	connectLine(img, landmarks, 17, 21); // 왼쪽 눈썹
	connectLine(img, landmarks, 22, 26); // 오른쪽 눈썹
	connectLine(img, landmarks, 27, 30); // 콧대
	connectLine(img, landmarks, 30, 35, true); // 낮은 코
	connectLine(img, landmarks, 36, 41, true); // 왼쪽 눈
	connectLine(img, landmarks, 42, 47, true); // 오른쪽 눈
	connectLine(img, landmarks, 48, 59, true); // 입술 바깥쪽
	connectLine(img, landmarks, 60, 67, true); // 입술 안쪽 부분
}

// dlib rectangle to opencv rect
cv::Rect dlibRectToOpencv(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right(), r.bottom()));
}

int main()
{
	Mat img = imread("lenna.bmp", IMREAD_COLOR);
	if (img.empty())
	{
		cerr << "img open fail" << endl;
		return -1;
	}

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor landmarkDetector;
	deserialize(".\\shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

	cv_image<bgr_pixel> dlib_img(img);
	std::vector<dlib::rectangle> faceRects = detector(dlib_img);
	int iFaceCount = faceRects.size();

	// draw
	for (int i = 0 ; i < iFaceCount ; i++)
	{
		full_object_detection faceLandmark = landmarkDetector(dlib_img, faceRects[i]);
		drawPolygon(img, faceLandmark);
	}

	imshow("img", img);
	waitKey();

	destroyAllWindows();
	return 0;
}
