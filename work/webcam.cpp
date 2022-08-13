#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;
using namespace cv;
#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 60

void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
    }
    cv::polylines(img, points, isClosed, cv::Scalar(255,0,0), 2, 16);

}

void render_face (cv::Mat &img, const dlib::full_object_detection& d)
{
	
    DLIB_CASSERT
    (
     d.num_parts() == 68,
     "\t std::vector<image_window::overlay_line> render_face_detections()"
     << "\n\t Invalid inputs were given to this function. "
     << "\n\t d.num_parts():  " << d.num_parts()
     );
	
    draw_polyline(img, d, 0, 16);           // Jaw line
    draw_polyline(img, d, 17, 21);          // Left eyebrow
    draw_polyline(img, d, 22, 26);          // Right eyebrow
    draw_polyline(img, d, 27, 30);          // Nose bridge
    draw_polyline(img, d, 30, 35, true);    // Lower nose
    draw_polyline(img, d, 36, 41, true);    // Left eye
    draw_polyline(img, d, 42, 47, true);    // Right Eye
    draw_polyline(img, d, 48, 59, true);    // Outer lip
    draw_polyline(img, d, 60, 67, true);    // Inner lip
	 
}

std::vector<cv::Point3d> get_3d_model_points()
{
    std::vector<cv::Point3d> modelPoints;

    modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0) while using POSIT
    modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
    modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
    modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
    modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));

    return modelPoints;

}

std::vector<cv::Point2d> get_2d_image_points(full_object_detection &d)
{
    std::vector<cv::Point2d> image_points;
    image_points.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );    // Nose tip
    image_points.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );      // Chin
    image_points.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    // Left eye left corner
    image_points.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );    // Right eye right corner
    image_points.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );    // Left Mouth corner
    image_points.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );    // Right mouth corner
    return image_points;

}

cv::Mat get_camera_matrix(float focal_length, cv::Point2d center)
{
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    return camera_matrix;
}

/*****************************************/

int main()
{
    try
    {
        cv::VideoCapture cap;
		cap.open("/dev/video2");
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

	    cv::Mat temp;
		cap >> temp;
		cv::Mat temp_small;
		cv::Mat temp_display;

		cv::resize(temp, temp_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
		cv::resize(temp, temp_display, cv::Size(), 0.5,0.5);

//        image_window win;

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;

		double fps = 0.0; 
		int count = 0;

//        while(!win.is_closed())
		while(1)
        {
			double	t1 = cv::getTickCount();

//			cv::resize(temp, temp_display, cv::Size(), 0.5,0.5);
            // Grab a frame
            if (!cap.read(temp))
            {
                break;
            }
			cv::resize(temp, temp_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);

            cv_image<bgr_pixel> cimg(temp);
            cv_image<bgr_pixel> cimg_small(temp_small);

			std::vector<dlib::rectangle> faces;
			// Detect faces
			if(count % SKIP_FRAMES == 0)
			{
            	faces = detector(cimg_small);
			}

            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
			full_object_detection shape;
				
			if(faces.size() != 0)
				std::cout << "faces size : " << faces.size()<< std::endl;
            for (unsigned long i = 0; i < faces.size(); ++i)
			{
				dlib::rectangle r(
                            (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
                            );
				shape = pose_model(cimg, r);
//              shapes.push_back(shape);
			
				render_face(temp, shape);

				/**********solvePnP*************/
				std::vector<cv::Point2d> image_points = get_2d_image_points(shape);
				std::vector<cv::Point3d> model_points = get_3d_model_points();
				double focal_length = temp.cols;
				cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(temp.cols/2,temp.rows/2));
				cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);
				cv::Mat rotation_vector;
				cv::Mat translation_vector;

				cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

				std::vector<cv::Point3d> nose_end_point3D;
				std::vector<cv::Point2d> nose_end_point2D;
				nose_end_point3D.push_back(cv::Point3d(0,0,1000.0));

				cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
				cv::line(temp,image_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);

				/********EulerAngle********/
				cv::Mat rotation_matrix;
				cv::Rodrigues(rotation_vector, rotation_matrix);
				// get eulerAngles
				cv::Vec3d eulerAngles;
				cv::Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
				double *_r = rotation_matrix.ptr<double>();
				double projMatrix[12] = {_r[0], _r[1], _r[2], 0,
					_r[3], _r[4], _r[5], 0,
					_r[6], _r[7], _r[8], 1};

				cv::decomposeProjectionMatrix(cv::Mat(3, 4, CV_64FC1, projMatrix),
						cameraMatrix,
						rotMatrix,
						transVect,
						rotMatrixX,
						rotMatrixY,
						rotMatrixZ,
						eulerAngles);

				double xTurn = eulerAngles[0];
				double yTurn = eulerAngles[1];
				double zTurn = eulerAngles[2];
				std::cout << "x : " << xTurn << std::endl;
				std::cout << "y : " << yTurn << std::endl;
				std::cout << "z : " << zTurn << std::endl;
			}

			cv::imshow("OpenCV Landmark Detector", temp);
			if ( count % 15 == 0)
			{
				int k = cv::waitKey(1);
				// Quit if 'q' or ESC is pressed
				if ( k == 'q' || k == 27)
				{
					return 0;
				}
			}

            // Display it all on the screen
//            win.clear_overlay();
//            win.set_image(cimg);
//            win.add_overlay(render_face_detections(shapes));

/*
			count++;
			cout << "count : " << count << endl;

			double t2 = cv::getTickCount();
		    double time = (t2-t1)/cv::getTickFrequency();
           	cout << "excute time(ms) :" << time << endl; 
*/
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

