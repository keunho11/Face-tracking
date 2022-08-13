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
#define SKIP_FRAMES 10

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

        image_window win;

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;

		double fps = 0.0; 
		int count = 0;

        while(!win.is_closed())
		while(1)
        {
			double	t1 = cv::getTickCount();

	        cv::Mat temp;
			cv::Mat temp_small;
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
            for (unsigned long i = 0; i < faces.size(); ++i)
			{
				dlib::rectangle r(
                            (long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO),
                            (long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO)
                            );
				shape = pose_model(cimg, r);
                shapes.push_back(shape);

			}

//			std::cout << "nose point : " << shape.part(33)<< std::endl;
			/****************************/
			/*
			render_face(temp, shape);
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
			*/

//			cv::putText(temp, cv::format("fps %.2f",fps), cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255),3);

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));

			count++;
			cout << "count : " << count << endl;

			double t2 = cv::getTickCount();
		    double time = (t2-t1)/cv::getTickFrequency();
           	cout << "excute time(ms) :" << time << endl; 

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

