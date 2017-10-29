#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>       

using namespace std;
using namespace cv;

/**
 * convert to greyscale image
 * @param frame frame from camera
 * @return new image
 */
Mat convertToGreyScale(const Mat &frame);

/**
 * find maximum brightness point.
 * @param image our frame
 * @return maximum brightness value.
 */
uchar maxBrightness(const Mat &image);

/**
 * find maximum bright points and draw circle on our frame.
 * @param greyImage greystcale image
 * @param frame to be displayed
 */
void findmaxBrightnessAndDrawImage(const Mat &greyImage, Mat &frame);

int main(){

    Mat image;                  // image
    VideoCapture capture(0);    // open the default camera

    if(!capture.isOpened())     // check if we succeeded
        return -1;

    // to calculate fps
    double fps = 0.0;
    int num_frames = 0;
     
    // Start and end times
    time_t start, end;
    time(&start);

    while(true){        

        Mat frame;

        capture >> frame;                  // get a new frame from camera.

        image = convertToGreyScale(frame); // convert frame to greyscale image.

        findmaxBrightnessAndDrawImage(image, frame);    // find maximum brightness

        // calculate fps.
        time(&end);
        num_frames++;
        double seconds = difftime (end, start);
        fps  = num_frames / seconds;
        
        // to convert double number to string
        ostringstream strs;
        strs << "fps: " << fps;
        string str = strs.str();

        putText(frame, str, cvPoint(10, 465), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255));
        imshow("Original Image", frame);   // display greyscale and original image.

        if(waitKey(30) >= 0)
            break;
    }

    return 0;
}

Mat convertToGreyScale(const Mat &frame){

    Mat image(frame.rows, frame.cols, CV_8UC1, Scalar(0, 0, 0));

    // Three algorithms for converting color to grayscale.
    // lightness, average, luminosity.
    // I used to luminosity.
    for(int i = 0; i < frame.rows; i++){
        for(int j = 0; j < frame.cols; j++){
            Vec3b pixel = frame.at<Vec3b>(i, j);
            image.at<uchar>(i,j) = pixel[0] * 0.07 + pixel[1] * 0.72 + pixel[2] * 0.21;
        }
    }

    return image;
}

uchar maxBrightness(const Mat &image){

    uchar brightness = 0;

    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            uchar point = image.at<uchar>(i, j);
            if(point >= brightness){
                brightness = point;
            }
        }
    }

    return brightness;
}

void findmaxBrightnessAndDrawImage(const Mat &greyImage, Mat &frame){

    vector<Point> firstPoints;
    vector<Point> secondPoints;
    vector<Point> thirdPoints;
    uchar max = maxBrightness(greyImage);

    // brightness range threshold.
    int first = 20;
    int second = 65;
    int third = 90;

    for( int i = 0; i < greyImage.rows; i++ ) {
        for( int j = 0; j < greyImage.cols; j++ ) {
            uchar pixel = greyImage.at<uchar>(i, j);
            if(pixel > max - first){
                firstPoints.push_back(Point(j,i));
            }

            if((pixel > max - second) && (pixel < max - first)){
                secondPoints.push_back(Point(j,i));
            }

            if((pixel > max - third) && (pixel < max - second)){
                thirdPoints.push_back(Point(j,i));
	    }
        }
    }

    Point p;
    Point minP;
    Point maxP;

    if(firstPoints.size() != 0) {
        p = firstPoints.at(firstPoints.size() / 2);
        minP = firstPoints.at(0);
        maxP = firstPoints.at(firstPoints.size() - 1);
        circle(frame, Point(p.x, p.y), abs(maxP.y - minP.y) / 6, Scalar(255, 255, 0), 2, 8);
    	putText(frame, "1.", cvPoint(p.x, p.y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(255, 255, 0), 1);
    }

    if(secondPoints.size() != 0) {
        p = secondPoints.at(secondPoints.size() / 2);
        minP = secondPoints.at(0);
        maxP = secondPoints.at(secondPoints.size() - 1);
        circle(frame, Point(p.x, p.y), abs(maxP.y - minP.y) / 6, Scalar(0, 255, 0), 2, 8);
    	putText(frame, "2.", cvPoint(p.x, p.y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 255, 0), 1);
    }

    if(thirdPoints.size()!= 0) {
        p = thirdPoints.at(thirdPoints.size() / 2);
        minP = thirdPoints.at(0);
        maxP = thirdPoints.at(thirdPoints.size() - 1);
        circle(frame, Point(p.x, p.y), abs(maxP.y - minP.y) / 6, Scalar(0, 0, 255), 2, 8);
    	putText(frame, "3.", cvPoint(p.x, p.y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 0, 255), 1);
    }
}
