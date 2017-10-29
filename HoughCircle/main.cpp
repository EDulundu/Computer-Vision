#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/types_c.h>

// Kullanici tarafindan degistirilebilen inputlar
#define PI     3.14159265
#define COLS   120
#define ROWS   160
#define RADIUS 200
#define RADIUS_INCREMENT 1
#define FI_INCREMENT 3

using namespace std;
using namespace cv;

// cos ve sin degerlerini bir arrayde tuttuk yoksa
// fps cok dusuk oluyordu.
int ***Accumulator;
double cosValues[120];
double sinValues[120];
bool stepByStep = false;

void ContinuousMode();
void StepByStepMode();
void HoughCircleTransform(Mat& image);

int main(){

    // allocate for accumulator pointer
    Accumulator  = new int **[COLS];
    for (int i = 0; i < COLS; ++i) {
        Accumulator[i] = new int *[ROWS];
        for (int j = 0; j < ROWS; ++j) {
            Accumulator[i][j] = new int[RADIUS];
            for (int k = 0; k < RADIUS; ++k) {
                Accumulator[i][j][k] = 0;
            }
        }
    }

    // evaluate sin and cos values for every angle
    for (int l = 0; l < 360; l += FI_INCREMENT) {
        cosValues[l/FI_INCREMENT] = cos((l * PI)/180);
        sinValues[l/FI_INCREMENT] = sin((l * PI)/180);
    }

    int mode;
    do{
        cout << "\n1) Continuous Mode\n2) Step by Step Mode\n" << endl;
        cout << "Please select Mode > ";
        cin >> mode;
        switch (mode){
            case 1:
                ContinuousMode();
                break;
            case 2:
                StepByStepMode();
                break;
            default:
                cerr << "Wrong input !!! Please select 1 or 2 !!!" << endl;
                break;
        }
    }while(mode != 1 && mode != 2);

    // deallocate for accumulator pointer
    for(int i = 0; i < COLS; ++i) {
        for(int j = 0; j < ROWS; ++j)
            delete[] Accumulator[i][j];
        delete[] Accumulator[i];
    }
    delete[] Accumulator;

    return 0;
}

void ContinuousMode(){

    VideoCapture capture(0);

    if(!capture.isOpened())
        return;

    time_t start, end;
    time(&start);

    double fps;
    int num_frames = 0;
    while(true) {

        Mat frame;
        capture >> frame;

        // calculate fps.
        time(&end);
        num_frames++;
        double seconds = difftime(end, start);
        fps = num_frames / seconds;

        ostringstream strs;
        strs << "fps: " << fps;
        string str = strs.str();

        // hough transform for circle detection
        HoughCircleTransform(frame);
        putText(frame, str, cvPoint(10, 465), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255));
        imshow("Circle Detection", frame);

        if(waitKey(30) >= 0)
            break;
    }
}

void StepByStepMode(){
    
}

void HoughCircleTransform(Mat& image) {

    Mat dst, cdst, frame, votingSpace;

    resize(image, frame, Size(image.cols * 0.25, image.rows * 0.25), 0.25, 0.25, CV_INTER_LINEAR);
    cvtColor(frame, dst, COLOR_BGR2GRAY);
    GaussianBlur(dst, cdst, Size(7,7), 1.5, 1.5);
    Canny(cdst, cdst, 50, 150, 3);

    vector<Point> edges;
    for (int i = 0; i < cdst.rows; i++)
        for (int j = 0; j < cdst.cols; j++)
            if ((int) cdst.at<uchar>(i, j) == 255)
                edges.push_back(Point(i, j));

    for(int i = 0; i < edges.size(); i++){
        Point pixel = edges.at(i);
        for (int r = 10; r < 100; r += RADIUS_INCREMENT) {
            for (int fi = 0; fi < 360; fi += FI_INCREMENT) {
                int a = pixel.x - r * cosValues[fi/3];
                int b = pixel.y - r * sinValues[fi/3];
                if( (a >= 0 && a < frame.rows) && (b >= 0 && b < frame.cols) )
                    Accumulator[a][b][r]++;
            }
        }
    }

    votingSpace.create(Size(640,480), CV_8UC3);
    votingSpace = Scalar::all(0);
    vector<Point> vote;

    int a = 0,b = 0,c = 0;
    int value = 0;
    for (int l = 0; l < cdst.rows; ++l) {
        for (int i = 0; i < cdst.cols; ++i) {
            for (int j = 0; j < RADIUS; j++) {
                int result = Accumulator[l][i][j];
                if (result > 80) {
                    a = l, b = i, c = j;
                    //value = Accumulator[l][i][j];
                    circle(image, Point(b * 4, a * 4), c * 4, Scalar(255, 0,255), 2, 8);
                    circle(image, Point(b * 4, a * 4), 2, Scalar(255, 0,255), 2, 8);
                }
                if(result > 60)
                    vote.push_back(Point(l,i));
                Accumulator[l][i][j] = 0;
            }
        }
    }

    for(int i = 0; i < vote.size(); ++i){
        int a = vote.at(i).x;
        int b = vote.at(i).y;
        circle(votingSpace, Point(b * 4, a * 4), c * 4, Scalar(128, 128,128), 2, 8);
    }

    GaussianBlur(votingSpace, votingSpace, Size(7,7), 1.5, 1.5);
    imshow("Voting Space", votingSpace);
}