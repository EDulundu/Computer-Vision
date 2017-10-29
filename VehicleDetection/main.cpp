#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <dirent.h>
#include <zconf.h>

#define RE_SIZE  Size(40, 40)
#define OUTPUT_PATH   "/home/emre/Desktop/hw4/cars/"
#define VIDEO_PATH    "/home/emre/Desktop/hw4/videoTraffic.mp4"

#define CAR_POSITIVE_PATH       "/home/emre/Desktop/hw4/DataSet/Cars/positive/"
#define CAR_NEGATIVE_PATH       "/home/emre/Desktop/hw4/DataSet/Cars/negative/"

#define BUS_POSITIVE_PATH       "/home/emre/Desktop/hw4/DataSet/Buses/positive/"
#define BUS_NEGATIVE_PATH       "/home/emre/Desktop/hw4/DataSet/Buses/negative/"

#define TRUCK_POSITIVE_PATH     "/home/emre/Desktop/hw4/DataSet/Trucks/positive/"
#define TRUCK_NEGATIVE_PATH     "/home/emre/Desktop/hw4/DataSet/Trucks/negative/"

#define MINIVAN_POSITIVE_PATH   "/home/emre/Desktop/hw4/DataSet/Minivans/positive/"
#define MINIVAN_NEGATIVE_PATH   "/home/emre/Desktop/hw4/DataSet/Minivans/negative/"

#define CAR_SVM_PATH        "/home/emre/Desktop/hw4/CARdetector.yml"
#define BUS_SVM_PATH        "/home/emre/Desktop/hw4/BUSdetector.yml"
#define TRUCK_SVM_PATH      "/home/emre/Desktop/hw4/TRUCKdetector.yml"
#define MINIVAN_SVM_PATH    "/home/emre/Desktop/hw4/MINIVANdetector.yml"

using namespace std;
using namespace cv;
using namespace cv::ml;

void handmarkVehicle();
void detectVehicle();
void loadDataSet(string path, vector<Mat>& imgList);
void getDetector(const Ptr<SVM>& svm, vector<float>& hog_detector);
void calculateHog(const vector<Mat>& imageList, vector<Mat>& gradientList);
void trainSVM(const vector<Mat>& gradientList, const vector<int>& labelList, string path);
void editRectangle(vector<Rect>& locations, int coef);
void drawLocations(const vector<Rect>& locations, Mat& image, Scalar color);
void countVehicle(const vector<Rect>& locations, Mat& image, int& count);

int main(int argc, char* argv[])
{
    // videonun 10 dakikasi icin program 1 saat acik tutuldu.
    // sonuc olarak yakalasik 55 bin resim elde edildi.
    // handmarkVehicle();

    if (access(CAR_SVM_PATH, 0) != 0 ||
            access(BUS_SVM_PATH, 0) != 0 ||
            access(TRUCK_SVM_PATH, 0) != 0 ||
            access(MINIVAN_SVM_PATH, 0) != 0)
    {
        // label ve gradient listesi
        vector<int> labelList;
        vector<Mat> gradientList;

        // pozitif resimler okunur ve kaydedilir.
        vector<Mat> positiveImageList;
        loadDataSet(CAR_POSITIVE_PATH, positiveImageList);
        labelList.assign(positiveImageList.size(), +1);  // pozitifler icin 1

        // negatif resimler okunur ve kaydedilir.
        vector<Mat> negativeImageList;
        loadDataSet(CAR_NEGATIVE_PATH, negativeImageList);
        labelList.insert(labelList.end(), negativeImageList.size(), -1); // negatifler icin -1

        // belirlenen hog parametreler doğrultusunda pozitif ve negatif resimler
        // icin hogdescriptor hesaplanir ve gradient listesi icine konulur.
        calculateHog(positiveImageList, gradientList);
        calculateHog(negativeImageList, gradientList);

        // gradient listesi svm verilerek svm train edilir.
        trainSVM(gradientList, labelList, CAR_SVM_PATH);

        labelList.clear();
        gradientList.clear();
        positiveImageList.clear();
        negativeImageList.clear();
        /////////////////////////////////////////////////////////////////////////////////////////

        loadDataSet(MINIVAN_POSITIVE_PATH, positiveImageList);
        labelList.assign(positiveImageList.size(), +1);  // pozitifler icin 1

        loadDataSet(MINIVAN_NEGATIVE_PATH, negativeImageList);
        labelList.insert(labelList.end(), negativeImageList.size(), -1); // negatifler icin -1

        calculateHog(positiveImageList, gradientList);
        calculateHog(negativeImageList, gradientList);
        trainSVM(gradientList, labelList, MINIVAN_SVM_PATH);

        labelList.clear();
        gradientList.clear();
        positiveImageList.clear();
        negativeImageList.clear();
        ////////////////////////////////////////////////////////////////////////////////////////

        loadDataSet(TRUCK_POSITIVE_PATH, positiveImageList);
        labelList.assign(positiveImageList.size(), +1);  // pozitifler icin 1

        loadDataSet(TRUCK_NEGATIVE_PATH, negativeImageList);
        labelList.insert(labelList.end(), negativeImageList.size(), -1); // negatifler icin -1

        calculateHog(positiveImageList, gradientList);
        calculateHog(negativeImageList, gradientList);
        trainSVM(gradientList, labelList, TRUCK_SVM_PATH);

        labelList.clear();
        gradientList.clear();
        positiveImageList.clear();
        negativeImageList.clear();
        /////////////////////////////////////////////////////////////////////////////////////////

        loadDataSet(BUS_POSITIVE_PATH, positiveImageList);
        labelList.assign(positiveImageList.size(), +1);  // pozitifler icin 1

        loadDataSet(BUS_NEGATIVE_PATH, negativeImageList);
        labelList.insert(labelList.end(), negativeImageList.size(), -1); // negatifler icin -1

        calculateHog(positiveImageList, gradientList);
        calculateHog(negativeImageList, gradientList);
        trainSVM(gradientList, labelList, BUS_SVM_PATH);

        labelList.clear();
        gradientList.clear();
        positiveImageList.clear();
        negativeImageList.clear();
    }

    detectVehicle();

    return 0;
}

/**
 * background substractor kullanilarak ilk once background
 * daha sonra foreground bulundu. Ardindan dataset olusturuldu.
 */
void handmarkVehicle()
{
    VideoCapture capture(VIDEO_PATH);

    if(!capture.isOpened())
    {
        cerr << "Unable to open video file: " << endl;
        exit(EXIT_FAILURE);
    }

    Mat strel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    //Mat strel2 = getStructuringElement(MORPH_ELLIPSE, Size(50, 50));

    Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();

    int fileNumber = 0;
    Mat original, background;

    char ch = 0;
    while(ch != 27)
    {
        if(!capture.read(original))
        {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

        // resmin sadece sol altina bakiyorum.
        Mat frame = Mat(original.rows, original.cols, CV_8UC3, Scalar(0, 0, 0));
        for (int i = original.rows / 2; i < original.rows; ++i)
            for (int j = 0; j < original.cols / 2; ++j)
                frame.at<Vec3b>(i, j) = original.at<Vec3b>(i, j);

        // background subtractor icin ayarlar
        pMOG2->setNMixtures(3);
        pMOG2->setVarThreshold(40);
        pMOG2->setDetectShadows(false);
        pMOG2->apply(frame, background);

        // resmi tamamlayarak daha guzel dataset olusturmak icin kullanildi.
        //erode(background, background, strel);
        //dilate(background, background, strel2);
        dilate(background, background, strel);
        dilate(background, background, strel);
        erode(background, background, strel);

        original.copyTo(frame);
        for (int i = 0; i < background.rows; ++i)
            for (int j = 0; j < background.cols; ++j)
                if (background.at<uchar>(i, j) == 0)
                    frame.at<Vec3b>(i, j) = 0;

        // contourlari bulmaya baslar buradan itibaren
        Mat gray, thresh_output;

        cvtColor(frame, gray, CV_BGR2GRAY);
        blur(gray, gray, Size(3, 3));
        threshold(gray, thresh_output, 40, 255, CV_THRESH_BINARY);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        findContours(thresh_output, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        vector<vector<Point>> contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());

        // belli bir alanin altinda kalan dikdortgenler alinmaz.
        // bunu yapmamin sebebi cop dosyalari azaltmak.
        for (int i = 0; i < contours.size(); i++)
        {
            approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect(Mat(contours_poly[i]));
            if (boundRect[i].area() > 7500)
            {
                rectangle( original, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0 );
                std::string out = OUTPUT_PATH + std::to_string(fileNumber) + ".png";
                //imwrite(out, original(boundRect[i]));
                ++fileNumber;
            }
        }

        imshow("Foreground", original);

        ch = waitKey(1);
    }

    capture.release();
}

/**
 * Dataset olarak cikartilmis resimler listelere eklenir.
 * @param path resimlerin klasor pathi.
 * @param imgList output olarak alinacak resimler listesi.
 */
void loadDataSet(string path, vector<Mat>& imgList)
{
    DIR *dirP = opendir(path.c_str());
    struct dirent *direntP;
    std::string file = path;

    if(dirP == NULL)
    {
        cerr << "Directory is not open!!!" << endl;
        exit(1);
    }

    while((direntP = readdir(dirP)) != NULL)
    {
        std::string imgName = direntP->d_name;
        if(imgName != "." && imgName != "..")
        {
            Mat frame = imread(file + imgName);
            resize(frame, frame, RE_SIZE);
            imgList.push_back(frame);
        }
    }

    closedir(dirP);
}

/**
 * Resimler tek tek griye çevirilerek hog'a verilir. Daha sonra hog bize resmin
 * 8 yonde gradientini verir. Ve sonuc olarak bize bir descriptor vermis olur.
 * Bunuda gradient listesine ekleriz.
 * @param imageList verilen resim listesi
 * @param gradientList output olarak alinacak resmin gradienti.
 */
void calculateHog(const vector<Mat>& imageList, vector<Mat>& gradientList)
{
    // location ve descriptor bunlar zaten kendi kullanacagi parametreler.
    // implementationda yazmaktadir.
    vector<Point> location;
    vector<float> descriptors;
    Size winStride = Size(8, 8);
    Size padding = Size(0, 0);
    Size winSize = RE_SIZE;

    HOGDescriptor HOG;
    HOG.winSize = winSize;

    for (int i = 0; i < imageList.size(); ++i)
    {
        Mat gray, frame = imageList[i];
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        HOG.compute(gray, descriptors, winStride, padding, location);
        Mat img = Mat(descriptors).clone();
        gradientList.push_back(img);
    }
}

/**
 * Resmin gradiyentine ve labellarina gore svmi belirlenen parametrelere gore egitilir.
 * @param gradientList resmin gradienti alinmis vektoru.
 * @param labelList resim icin belirlenmis label listesi.
 */
void trainSVM(const vector<Mat>& gradientList, const vector<int>& labelList, string path)
{
    // label listesi ile gradient listesinin satirlari ayni olmak zorunda oldugu icin
    // gradientin transpozesini alarak ROW_SAMPLE sartini yerine getiririz.
    Mat temp(1, gradientList.at(0).cols, CV_32FC1);
    Mat trainingData = Mat(gradientList.size(), gradientList.at(0).rows, CV_32FC1);
    for (int i = 0; i < gradientList.size(); ++i)
    {
        transpose(gradientList[i], temp);
        temp.copyTo(trainingData.row(i));
    }

    /* Default values to train SVM */
    SVM::Params params;
    params.coef0 = 0.0;
    params.degree = 3;
    params.termCrit = TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3);
    params.gamma = 0;
    params.kernelType = SVM::LINEAR;
    params.nu = 0.5;
    params.p = 0.1;
    params.C = 0.01;
    params.svmType = SVM::EPS_SVR;

    Ptr<SVM> svm = StatModel::train<SVM>(trainingData, ROW_SAMPLE, Mat(labelList), params);
    svm->save(path);
}

/**
 * Bu fonksiyonu birden fazla vehicle tipi oldugu icin yazildi.
 * Her bir tip icin ayri bir svm olacagindan her bir svm icin kullanildi.
 * @param svm support vector machine.
 * @param hog_detector svm icerisinde hesaplanan agirliklari hog'a set etmek icin gonderilir.
 */
void getDetector(const Ptr<SVM>& svm, vector<float>& hog_detector)
{
    // Hogu calistirabilmek icin SVM'in olusturdugu
    // agirliklar support vector alinir. Daha sonra bu veri
    // set edilir.
    // http://stackoverflow.com/questions/15339657/training-custom-svm-to-use-with-hogdescriptor-in-opencv
    Mat sv = svm->getSupportVectors();
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(float));
    hog_detector[sv.cols] = (float) -rho;
}

/**
 * Kucultulen resim uzerindeki dikdortgenin degerlerinin eski haline cevrilmesi
 * @param locations rectangle
 * @param coef katsayi
 */
void editRectangle(vector<Rect>& locations, int coef)
{
    for (int i = 0; i < locations.size(); ++i)
    {
        locations[i].x = locations[i].x * coef;
        locations[i].y = locations[i].y * coef;
        locations[i].width = locations[i].width * coef;
        locations[i].height = locations[i].height * coef;
    }
}

/**
 * Verilen location dikdortgenlerini resim uzerini cizer.
 * @param locations rectangles
 * @param image cizilecek resim
 * @param color renk
 */
void drawLocations(const vector<Rect>& locations, Mat& image, Scalar color)
{
    for(int i = 0; i < locations.size(); ++i)
        rectangle(image, locations[i], color, 2, 8, 0);
}

/**
 * Vehicle detection HOG + SVM
 * Svm deki hesaplana agirlik alinip hog descriptor'a verilir.
 * Daha sonra multiscale yapilarak farklı boyutta olabilecek araclar tespit edilir.
 */
void detectVehicle()
{
    Ptr<SVM> svmCar = StatModel::load<SVM>(CAR_SVM_PATH);
    vector<float> hogCarDetector;
    getDetector(svmCar, hogCarDetector);

    HOGDescriptor hogCar;
    hogCar.winSize = RE_SIZE;
    hogCar.setSVMDetector(hogCarDetector);

    //////////////////////////////////////

    Ptr<SVM> svmBus = StatModel::load<SVM>(BUS_SVM_PATH);
    vector<float> hogBusDetector;
    getDetector(svmBus, hogBusDetector);

    HOGDescriptor hogBus;
    hogBus.winSize = RE_SIZE;
    hogBus.setSVMDetector(hogBusDetector);

    //////////////////////////////////////

    Ptr<SVM> svmTruck = StatModel::load<SVM>(TRUCK_SVM_PATH);
    vector<float> hogTruckDetector;
    getDetector(svmTruck, hogTruckDetector);

    HOGDescriptor hogTruck;
    hogTruck.winSize = RE_SIZE;
    hogTruck.setSVMDetector(hogTruckDetector);

    //////////////////////////////////////

    Ptr<SVM> svmMinivan = StatModel::load<SVM>(MINIVAN_SVM_PATH);
    vector<float> hogMinivanDetector;
    getDetector(svmMinivan, hogMinivanDetector);

    HOGDescriptor hogMinivan;
    hogMinivan.winSize = RE_SIZE;
    hogMinivan.setSVMDetector(hogMinivanDetector);

    //////////////////////////////////////

    VideoCapture capture(VIDEO_PATH);
    if(!capture.isOpened())
    {
        cerr << "Unable to open video file: " << endl;
        exit(EXIT_FAILURE);
    }

    vector<Rect> locations;
    vector<Rect> locations2;
    vector<Rect> locations3;
    vector<Rect> locations4;
    char ch = 0;
    int carCount = 0;
    int busCount = 0;
    int truckCount = 0;
    int minivanCount = 0;
    while (ch != 'q' && ch != 27)
    {
        Mat original, frame, image, detected;

        capture.read(original);

        resize(original, frame, Size(1280, 720), 0, 0, CV_INTER_LINEAR);

        frame.copyTo(detected);

        for (int i = 0; i < frame.rows; ++i)
            for (int j = 0; j < frame.cols; ++j)
                if (j > frame.cols / 2)
                    frame.at<Vec3b>(i, j) = 0;


        // hizlandirmak icin resmi kuculterek gonderdim.
        resize(frame, image, Size(frame.cols * 0.5, frame.rows * 0.5), 0.5, 0.5, CV_INTER_LINEAR);

        locations.clear();
        locations2.clear();
        locations3.clear();
        locations4.clear();

        hogCar.detectMultiScale(image, locations, 0,  Size(), Size(), 1.10);
        hogBus.detectMultiScale(image, locations2, 0,  Size(), Size(), 1.10);
        hogTruck.detectMultiScale(image, locations3, 0, Size(), Size(), 1.10);
        hogMinivan.detectMultiScale(image, locations4, 0,  Size(), Size(), 1.10);

        line(detected, Point(430, 450), Point(600, 450), Scalar(0, 0, 255), 3);

        // kucultulen resmin eski halindeki kordinatlarina cevirilir.
        editRectangle(locations, 2);
        editRectangle(locations2, 2);
        editRectangle(locations3, 2);
        editRectangle(locations4, 2);

        // dikdortgenler resmin uzerine cizilir.
        drawLocations(locations, detected, Scalar(0, 255, 0));
        drawLocations(locations2, detected, Scalar(255, 0, 0));
        drawLocations(locations3, detected, Scalar(0, 0, 255));
        drawLocations(locations4, detected, Scalar(0, 0, 0));

        countVehicle(locations, detected, carCount);
        countVehicle(locations2, detected, busCount);
        countVehicle(locations3, detected, truckCount);
        countVehicle(locations4, detected, minivanCount);

        putText(detected, "Lane 3-Number Of Cars: " + to_string(carCount), Point(10, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
        putText(detected, "Lane 3-Number Of Buses: " + to_string(busCount), Point(10, 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 0, 0));
        putText(detected, "Lane 3-Number Of Trucks: " + to_string(truckCount), Point(10, 70), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 255, 0));
        putText(detected, "Lane 3-Number Of Minivans: " + to_string(minivanCount), Point(10, 90), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));

        imshow("Vehicle Detection", detected);

        ch = (char) waitKey(10);
    }
}

void countVehicle(const vector<Rect>& locations, Mat& image, int& count)
{
    for (int i = 0; i < locations.size(); ++i)
    {
        int x = (locations[i].width / 2) + locations[i].x;
        int y = (locations[i].height / 2) + locations[i].y;
        if( (x >= 415 && x <= 600) && (y > 448 && y < 452) )
        {
            rectangle(image, locations[i], Scalar(255, 0, 0), 2, 8, 0);
            ++count;
        }
    }
}
