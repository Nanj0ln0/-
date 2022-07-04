#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat watershedCluster(Mat &image,int &numSegments);
void creatDisplaySegments(Mat &segments,int numsegments,Mat &image);
int main() {
	Mat src = imread("D:/OpenCV/picture zone/toux.jpg");
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}
	imshow("input", src); 

	int numSegments;
	Mat markers = watershedCluster(src,numSegments);
	creatDisplaySegments(markers,numSegments,src);



	waitKey(0);
	return 0;
}

//分水岭
Mat watershedCluster(Mat &image, int& numCump) {
	//二值化
	Mat gray, binary, shifted;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	
	//形态学与距离变换
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(0, 0));
	morphologyEx(binary, binary, MORPH_OPEN, k);
	Mat dist;
	distanceTransform(binary, dist, DistanceTypes::DIST_L2, 3, CV_32F);
	normalize(dist, dist, 0, 1, NORM_MINMAX);

	//开始生成标记
	threshold(dist, dist, 0.1, 1, THRESH_BINARY);
	normalize(dist, dist, 0, 255, NORM_MINMAX);
	dist.convertTo(dist, CV_8UC1);

	//标记开始
	vector<vector<Point>>contours;
	vector<Vec4i>hireachy;
	findContours(dist, contours,hireachy ,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.empty())
	{
		return Mat();
	}

	//creak marks
	Mat markers = Mat::zeros(dist.size(), CV_32S);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, i, Scalar(i+1),-1,8, hireachy,INT_MAX);
	}
	circle(markers, Point(5, 5), 3, Scalar(255), -1);

	//分水岭
	watershed(image, markers);
	numCump = contours.size();
	return markers;
}

void creatDisplaySegments(Mat &markers, int numsegments, Mat& image) {

	//创建随机颜色
	vector<Vec3b> colors;
	for (size_t i = 0; i < numsegments; i++) {
		int r = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int b = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	//上色
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	for (int row = 0; row < markers.rows; row++)
	{
		for (int col = 0; col < markers.cols; col++)
		{
			int index = markers.at<int>(row, col);

			if (index > 0 && index <= static_cast<int>(numsegments)) {
				dst.at<Vec3b>(row, col) = colors[index - 1];
			}
			else {
				dst.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
			}
		}
	}
	imshow("分水岭图像分割",dst);
}
