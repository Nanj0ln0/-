#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat src = imread("D:/OpenCV/picture zone/pill_002.png");//coins_001.jpg,pill_002.png
	if (!src.data)
	{
		printf("ERROR");
	}
	imshow("input",src);

	Mat gray, binary, shifted;
	pyrMeanShiftFiltering(src,shifted,21,51);  //图像模糊，突出主体
	//imshow("shifted",shifted);

	cvtColor(shifted,gray,COLOR_BGR2GRAY);
	threshold(gray,binary,0,255,THRESH_BINARY|THRESH_OTSU);
	//imshow("binary",binary);


	//距离变换
	Mat dist;
	distanceTransform(binary,dist,DistanceTypes::DIST_L2,3,CV_32F);
	normalize(dist,dist,0,1,NORM_MINMAX);
	threshold(dist,dist,0.4,1,THRESH_BINARY);
	//imshow("dist", dist);

	//marks,寻找轮廓
	Mat dist_m;
	dist.convertTo(dist_m,CV_8U);
	vector<vector<Point>>contours;
	findContours(dist_m,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE,Point(0,0));

	//creak marks
	Mat markers = Mat::zeros(src.size(),CV_32SC1);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers,contours,static_cast<int>(i),Scalar::all(static_cast<int>(i) + 1),-1 );
	}
	circle(markers,Point(5,5),3,Scalar(255),-1);

	//形态学操作 - 彩色图像，目的是去掉干扰，让结果更好
	Mat k = getStructuringElement(MORPH_RECT,Size(3,3),Point(0,0));
	morphologyEx(src,src,MORPH_ERODE,k);


	//分水岭
	watershed(src,markers);
	Mat mark = Mat::zeros(markers.size(), CV_8UC1);
	markers.convertTo(mark, CV_8UC1);
	bitwise_not(mark, mark);
	imshow("分水岭变换", mark);

	//创建随机颜色
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
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

			if (index > 0 && index <= static_cast<int>(contours.size())) {
				dst.at<Vec3b>(row, col) = colors[index - 1];
			}
			else {
				dst.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
			}
		}
	}
	imshow("dst",dst);
	printf("number of objects:%d",contours.size());

	waitKey(0);
	return 0;
}