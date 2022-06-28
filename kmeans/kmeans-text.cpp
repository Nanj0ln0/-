#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat src(500,500,CV_8UC3);
	RNG rng(12345);

	Scalar colorTab[] = {
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 0, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255)
	};

	int numCluster = rng.uniform(2,5);
	printf("number of clusters:%d\n",numCluster);

	int samplerCount = rng.uniform(5, 1000);
	Mat points(samplerCount, 1, CV_32FC2);
	Mat labes;
	Mat center;

	//生成随机数
	for (int k = 0; k < numCluster; k++)
	{
		Point center;
		center.x = rng.uniform(0,src.cols);
		center.y = rng.uniform(0,src.rows);
		Mat pointChunk = points.rowRange(k*samplerCount/numCluster , k==numCluster-1?samplerCount:(k+1)*samplerCount/numCluster);
		rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(src.cols * 0.05, src.rows * 0.05));
	}
	randShuffle(points,1,&rng);

	//使用Kmeans
	kmeans(points,numCluster,labes,TermCriteria(TermCriteria::EPS + TermCriteria::COUNT,10,0.1),3,KMEANS_PP_CENTERS	,center);


	//分类显示
	src = Scalar::all(255);
	for (int i = 0; i < samplerCount; i++)
	{
		int index = labes.at<int>(i);
		Point p = points.at<Point2f>(i);
		circle(src, p, 2, colorTab[index], -1, 8);

	}

	// 每个聚类的中心来绘制圆
	for (int i = 0; i < center.rows; i++)
	{
		int x = center.at<float>(i, 0);
		int y = center.at<float>(i, 1);
		printf("c.x= %d, c.y=%d", x, y);
		circle(src, Point(x, y), 40, colorTab[i], 1, LINE_AA);
	}


	imshow("kmeans",src);

	waitKey(0);
	return 0;

}