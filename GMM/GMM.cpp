#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
	Mat src(500, 500, CV_8UC3);
	RNG rng(12345);

	Scalar colorTab[] = {
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 0, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255)
	};

	int numCluster = rng.uniform(2, 5);
	printf("number of clusters:%d\n", numCluster);

	int samplerCount = rng.uniform(5, 1000);
	Mat points(samplerCount, 2, CV_32FC1);
	Mat labes;
	Mat center;

	//生成随机数
	for (int k = 0; k < numCluster; k++)
	{
		Point center;
		center.x = rng.uniform(0, src.cols);
		center.y = rng.uniform(0, src.rows);
		Mat pointChunk = points.rowRange(k * samplerCount / numCluster, 
			k == numCluster - 1 ? samplerCount : (k + 1) * samplerCount / numCluster);
		rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(src.cols * 0.05, src.rows * 0.05));
	}
	randShuffle(points, 1, &rng);

	//使用GMM
	Ptr<EM>em_model = EM::create();
	em_model->setClustersNumber(numCluster);
	em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
	em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,100,0.1));
	em_model->trainEM(points, noArray(), labes,noArray());
	 
	//对每个图像像素进行分类
	Mat sample(1,2,CV_32FC1);
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			sample.at<float>(0) = (float)col;
			sample.at<float>(1) = (float)row;
			int response = cvRound(em_model->predict2(sample,noArray())[1]);  //预言
			
			Scalar c = colorTab[response]; //颜色
			circle(src, Point(col,row),1,c*0.75,-1);
		}
	}

	//画出数据点
	for (int i = 0; i < samplerCount; i++)
	{

		Point p(cvRound(points.at<float>(i, 0)), cvRound(points.at<float>(i, 1)));
		circle(src,p,1, colorTab[labes.at<int>(i)],-1);
	}

	imshow("GMM DEMO",src);

	waitKey(0);
	return 0;

}