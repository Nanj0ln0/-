#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {
	Mat src = imread("D:/OpenCV/picture zone/cvtest.png");
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);

	

	//初始化定义
	int numCluster = 3;
	Scalar colorTab[] = {
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 0, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255)
	};

	int width = src.cols;
	int height = src.rows;
	int dims = src.channels();
	int nsample = width * height;
	Mat points(nsample,dims,CV_64FC1);
	Mat labels;
	

	//图像RGB转为样本数据
	int index = 0;
	for (int  row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			index = row * width + col;
			Vec3b bgr = src.at<Vec3b>(row, col);
			points.at<double>(index, 0) = static_cast<int>(bgr[0]);
			points.at<double>(index, 1) = static_cast<int>(bgr[1]);
			points.at<double>(index, 2) = static_cast<int>(bgr[2]);

		}
	}

	
	//EM Cluster train  训练
	Ptr<EM>em_model = EM::create();
	em_model->setClustersNumber(numCluster);   //分类个数
	em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);//协方差矩阵类型
	em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 0.1));  //停止条件
	em_model->trainEM(points, noArray(), labels, noArray());
	//第一个：表示输入的数据集合，可以一维或者多维数据，类型是Mat类型，
	//第二个：可选项，输出一个矩阵，里面包含每个样本的似然对数值，如果不需要则为noArray()
	//第三个：labels表示计算之后各个数据点的最终的分类索引，是一个INT类型的Mat对象，类型和长宽与原图像一致
	//第四个：//可选项，输出一个矩阵，里面包含每个隐性变量的后验概率，如果不需要则为noArray()

	
	//将数据点转换为图像并显示
	Mat sample(dims,1,CV_64FC1);
	int r = 0, b = 0, g = 0;
	Mat result = Mat::zeros(src.size(), CV_8UC3);
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			index = row * width + col;

			//普通方式
			/*
			int label = labels.at<int>(index, 0);	//把每个像素点对应的标签所对应的颜色赋给新图像
			result.at<Vec3b>(row, col)[0] = colorTab[label][0];
			result.at<Vec3b>(row, col)[1] = colorTab[label][1];
			result.at<Vec3b>(row, col)[2] = colorTab[label][2];
			*/

			//预言方式
			b = src.at<Vec3b>(row, col)[0];
			g = src.at<Vec3b>(row, col)[1];
			r = src.at<Vec3b>(row, col)[2];
			
			sample.at<double>(0) = b; 
			sample.at<double>(1) = g; 
			sample.at<double>(2) = r; 

			int response = cvRound(em_model->predict2(sample,noArray())[1]);
			result.at<Vec3b>(row, col)[0] = colorTab[response][0];
			result.at<Vec3b>(row, col)[1] = colorTab[response][1];
			result.at<Vec3b>(row, col)[2] = colorTab[response][2];


		}
	}
	imshow("output", result);
	
	waitKey(0);
	return 0;

}