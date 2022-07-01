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

	

	//��ʼ������
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
	

	//ͼ��RGBתΪ��������
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

	
	//EM Cluster train  ѵ��
	Ptr<EM>em_model = EM::create();
	em_model->setClustersNumber(numCluster);   //�������
	em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);//Э�����������
	em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 0.1));  //ֹͣ����
	em_model->trainEM(points, noArray(), labels, noArray());
	//��һ������ʾ��������ݼ��ϣ�����һά���߶�ά���ݣ�������Mat���ͣ�
	//�ڶ�������ѡ����һ�������������ÿ����������Ȼ����ֵ���������Ҫ��ΪnoArray()
	//��������labels��ʾ����֮��������ݵ�����յķ�����������һ��INT���͵�Mat�������ͺͳ�����ԭͼ��һ��
	//���ĸ���//��ѡ����һ�������������ÿ�����Ա����ĺ�����ʣ��������Ҫ��ΪnoArray()

	
	//�����ݵ�ת��Ϊͼ����ʾ
	Mat sample(dims,1,CV_64FC1);
	int r = 0, b = 0, g = 0;
	Mat result = Mat::zeros(src.size(), CV_8UC3);
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			index = row * width + col;

			//��ͨ��ʽ
			/*
			int label = labels.at<int>(index, 0);	//��ÿ�����ص��Ӧ�ı�ǩ����Ӧ����ɫ������ͼ��
			result.at<Vec3b>(row, col)[0] = colorTab[label][0];
			result.at<Vec3b>(row, col)[1] = colorTab[label][1];
			result.at<Vec3b>(row, col)[2] = colorTab[label][2];
			*/

			//Ԥ�Է�ʽ
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