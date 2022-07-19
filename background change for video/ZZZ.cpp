#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat mat_to_samples(Mat& image);  //kmeans ��һ������һ��ͼת������������

int main() {
	

	Mat src = imread("D:/OpenCV/picture zone/toux.jpg");
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);

	
	//��װ����
	Mat points = mat_to_samples(src);

	
	//����kmeans
	int clusterCount = 4;
	Mat labels;
	Mat centers;
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1); //��ֹ����
	kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, centers);
	// kmeans����  �������ݣ��ּ��ࣨ�����4���� ��ǩ��  ����ֹ���������Զ��ٴΣ�ͨ�����Ļ��㷨�����ĵ�

	//ȥ���� + ��������
	Mat mask = Mat::zeros(src.size(),CV_8UC1); //����
	int index = src.rows * 2 + 2;   //���ж��е�����
	int cindex = labels.at<int>(index);   ///ȡ�����ж��еı�ǩ

	int width = src.cols;				//��
	int height = src.rows;			//�� 
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			index = row * width + col;
			int label = labels.at<int>(index);
				if (label == cindex)
				{ //����  (���ֲ�)
					mask.at<uchar>(row, col) = 0;  
				}
				else
				{//ǰ��   (ȥ����)
					mask.at<uchar>(row, col) = 255;
				}

		}

	}
	//imshow("mask",mask);

	//��ʴ(�Ա���) + ��˹ģ��
	Mat k = getStructuringElement(MORPH_RECT,Size(3,3),Point(-1,-1));
	erode(mask, mask,k);
		//imshow("erode",mask);
	GaussianBlur(mask,mask,Size(3,3),0,0);
		//imshow("GaussianBlur", mask);

	//ͨ�����
		//�����ɫ
	Vec3b color;
	color[0] = theRNG().uniform(0, 255);
	color[1] = theRNG().uniform(0, 255);
	color[2] = theRNG().uniform(0, 255);
	
	Mat result(src.size(),src.type()); //���
	double w = 0.0;
	int b = 0, g = 0, r = 0;
	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;
	
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int m = mask.at<uchar>(row, col);
			if ( m == 255) //�ж�ǰ��
			{
				result.at<Vec3b>(row, col) = src.at<Vec3b>(row, col);  //ǰ����ֵ
			}
			else if(m == 0) //�жϱ���
			{
				result.at<Vec3b>(row, col) = color;		//��������ɫ
			}
			else
			{     //ǰ���ͱ����غϵ�λ�� ����Ȩ�ػ��
				w = m / 255.0; //Ȩ��
				b1 = src.at<Vec3b>(row, col)[0];
				g1 = src.at<Vec3b>(row, col)[1];
				r1 = src.at<Vec3b>(row, col)[2];

				//����
				b2 = color[0];
				g2 = color[1];
				r2 = color[2];

				//���
				int b = b1 * w + b2 * (1.0 - w);
				int g = g1 * w + g2 * (1.0 - w);
				int r = r1 * w + r2 * (1.0 - w);
			
				result.at<Vec3b>(row, col)[0] = b;
				result.at<Vec3b>(row, col)[1] = g;
				result.at<Vec3b>(row, col)[2] = r;	
			}

		}
	}

	imshow("�����滻",result);



		waitKey(0);
		return 0;

}

Mat mat_to_samples(Mat& image) {

	int width = image.cols;				//��
	int height = image.rows;			//��
	int sampleCount = width * height;	//���
	int dims = image.channels();		//��ɫͨ��
	int clusterCount = 4;
	Mat points(sampleCount, dims, CV_32F, Scalar(10));  //�������ݵķ��ؽ��
	
	
	int index = 0;
	for (int row = 0; row < height; row++)	//����ÿһ�����ص㣬��ÿһ�����ص���һ������
	{
		for (int col = 0; col < width; col++)
		{
			index = row * width + col;    //ͼƬ�ڼ��������һ��������
			Vec3b bgr = image.at<Vec3b>(row, col); //ȡ��ÿһ�����ص���ͨ��ֵ

			points.at<float>(index, 0) = static_cast<int>(bgr[0]); //B
			points.at<float>(index, 1) = static_cast<int>(bgr[1]); //G 
			points.at<float>(index, 2) = static_cast<int>(bgr[2]); //R
		}
	}

	return points;
}