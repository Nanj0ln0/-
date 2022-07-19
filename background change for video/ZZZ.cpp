#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat mat_to_samples(Mat& image);  //kmeans 第一步，将一张图转换成样本数据

int main() {
	

	Mat src = imread("D:/OpenCV/picture zone/toux.jpg");
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);

	
	//组装数据
	Mat points = mat_to_samples(src);

	
	//运行kmeans
	int clusterCount = 4;
	Mat labels;
	Mat centers;
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1); //终止条件
	kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, centers);
	// kmeans参数  样本数据，分几类（这边是4）， 标签类  ，终止条件，尝试多少次，通过中心化算法，中心点

	//去背景 + 遮罩生成
	Mat mask = Mat::zeros(src.size(),CV_8UC1); //遮罩
	int index = src.rows * 2 + 2;   //二行二列的坐标
	int cindex = labels.at<int>(index);   ///取出二行二列的标签

	int width = src.cols;				//列
	int height = src.rows;			//行 
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			index = row * width + col;
			int label = labels.at<int>(index);
				if (label == cindex)
				{ //背景  (遮罩层)
					mask.at<uchar>(row, col) = 0;  
				}
				else
				{//前景   (去背景)
					mask.at<uchar>(row, col) = 255;
				}

		}

	}
	//imshow("mask",mask);

	//腐蚀(对背景) + 高斯模糊
	Mat k = getStructuringElement(MORPH_RECT,Size(3,3),Point(-1,-1));
	erode(mask, mask,k);
		//imshow("erode",mask);
	GaussianBlur(mask,mask,Size(3,3),0,0);
		//imshow("GaussianBlur", mask);

	//通道混合
		//随机颜色
	Vec3b color;
	color[0] = theRNG().uniform(0, 255);
	color[1] = theRNG().uniform(0, 255);
	color[2] = theRNG().uniform(0, 255);
	
	Mat result(src.size(),src.type()); //结果
	double w = 0.0;
	int b = 0, g = 0, r = 0;
	int b1 = 0, g1 = 0, r1 = 0;
	int b2 = 0, g2 = 0, r2 = 0;
	
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			int m = mask.at<uchar>(row, col);
			if ( m == 255) //判断前景
			{
				result.at<Vec3b>(row, col) = src.at<Vec3b>(row, col);  //前景赋值
			}
			else if(m == 0) //判断背景
			{
				result.at<Vec3b>(row, col) = color;		//背景改颜色
			}
			else
			{     //前景和背景重合的位置 ，按权重混合
				w = m / 255.0; //权重
				b1 = src.at<Vec3b>(row, col)[0];
				g1 = src.at<Vec3b>(row, col)[1];
				r1 = src.at<Vec3b>(row, col)[2];

				//背景
				b2 = color[0];
				g2 = color[1];
				r2 = color[2];

				//混合
				int b = b1 * w + b2 * (1.0 - w);
				int g = g1 * w + g2 * (1.0 - w);
				int r = r1 * w + r2 * (1.0 - w);
			
				result.at<Vec3b>(row, col)[0] = b;
				result.at<Vec3b>(row, col)[1] = g;
				result.at<Vec3b>(row, col)[2] = r;	
			}

		}
	}

	imshow("背景替换",result);



		waitKey(0);
		return 0;

}

Mat mat_to_samples(Mat& image) {

	int width = image.cols;				//列
	int height = image.rows;			//行
	int sampleCount = width * height;	//面积
	int dims = image.channels();		//颜色通道
	int clusterCount = 4;
	Mat points(sampleCount, dims, CV_32F, Scalar(10));  //样本数据的返回结果
	
	
	int index = 0;
	for (int row = 0; row < height; row++)	//遍历每一个像素点，将每一个像素点变成一个对象
	{
		for (int col = 0; col < width; col++)
		{
			index = row * width + col;    //图片在计算机里是一个长数组
			Vec3b bgr = image.at<Vec3b>(row, col); //取出每一个像素的三通道值

			points.at<float>(index, 0) = static_cast<int>(bgr[0]); //B
			points.at<float>(index, 1) = static_cast<int>(bgr[1]); //G 
			points.at<float>(index, 2) = static_cast<int>(bgr[2]); //R
		}
	}

	return points;
}