#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


Mat replace_and_blend(Mat &frame,Mat &mask);
Mat background = imread("D:/OpenCV/picture zone/background.jpg");  
int main() {

	
	VideoCapture capture;
	capture.open("D:/OpenCV/picture zone/01.mp4");
	if (!capture.isOpened())
	{
		printf("video is not loading.....");
		return -1;
	}
	const char* title = "input video";
	const char* resultWin = "video matting";
	namedWindow(title,CV_WINDOW_AUTOSIZE);
	Mat frame;
	Mat hsv, mask;


	while (capture.read(frame))
	{
		cvtColor(frame,hsv,COLOR_BGR2HSV);  //  从BGR的色彩空间转换到HSV空间
		inRange(hsv, Scalar(35, 43, 46), Scalar(155,255,255),mask);   //检查数组元素是否在另外两个数组元素值之间,输出的是一个二值化的图像

		//形态学操作
		Mat k = getStructuringElement(MORPH_RECT,Size(3,3),Point(-1,-1));
		morphologyEx(mask,mask,MORPH_CLOSE,k);
		erode(mask,mask,k);
		GaussianBlur(mask,mask,Size(3,3),0,0);

		Mat result = replace_and_blend(frame, mask);   //替换和混合
		char c = waitKey(1);
		if (c == 27)
		{
			break;
		}

		imshow(resultWin,result);
		imshow(title,frame);
		
	}


	waitKey(0);
	return 0;
}

//替换和混合
Mat replace_and_blend(Mat& frame, Mat& mask) {
	Mat result = Mat::zeros(frame.size(),frame.type());;
	int h = frame.rows;
	int w = frame.cols;
	int dims = frame.channels();

	//替换和混合
	int	m = 0;
	double weight = 0;
	int r = 0, g = 0, b = 0;
	int r1 = 0, g1 = 0, b1 = 0;
	int r2 = 0, g2 = 0, b2 = 0;

	for (int row = 0; row < h; row++)
	{
		uchar* current = frame.ptr<uchar>(row);
		uchar* bgrow = background.ptr<uchar>(row);
		uchar* maskrow = mask.ptr<uchar>(row);
		uchar* targetrow = result.ptr<uchar>(row); 
		for (int col = 0; col < w; col++)
		{
			m = *maskrow++;
			if (m == 255) //背景
			{
				*targetrow++ = *bgrow++; //背景赋值  因为是3个通道所有来三次
				*targetrow++ = *bgrow++;
				*targetrow++ = *bgrow++;
				current += 3;    //前景指针同时移动到相同位置
			}
			else if (m == 0) //前景
			{
				*targetrow++ = *current++;//前景赋值  因为是3个通道所有来三次
				*targetrow++ = *current++;
				*targetrow++ = *current++;
				bgrow += 3; //背景指针同时移动到相同位置
			}
			else  //交界处混合
			{
				b1 = *bgrow++;
				g1 = *bgrow++;
				r1 = *bgrow++;

				b2 = *current++;
				g2 = *current++;
				r2 = *current++;

				//混合权重
				weight = m / 255.0; //权重

				int b = b1 * weight + b2 * (1.0 - weight);
				int g = g1 * weight + g2 * (1.0 - weight);
				int r = r1 * weight + r2 * (1.0 - weight);

				*targetrow++ = b;
				*targetrow++ = g;
				*targetrow++ = r;

			}
			
		}


	}
	return result;
}


	
	