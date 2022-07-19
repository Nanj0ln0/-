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
		cvtColor(frame,hsv,COLOR_BGR2HSV);  //  ��BGR��ɫ�ʿռ�ת����HSV�ռ�
		inRange(hsv, Scalar(35, 43, 46), Scalar(155,255,255),mask);   //�������Ԫ���Ƿ���������������Ԫ��ֵ֮��,�������һ����ֵ����ͼ��

		//��̬ѧ����
		Mat k = getStructuringElement(MORPH_RECT,Size(3,3),Point(-1,-1));
		morphologyEx(mask,mask,MORPH_CLOSE,k);
		erode(mask,mask,k);
		GaussianBlur(mask,mask,Size(3,3),0,0);

		Mat result = replace_and_blend(frame, mask);   //�滻�ͻ��
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

//�滻�ͻ��
Mat replace_and_blend(Mat& frame, Mat& mask) {
	Mat result = Mat::zeros(frame.size(),frame.type());;
	int h = frame.rows;
	int w = frame.cols;
	int dims = frame.channels();

	//�滻�ͻ��
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
			if (m == 255) //����
			{
				*targetrow++ = *bgrow++; //������ֵ  ��Ϊ��3��ͨ������������
				*targetrow++ = *bgrow++;
				*targetrow++ = *bgrow++;
				current += 3;    //ǰ��ָ��ͬʱ�ƶ�����ͬλ��
			}
			else if (m == 0) //ǰ��
			{
				*targetrow++ = *current++;//ǰ����ֵ  ��Ϊ��3��ͨ������������
				*targetrow++ = *current++;
				*targetrow++ = *current++;
				bgrow += 3; //����ָ��ͬʱ�ƶ�����ͬλ��
			}
			else  //���紦���
			{
				b1 = *bgrow++;
				g1 = *bgrow++;
				r1 = *bgrow++;

				b2 = *current++;
				g2 = *current++;
				r2 = *current++;

				//���Ȩ��
				weight = m / 255.0; //Ȩ��

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


	
	