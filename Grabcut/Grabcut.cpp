#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;
 
Rect rect;  //���ζ���
Mat src,image;
Mat mask, bgModel, fgmodel;//mask���򣬱�����ǰ��
int numRun = 0;
bool init = false;  //�Ƿ��һ��ִ�еı�־λ

void runGrabCut();
void onmouse(int event, int x, int y, int flags,void*param);//��������ѡ
void showImage();//��ʾ����ѡ����
void setRoiMask();//����ѡ������ΪROI����

int main() {
	src = imread("D:/OpenCV/picture zone/canjian.jpg",1);
	if (!src.data) {
		printf("ERROR");

		return -1;
	}

	mask.create(src.size(),CV_8UC1);  //mask����
	mask.setTo(Scalar::all(GC_BGD));  //��ʼ��mask��ȫ������Ϊ����

	namedWindow("input",CV_WINDOW_AUTOSIZE);
	setMouseCallback("input",onmouse,0);
	imshow("input", src);

	while (true)
	{
		char c = (char)waitKey(0);
		if (c=='n')
		{
			runGrabCut();
			numRun++;
			showImage();
			printf("����������%d\n",numRun);
		}
		if ((int)c == 27)
		{
			break;
		}

	}


	waitKey(0);
	return 0;
}

void setRoiMask() {
	//GC_BGD = 0	����
	//GC_FGD = 1	ǰ��
	//GC_PR_BGD = 2		���ܵı���
	//GC_PR_FGD = 3		���ܵ�ǰ��
	
	mask.setTo(GC_BGD);//�ٴγ�ʼ������һ����Ϊ�˷�ֹ��ʼ��ʧЧ��֮��������ñ���
	
	//��ֹѡ����������
	rect.x = max(0,rect.x);
	rect.y = max(0,rect.y);
	rect.width = min(rect.width,src.cols - rect.x);
	rect.height = min(rect.height,src.rows - rect.y);

	mask(rect).setTo(Scalar(GC_PR_FGD));//��ѡ������ѡΪ���ܵ�ǰ��




}

//grabcutͼ��ָ�
void runGrabCut() {
	//�ж��û��Ƿ��ѡ����
	if (rect.width < 2 || rect.height<2)
	{
		return;
	}

	if (init)  //�ж��Ƿ��һ��ִ��  
		//��
	{
		grabCut(src, mask, rect, bgModel, fgmodel, 1);
	}
	else
		//��
	{
		grabCut(src,mask,rect,bgModel,fgmodel,1,GC_INIT_WITH_RECT); // 1 ���м��Σ�GC_INIT_WITH_RECT ÿ�γ�ʼ����ʽ
		init = true;
	}

}

//������
void onmouse(int event, int x, int y, int flags, void* param) {
	switch (event)
	{
	case EVENT_LBUTTONDOWN:        
		//�������
		rect.x = x;
		rect.y = y;
		rect.width = 1;
		rect.height = 1;
		init = false;
		numRun = 0;
		break;

	case EVENT_MOUSEMOVE:
		//��갴ס
		if (flags &EVENT_FLAG_LBUTTON)
		{
			rect = Rect(Point(rect.x , rect.y),Point(x,y));
			showImage();
		}
		break;

	case EVENT_LBUTTONUP:
		//��굯��
		if (rect.width >1 && rect.height>1)
		{
			showImage();
			setRoiMask();
		}
		break;

	default:
		break;
	}


}

//����� �� ��ʾ
void showImage() {   
	Mat result,binMask;
	binMask.create(mask.size(),CV_8UC1);
	binMask = mask & 1;
	if (init)
	{
		src.copyTo(result,binMask);
	}
	else {
		src.copyTo(result);
	}

	
	rectangle(result, rect, Scalar(0, 0, 255), 2, 8, 0);
	imshow("input", result);

}