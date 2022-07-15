#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;
 
Rect rect;  //矩形定义
Mat src,image;
Mat mask, bgModel, fgmodel;//mask区域，背景，前景
int numRun = 0;
bool init = false;  //是否第一次执行的标志位

void runGrabCut();
void onmouse(int event, int x, int y, int flags,void*param);//控制鼠标框选
void showImage();//显示鼠标框选区域
void setRoiMask();//将框选区域作为ROI区域

int main() {
	src = imread("D:/OpenCV/picture zone/canjian.jpg",1);
	if (!src.data) {
		printf("ERROR");

		return -1;
	}

	mask.create(src.size(),CV_8UC1);  //mask定义
	mask.setTo(Scalar::all(GC_BGD));  //初始化mask，全部设置为背景

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
			printf("迭代次数：%d\n",numRun);
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
	//GC_BGD = 0	背景
	//GC_FGD = 1	前景
	//GC_PR_BGD = 2		可能的背景
	//GC_PR_FGD = 3		可能的前景
	
	mask.setTo(GC_BGD);//再次初始化，第一次是为了防止初始化失效，之后就是重置背景
	
	//防止选择的区域过界
	rect.x = max(0,rect.x);
	rect.y = max(0,rect.y);
	rect.width = min(rect.width,src.cols - rect.x);
	rect.height = min(rect.height,src.rows - rect.y);

	mask(rect).setTo(Scalar(GC_PR_FGD));//将选定区域选为可能的前景




}

//grabcut图像分割
void runGrabCut() {
	//判断用户是否框选区域
	if (rect.width < 2 || rect.height<2)
	{
		return;
	}

	if (init)  //判断是否第一次执行  
		//否
	{
		grabCut(src, mask, rect, bgModel, fgmodel, 1);
	}
	else
		//是
	{
		grabCut(src,mask,rect,bgModel,fgmodel,1,GC_INIT_WITH_RECT); // 1 运行几次，GC_INIT_WITH_RECT 每次初始化方式
		init = true;
	}

}

//鼠标控制
void onmouse(int event, int x, int y, int flags, void* param) {
	switch (event)
	{
	case EVENT_LBUTTONDOWN:        
		//按下鼠标
		rect.x = x;
		rect.y = y;
		rect.width = 1;
		rect.height = 1;
		init = false;
		numRun = 0;
		break;

	case EVENT_MOUSEMOVE:
		//鼠标按住
		if (flags &EVENT_FLAG_LBUTTON)
		{
			rect = Rect(Point(rect.x , rect.y),Point(x,y));
			showImage();
		}
		break;

	case EVENT_LBUTTONUP:
		//鼠标弹起
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

//画框框 并 显示
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