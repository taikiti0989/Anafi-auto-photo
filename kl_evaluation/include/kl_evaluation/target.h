#include <cv.h>
#include <math.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>  
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
using namespace std;
using namespace cv;
////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////for composition evaluation/////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

///////////Person class/////////////////////

class Person
{
public:
	float weight;
	/////////////////for face//////////////////////////////////ピクセル
	Point face_pixPoint;                //////////////////顔中心のピクセル座標
	Point face_leftup;                  //////////////////顔の左上座標
	Point face_rightdown;               //////////////////顔の右下座標
	int   face_width;                        //////////////////顔の幅
	int   face_height;                       //////////////////顔の高さ
	int   face_area;
	Rect  rectangle;                       //輪郭の外接矩形
	Rect  rectangle2;
	int   face_angle;
	double abs_angle;         /////radian////////////
	double position_angle;
	double h_position_x;
	double h_position_y;
	double face_radian;
	/////////////////////////////////////bayes/////////////////////////////////
	double relative_angle;
	double umap;           /////+////////
	double tr_angle;       /////+ -/////////
	double kl;
	double kl2;
	/////////////////////////////////////////////////加重移動平均/////////////////////////////////
	double ema;
	double kle;
	/////////////////for face//////////////////////////////////actual
	Point3f facecenter;                    //その物体のカメラからの位置(x,y,z)[m]の最小値  
	double  face_rarea;                    //m
	short int  face_rwidth;                //実際の幅[mm] 
	short int  face_rheight;              //実際の高さ[mm]

	Mat mask;
	////////////////for the upper part of the body//////////////////
	Point body_pixPoint;
	Point body_leftup;
	Point body_rightdown;
	int   body_width;
	int   body_height;
	int   body_area;
	////////////////////////////////////////////////////
	float		initialD;
	float		initialA;
	float           initialy;

	//////////////////////クラスタリング関係///////////////////
	int      group_number;
	Scalar face_color;

};

///////////Object class/////////////////////

class TanObject
{
    public:
	float  object_area;
	float  object_radius;
	Point2f safety_radius;
	Point2f pix_center;
	float  weight;
	Rect   boundingbox;
	Rect   estimateboundingbox;
	Rect   pixSaferect;
	float  initialD;
	float  initialA;
	float  initialy;

	float  i_width;
	float  i_height;

};


