#include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PI 3.141592653589793238
#define MU 0
#define SIGMA 0.2
#define H_BIN 32
class Points
{
public:

	int     x;
	int     y;

private:

};

class Object
{
    public:
	   size_t       objnum;              //objects' number
	   Rect         boundingbox;         //bounding box
	   Point        center;              //pixel center
	   Point3f      point3D;             //center position in real world in Robot coordinate
           int          area;                //pixel area
	   Rect         boundingbox2;
	   int          face_width;
	   int          face_height;
 Point pixPoint;

};

extern vector<Object> preface_body;

class Body_lan
{
public:	
	string Class;
	float  xmin;
	float  ymin;
	float  xmax;
	float  ymax;
	int    center_x;
	int    center_y;
	double sum_d;
	double sum_x;
	double sum_y;
	int sum_num;
	int else_num;
	double depth_ave;
	int    depth_x1;
	int    depth_x2;
	int    depth_y1;
	int    depth_y2;
	double x;
	double y;
	double z;
	Point3f point3D;
	Rect    boundingbox;
	Rect    depth_box;
        float   weight;


};

typedef struct{
	
	int    no;
	float  x;
	float  y;
	float  width;
	float  height;

} Face_lan;

typedef struct{
	
	string Class;
	float  xmin;
	float  ymin;
	float  xmax;
	float  ymax;

} Bodylan;

typedef struct{
	float  x;
	float  y;
	float  z;
} Position_lan;

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

class ObservationPoint
{
    public:
	int       gx;
	int       gy;
	double    kl;
	Scalar    color;
        Point  ob_point;    //////////coordinate in composition map
	double G_x;         //////////coordinate in real world
	double G_y;
	double dist;        ///pixel dist from present position
	short int cols;     ///col number
	short int rows;     ///row number
	double    ob_angle; ///position angle with target
	double F;           ///composition evaluation value;
	double ave_score;   ///average score
	double ave_score2; 
	int change_flag;
	int occlusion_flag;
	double occlusion;


void estimatepositioninf2(vector<TanObject> obs,Point2f G_point,vector<double> z);
void estimatepositionclu(vector<TanObject> obs,Point2f G_point,vector<double> z);
};

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

