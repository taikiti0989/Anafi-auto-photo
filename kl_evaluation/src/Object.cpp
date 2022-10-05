#include "kl_evaluation/Object.h"

using namespace std;
using namespace cv;


double Distance(Point3f p1, Point3f p2){
	return sqrt((float)((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z)));
}

double Distance(Point2f p1, Point2f p2)
{
	return sqrt((double)((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)));
}

double Gaussian(double x){
	return exp(-(x/1000.0-MU)*(x/1000.0-MU)/(2*SIGMA*SIGMA));///NC;
}

double Sigmoid(double x, float a, float b){
	return pow(1+exp(-a*(x-b)),-1);
}



void ObservationPoint::estimatepositioninf2(vector<TanObject> obs,Point2f G_point,vector<double> z)
{
	///////////////face////////////////////////
	double      new_x[50];
	double      new_y[50];
	double      new_area[50];
	double      new_width[50];
	double      new_height[50];
	///////////////////body and face///////////////////////////
	double      zx[50];
	double      zy[50];
	double      zw[50];
	double      zh[50];

	Object      temp_object[50];

	float screenG=0;
	int   n=0;

	vector<double> recx;
	vector<double> recy;
	float theta = 45.0*3.141592/180;//realsense d435の視野角
	float cosfai;
	float tanfai;
	double q,p;
	double fai=32.0*3.141592/180;
	double cosarufa,tanarufa;
	double predist_y;


	Point2f OP(G_point.x - ob_point.x,G_point.y - ob_point.y);//(観測地点→重心)ベクトル
	Point2f OA;//(観測地点→物体)ベクトル
	float projection_dist=0;

	//float a = (obserPoint.y - G_point.y)/(obserPoint.x - G_point.x);
	for(size_t j=0;j<obs.size();j++){
		Point2f temp(0,0);
		OA.x = obs[j].pix_center.x - ob_point.x;
		OA.y = obs[j].pix_center.y - ob_point.y;
		cosfai = (OP.x*OA.x + OP.y*OA.y)/(norm(OP)*norm(OA));
		tanfai = sqrt(1/(cosfai*cosfai) - 1);
		projection_dist = 320*tanfai/tan(theta);
		if(obs[j].pix_center.x<G_point.x)
			new_x[j]=320-projection_dist; //////////predicted x coordinate
		else
			new_x[j]=320+projection_dist;     		                              
		dist = sqrt((ob_point.x - obs[j].pix_center.x)*(ob_point.x - obs[j].pix_center.x) + (ob_point.y - obs[j].pix_center.y)*(ob_point.y - obs[j].pix_center.y));
        new_area[j]=obs[j].initialA * ((obs[j].initialD*obs[j].initialD)/(dist*dist));//////////predicted actual area ////////////
		new_width[j]=obs[j].i_width*obs[j].initialD/dist;
		new_height[j]=obs[j].i_height*obs[j].initialD/dist;
		q=norm(OA)/100;
		p=z[j];
		Point2f OT(q,p);

		if(p>0.77)
			tanarufa=(p-0.77)/q;
		else
			tanarufa=(0.77-p)/q;
		predist_y=240*tanarufa/tan(fai);
        if(predist_y>240)
			predist_y=240;
		if(p>0.77)
			new_y[j]=240-predist_y;
		else
			new_y[j]=240+predist_y;

		zx[j]=new_x[j];
		zy[j]=new_y[j];
		zw[j]=new_width[j];
		zh[j]=new_height[j];

	    if((480-zy[j])>(zh[j]/2))
		{
			temp_object[j].pixPoint.x=zx[j];
			temp_object[j].pixPoint.y=zy[j];
			temp_object[j].boundingbox.width=zw[j];
			temp_object[j].boundingbox.height=zh[j];
		}
		else
		{
			temp_object[j].pixPoint.x=zx[j];
			temp_object[j].pixPoint.y=zy[j];
			temp_object[j].boundingbox.width=zw[j];
			temp_object[j].boundingbox.height=480-zy[j]+zh[j]/2;
		}
		preface_body.push_back(temp_object[j]);

	}

	
}

void ObservationPoint::estimatepositionclu(vector<TanObject> obs,Point2f G_point,vector<double> z)
{
	///////////////face////////////////////////
	double      new_x[50];
	double      new_y[50];
	double      new_area[50];
	double      new_width[50];
	double      new_height[50];
	///////////////////body and face///////////////////////////
	double      zx[50];
	double      zy[50];
	double      zw[50];
	double      zh[50];

	Object      temp_object[50];

	float screenG=0;
	int   n=0;

	vector<double> recx;
	vector<double> recy;
	float theta = 45.0*3.141592/180;//realsense cameraの視野角
	float cosfai;
	float tanfai;
	double q,p;
	double fai=32.0*3.141592/180;
	double cosarufa,tanarufa;
	double predist_y;


	Point2f OP(G_point.x - ob_point.x,G_point.y - ob_point.y);//(観測地点→重心)ベクトル
	Point2f OA;//(観測地点→物体)ベクトル
	float projection_dist=0;

	//float a = (obserPoint.y - G_point.y)/(obserPoint.x - G_point.x);
	for(size_t j=0;j<obs.size();j++){
		Point2f temp(0,0);
		OA.x = obs[j].pix_center.x - ob_point.x;
		OA.y = obs[j].pix_center.y - ob_point.y;
		cosfai = (OP.x*OA.x + OP.y*OA.y)/(norm(OP)*norm(OA));
		tanfai = sqrt(1/(cosfai*cosfai) - 1);
		projection_dist = 320*tanfai/tan(theta);
		if(obs[j].pix_center.x<G_point.x)
			new_x[j]=320-projection_dist;                            /////////////////////预测的x坐标////////////////////////////
		else
			new_x[j]=320+projection_dist;                              	////////////////////预测的x坐标///////////////////////		                              
		dist = sqrt((ob_point.x - obs[j].pix_center.x)*(ob_point.x - obs[j].pix_center.x) + (ob_point.y - obs[j].pix_center.y)*(ob_point.y - obs[j].pix_center.y));
        new_area[j]=obs[j].initialA * ((obs[j].initialD*obs[j].initialD)/(dist*dist));//////////////////预测的面积//////////////////////////////
		new_width[j]=obs[j].i_width*obs[j].initialD/dist;
		new_height[j]=obs[j].i_height*obs[j].initialD/dist;
		q=norm(OA)/100;
		p=z[j];
		Point2f OT(q,p);

		if(p>0.77)
			tanarufa=(p-0.77)/q;
		else
			tanarufa=(0.77-p)/q;
		predist_y=240*tanarufa/tan(fai);

		if(p>0.77)
			new_y[j]=240-predist_y;
		else
			new_y[j]=240+predist_y;

		zx[j]=new_x[j];
		zy[j]=new_y[j];
		zw[j]=new_width[j];
		zh[j]=new_height[j];

	    if((480-zy[j])>(zh[j]/2))
		{
			temp_object[j].pixPoint.x=zx[j];
			temp_object[j].pixPoint.y=zy[j];
			temp_object[j].boundingbox.width=zw[j];
			temp_object[j].boundingbox.height=zh[j];
		}
		else
		{
			temp_object[j].pixPoint.x=zx[j];
			temp_object[j].pixPoint.y=zy[j];
			temp_object[j].boundingbox.width=zw[j];
			temp_object[j].boundingbox.height=480-zy[j]+zh[j]/2;
		}
		preface_body.push_back(temp_object[j]);

	}

	
}
