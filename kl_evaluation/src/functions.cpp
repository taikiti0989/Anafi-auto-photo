#include "kl_evaluation/functions.h"
#include <iostream>
Point3f extract3DPoint(Rect bb, Mat& pointCloudMap)
{
	Ransac ransac;
	ransac.point_cloud = pointCloudMap;
	
	Point3f point3D;
	double point_num=0;
	Point3f mean, p_out(4.0f,4.0f,4.0f);
	Mat t_roi(ransac.point_cloud,bb);
	for(int y=0;y<t_roi.rows;++y){
		for(int x=0;x<t_roi.cols;++x){
			Point3f p3 = ((Point3f*)(t_roi.data+t_roi.step.p[0]*y))[x];
			if(p3.x < p_out.x && p3.y < p_out.y && p3.z < p_out.z){
				mean += p3;
				point_num++;
			}
		}
	}
	point3D = mean*(1/point_num);

	point3D = Point3f(point3D.x, point3D.y*cos(8*PI/180)-point3D.z*sin(8*PI/180), +point3D.y*sin(8*PI/180)+point3D.z*cos(8*PI/180));


	return Point3f(point3D.x,point3D.z+0, point3D.y+0.77);


}

// double update_velocity(double x,double y,double vx,double vy,double p,double g){
//     double ro=0.1;
// 	double w=0.5;
// 	double new_vx,new_vy;
//     new_vx = w * vx + ro * (p - x);
//     new_vy = w * vy + ro * (p - y);
// 	retun result;
// }

Point3f extract3DPointlan(Rect bb, Mat& pointCloudMap, Mat& mask)
{
	Ransac ransac;
	ransac.point_cloud = pointCloudMap;
	
	Point3f point3D;
	double point_num=0;
	Point3f mean, p_out(4.0f,4.0f,4.0f);
	Mat t_roi(ransac.point_cloud,bb);
	Mat roi_mask(mask,bb);
	for(int y=0;y<t_roi.rows;++y){
		for(int x=0;x<t_roi.cols;++x){
			if(roi_mask.data[y*roi_mask.step+x*roi_mask.elemSize()]){
				Point3f p3 = ((Point3f*)(t_roi.data+t_roi.step.p[0]*y))[x];
				if(p3.x < p_out.x && p3.y < p_out.y && p3.z < p_out.z){
					mean += p3;
					point_num++;
				}
			}
		}
	}
	point3D = mean*(1/point_num);

	point3D = Point3f(point3D.x, point3D.y*cos(8*PI/180)-point3D.z*sin(8*PI/180), +point3D.y*sin(8*PI/180)+point3D.z*cos(8*PI/180));


	return Point3f(point3D.x,point3D.z+0, point3D.y+0.77);


}

void clac_weightlan(vector<Body_lan> &obj)
{
	for(size_t i=0;i<obj.size();i++)
		obj[i].weight=1;

}

void get_center_of_balance(vector<TanObject> obs, Point2f& gpoint)
{
	float w_x=0;
	float w_y=0;
	float w=0;

	for(size_t i=0;i<obs.size();i++)
	{
		w_x=w_x+obs[i].weight*obs[i].pix_center.x;
		w_y=w_y+obs[i].weight*obs[i].pix_center.y;
		w=w+obs[i].weight;
	}
	gpoint.x=w_x/w;
	gpoint.y=w_y/w;

}

Point transRotCoordinate(Point before, Point center, double rot){
	Point after;
	after.x=cos(rot)*before.x-sin(rot)*before.y+center.x-center.x*cos(rot)+center.y*sin(rot);
	after.y=sin(rot)*before.x+cos(rot)*before.y+center.y-center.x*sin(rot)-center.y*cos(rot);
	return after;
}

Point transCoordinate2(Point before, Point center, double R, int row_num){
	Point after;
	Point AG(center.x-before.x,center.y-before.y);
	after.x=before.x-AG.x/norm(AG)*R*(row_num-1);
	after.y=before.y-AG.y/norm(AG)*R*(row_num-1);
	return after;
}

Point transCoordinate(Point before, Point center ,double R)//same col
{
	Point after;
	Point AG(center.x-before.x,center.y-before.y);
	double norm=(double)(AG.x*AG.x)+(double)(AG.y*AG.y);
	norm=sqrt(norm);
	after.x=before.x+AG.x/norm*R;
	after.y=before.y+AG.y/norm*R;
	
	return after;
}

double calcAngle(Point my_position, Point2f G_point, TanObject tanob){
	Point2f OG;////my_position to group center
	Point2f OA;////my_position to target center
	Point2f OR;////my_position to right view angle
	Point2f OL;////my_position to left view angle
	double angle;
	double angle_R;
	double angle_L;
	double view_angle=58*PI/180;
	Point view_ang_R=transRotCoordinate(G_point, my_position, view_angle);
	Point view_ang_L=transRotCoordinate(G_point, my_position, -view_angle);
	
	OG.x=G_point.x-my_position.x;
	OG.y=G_point.y-my_position.y;
	OA.x=tanob.pix_center.x-my_position.x;
	OA.y=tanob.pix_center.y-my_position.y;
	OR.x=view_ang_R.x-my_position.x;
	OR.y=view_ang_R.y-my_position.y;
	OL.x=view_ang_L.x-my_position.x;
	OL.y=view_ang_L.y-my_position.y;

	angle=acos(OG.ddot(OA)/(norm(OG)*norm(OA)))*180/PI;//PI/180
	angle_R=acos(OR.ddot(OA)/(norm(OR)*norm(OA)))*180/PI;
	angle_L=acos(OR.ddot(OA)/(norm(OL)*norm(OA)))*180/PI;

	if(angle_R<angle_L)
		angle=-angle;

	return angle;
}


void calc_sub_angle(vector<double> angle, vector<double>& sub_angle){
    for(size_t i=0;i<angle.size()-1;i++){
	for(size_t j=i+1;j<angle.size();j++){
		sub_angle.push_back(abs(angle[i]-angle[j]));
	}
    }
}

Scalar hsv2rgb( float hue )
{	int rgb[3], p, sector;
    static const int sector_data[][3]=
        {{0,2,1}, {1,2,0}, {1,0,2}, {2,0,1}, {2,1,0}, {0,1,2}};
    hue /= 30.0;
    sector = cvFloor(hue);
    p = cvRound(255*(hue - sector));
    p ^= sector & 1 ? 255 : 0;
    if(sector<0)
	sector=0;
    if(sector>5)
	sector=5;
    rgb[sector_data[sector][0]] = 255;
    rgb[sector_data[sector][1]] = 0;
    rgb[sector_data[sector][2]] = p;

    return Scalar(rgb[2], rgb[1], rgb[0],0);
}
void cal_force(vector<Object> objects,double& result)
{
	Mat src(Size(640,480),CV_8UC1,Scalar(255,255,255));
	Mat blur(Size(640,480),CV_8UC1,Scalar(255,255,255));
	for(size_t i=0;i<objects.size();i++){
		rectangle(src,objects[i].boundingbox,Scalar(0,0,0),-1,cv::LINE_AA);
		rectangle(blur,objects[i].boundingbox2,Scalar(0,0,0),-1,cv::LINE_AA);
	}
	//imshow("blur",blur);
	//imshow("src",src);
	//waitKey(10);
	/////////////////////////////////for grid//////////////////////////
	vector<Points> y_point;
	vector<Points> o_point;
	y_point.clear();
	o_point.clear();
	//int interval=50;
	int interval_x,interval_y;
	interval_x=20;
	interval_y=20;
	int y_count_x=0;
	int y_count_y=0;
	int o_count_x=0;
	int o_count_y=0;
	y_count_x=(int)(src.cols/interval_x);
	y_count_y=(int)(src.rows/interval_y);
	o_count_x=(int)(src.cols/interval_x);
	o_count_y=(int)(src.rows/interval_y);
	Points y_singlepoint,o_singlepoint; 
	double f_y_x,f_y_y,f_o_x,f_o_y;
/////////////////////////////////yohaku_force//////////////////////
	double y_sum_x=0;
	double y_sum_y=0;
	double y_E_sum=0;
	double img_center_x=src.cols/2.0;
	double img_center_y=src.rows/2.0;
	int y_num=0;
	double y_center_x=0;
	double y_center_y=0;
	for(int i=0;i<y_count_y;i++)
	{
    	    for(int j=0;j<y_count_x;j++)
    	    {
		if(blur.at<uchar>(i*interval_y,j*interval_x)==255)
	    	{
		    y_singlepoint.x=j;
		    y_singlepoint.y=i;
	     	    y_point.push_back(y_singlepoint);
		    y_sum_x=y_sum_x+j*interval_x-img_center_x;
		    y_sum_y=y_sum_y+i*interval_y-img_center_y;
	    	}
    	    }
	}
	f_y_x=y_sum_x*k;
	f_y_y=y_sum_y*k;
	//cout<<"y_fx:"<<f_y_x<<endl;
	//cout<<"y_fy:"<<f_y_y<<endl;
////////////////////////////////object_force//////////////
	double o_sum_x=0;
	double o_sum_y=0;
	double o_E_sum=0;
	double o_center_x=0;
	double o_center_y=0;
	int o_num=0;
	for(int i=0;i<o_count_y;i++)
	{
	    for(int j=0;j<o_count_x;j++)
	    {
		if(src.at<uchar>(i*interval_y,j*interval_x)==0)
		{
			o_singlepoint.x=j;
			o_singlepoint.y=i;
			o_point.push_back(o_singlepoint);
			o_sum_x=o_sum_x+j*interval_x-img_center_x;
			o_sum_y=o_sum_y+i*interval_y-img_center_y;
		}
	    }
	}
	f_o_x=o_sum_x*k;
	f_o_y=o_sum_y*k;
	//cout<<"o_fx:"<<f_o_x<<endl;
	//cout<<"o_fy:"<<f_o_y<<endl;

	double delta_fx,delta_fy;
	delta_fx=0.5*f_o_x+0.5*f_y_x;
	delta_fy=0.5*f_o_y+0.5*f_y_y;
	//cout<<"fx:"<<delta_fx<<endl;
	//cout<<"fy:"<<delta_fy<<endl;

	double norm;
	norm=delta_fx*delta_fx+delta_fy*delta_fy;
	result=sqrt(norm);
	//cout<<"result="<<result<<endl;
	//imshow("src",src);
	//waitKey(10);
	
}

///filtering: robust, weighted/exponential moving average,kalman, predict...
void apply_Filter(const unsigned int& type, const std::vector<double>& prv, const double& now, double& flt){
	if(type==0){
	//without filter
		flt=now;
	}else if(type==1){
	//low-pass filter LPF
		flt=now;
		for(int i=0;i<(int)prv.size();i++){
			flt+=prv[i];
		}
		flt=flt/(double)((int)prv.size()+1.0);
	}else if(type==2){
	//WMA
		flt=((double)prv.size()+1.0)*now;
		double normalize=((double)prv.size()+1.0);
		for(int i=0;i<(int)prv.size();i++){
			flt+=(double)(((int)prv.size())-i)*prv[i];
			normalize+=(double)(((int)prv.size())-i);
		}
		flt=flt/normalize;
	}else if(type==3){
	//EMA
		flt=now;
		double normalize=1.0;
		double alpha=2.0/((double)prv.size()+1.0);
	
		for(int i=0;i<(int)prv.size();i++){
			flt+=pow(1.0-alpha,i+1)*prv[i];
			normalize+=pow(1.0-alpha,i+1);
		}
		flt=flt/normalize;
	}else if(type==4){
	//simple ema
		//double weight=DIVISION( 1.0 , (double)prv.size() );
		//flt=weight*now+(1.0-weight)*flt;
	}else if(type==5){
		flt=0;
		double normalize=0;
		double alpha=-0.5;
		for(int i=0;i<(int)prv.size();i++){
			normalize+=exp(alpha*i);
			flt+=prv[i]*exp(alpha*i);
		}
		flt=flt/normalize;
	}else{

		
		flt=now;
	}
}
double Gaussian(double sigma, double x){
	double nc;
	nc=sqrt(2*PI*sigma*sigma);
	return exp(-x*x/(2*sigma*sigma))/nc;

}

double Distancelan(Point2f p1, Point2f p2){
	return sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2));

}
void selectPoint_information2(vector<ObservationPoint>& obserPoint,Point2f& Goal)//
{
	vector< vector<double> > dist;
	vector< vector<size_t> > order;
	order.resize(obserPoint.size());
	dist.resize(obserPoint.size());
	double score1[500];
        double score2[500];
	int score3[500];
	double score4[500];
	double gaosi,gaosi1,Gaosi1,gaosi2,Gaosi2;
	vector<double> score;//列ごとのスコア
	vector<double> score_l;//列ごとのスコア
	vector<size_t> order2;//scoreの順番を入れる
	vector<size_t> order_l;//score_lの順番を入れる
	vector<double> q;
	double miti=0.2;
	double sum_score=0;
	double new_score=0;
	
	string filename="total.csv";
	ofstream fout;
	fout.open(filename.c_str());

	for(size_t i=0; i<obserPoint.size();i++){		
		sum_score=obserPoint[i].F;
		if(cishu<11)
		{
		//cout<<"fff"<<endl;
			for(int quanshu=0;quanshu<cishu;quanshu++)
			{  		   
				q.push_back(pingjunf[quanshu][i]);   	
			}
		}
		else
		{
			for(int quanshu=cishu-10;quanshu<cishu;quanshu++)
				q.push_back(pingjunf[quanshu][i]);		
		}

		apply_Filter(3,q,sum_score,new_score);		
		obserPoint[i].ave_score2=new_score;
		obserPoint[i].ave_score=(new_score)*Gaussian(250,Distancelan(my_position,obserPoint[i].ob_point))*Gaussian(180,Distancelan(obserPoint[i].ob_point,Goal))*(1-obserPoint[i].occlusion)*10000.0;
		fout<<obserPoint[i].ave_score<<endl;
		q.clear();
		score.push_back(obserPoint[i].ave_score);	
		score_l.push_back(obserPoint[i].ave_score2);
		sum_score=0;
		new_score=0;
	}

	getSortOrder(score,order2,true);
	Goal.x=obserPoint[order2[0]].ob_point.x;
	Goal.y=obserPoint[order2[0]].ob_point.y;
	Goalr.x=(Goal.x/100-6);
	Goalr.y=(9-Goal.y/100);
	fout.close();

}

void selectPoint_information(vector<ObservationPoint>& obserPoint,Point2f& Goal)//
{
	vector< vector<double> > dist;
	vector< vector<size_t> > order;
	order.resize(obserPoint.size());
	dist.resize(obserPoint.size());
	double score1[500];
        double score2[500];
	int score3[500];
	double score4[500];
	double gaosi,gaosi1,Gaosi1,gaosi2,Gaosi2;
	vector<double> score;//列ごとのスコア
	vector<double> score_l;//列ごとのスコア
	vector<size_t> order2;//scoreの順番を入れる
	vector<size_t> order_l;//score_lの順番を入れる
	vector<double> q;
	double miti=0.2;
	double sum_score=0;
	double new_score=0;


	for(size_t i=0; i<obserPoint.size();i++){		
		sum_score=obserPoint[i].F;
		if(cishu<6)
		{
		//cout<<"fff"<<endl;
			for(int quanshu=0;quanshu<cishu;quanshu++)
			{  		   
				q.push_back(pingjunf[quanshu][i]);   	
			}
		}
		else
		{
			for(int quanshu=cishu-5;quanshu<cishu;quanshu++)
				q.push_back(pingjunf[quanshu][i]);		
		}

		apply_Filter(2,q,sum_score,new_score);		
		obserPoint[i].ave_score2=new_score;

		obserPoint[i].ave_score=(new_score)*Gaussian(200,Distancelan(my_position,obserPoint[i].ob_point))*Gaussian(200,Distancelan(obserPoint[i].ob_point,Goal));
		q.clear();
		score.push_back(obserPoint[i].ave_score);	
		score_l.push_back(obserPoint[i].ave_score2);
		sum_score=0;
		new_score=0;
	}

	getSortOrder(score,order2,true);
	Goal.x=obserPoint[order2[0]].ob_point.x;
	Goal.y=obserPoint[order2[0]].ob_point.y;
	Goalr.x=(Goal.x/100-6);
	Goalr.y=(9-Goal.y/100);

}


double calcPresentScore(vector<ObservationPoint>& obserPoint){
	vector<double> dist;
	vector<size_t> order;
	double score=0;
	
	for(size_t i=0;i<obserPoint.size();i++)
		dist.push_back(obserPoint[i].dist);
	
	getSortOrder(dist,order,false);

	//for(size_t i=0;i<NEIGHBORHOOD;i++)
		//score = score + obserPoint[order[i]].F;
	score=obserPoint[0].F;
	//score=score/NEIGHBORHOOD;
	return score;

}

//////////////////////////////////////viewpoint selection/////composition map
void evalFunctionlan21(Mat& voronoi_Image, vector<Body_lan>& objects,Point2f& G_point,Point2f& Goal_point,double& present_score,vector<double>& tkl)
{
	///////////変数の用意//////////////
	int	angle_interval=15;	//デフォルト::15
	float angle_interval2;
	int angle = 0;
	size_t N=360/angle_interval;//colの数
	vector<TanObject>	tanob;

	short int row_num=5;//同径方向の候補地点の数(デフォルト::4)
	float	min=0;
	float	max_angle=29.0f;
	float	occlusion_angle=5.0f;
	float	margin=100;
	float	R=30;//同径方向の距離(デフォルト::30)
	min = R+margin;
	vector<double> dist;
	vector<double> obs_angle;
	vector<double> abs_obs_angle;
	vector<double> sub_angle;

	vector<size_t> order;
	vector<size_t> order2;
	vector<size_t> order3;
	vector<size_t> order4;
	vector<int> obser_flag;
	Rect field(50,100,500,400);//フィールドの設定
	//その他
	int	sum_change_flag=0;
	const int image_dim=3;
	float test[image_dim];

	vector<ObservationPoint> obserPoint;
	int Goal_num=0;
	vector<double> temp_F;
	tanob.clear();
	//double new_y[50];
	///////////////////////////////////////////////////////////////////
if(objects.size()>0)
	{
	clac_weightlan(objects);             //対象物の重み,全て1//////////////////////////////////
	vector<Point>	ellipse_pt;
	vector<double> zz;
	////////////////////////////表示用変数tanobの設定//////////////////////////////////////
	//////////////////composition map の対象物位置確定//////////////////////
	for(size_t i = 0;i< objects.size();i++){ 
		TanObject temp;
		temp.pix_center.x = my_position.x + (objects[i].point3D.x*100.0) ;
		temp.pix_center.y = my_position.y - (objects[i].point3D.y*100.0) ;
		//temp.pixSaferect.width = objects[i].face_rwidth * 0.3;
		//temp.pixSaferect.height = objects[i].face_rheight * 0.3;//上から見ると正方形と仮定
		temp.pixSaferect.width = 50;
		temp.pixSaferect.height = 50;//上から見ると正方形と仮定	
		temp.weight = objects[i].weight;
		temp.estimateboundingbox = objects[i].boundingbox;
		double aaaaa;
		//aaaaa=objects[i].face_rheight;
		aaaaa=200;
		double r;
		r=objects[i].point3D.z;
		zz.push_back(r);
		ellipse_pt.push_back(temp.pix_center);
		temp.initialD = sqrt((my_position.x - temp.pix_center.x)*(my_position.x - temp.pix_center.x) + (my_position.y - temp.pix_center.y)*(my_position.y - temp.pix_center.y));
		temp.initialA = objects[i].boundingbox.area();
		temp.i_width=objects[i].xmax-objects[i].xmin;
		temp.i_height=objects[i].ymax-objects[i].ymin;
		tanob.push_back(temp);
	}
	
	////////////////////////////対象物体の描画//////////////////////////////
	for(size_t i=0;i<tanob.size();i++){
		ostringstream os;
		ostringstream xx;
		ostringstream gg;
		os.precision(3);
		xx.precision(3);
		os << i;
		circle( voronoi_Image, Point(cvRound((tanob[i].pix_center.x)), cvRound((tanob[i].pix_center.y))), 5, Scalar(0,0,0), cv::FILLED, 8, 0 );
		
	}

	get_center_of_balance(tanob,G_point);//物体の重心の計算(ピクセル座標)
	circle( voronoi_Image,G_point, 5, Scalar(0,255,200), cv::FILLED, 8, 0 );
	
	///////////////////////////////////////////////////////////////////////////
	//////////////////////////環境周辺の評価値作成/////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	string filename1="tkl.csv";
	ofstream fout;
	fout.open(filename1.c_str());
	tkl.clear();

	for(size_t i=0;i<N*row_num;i++){
	/////////////////////////////////
	/////////観測地点変換////////////
	/////////////////////////////////
	
		ObservationPoint temp_point;
		if(i==0){//自分の位置
			temp_point.ob_point = Point(my_position.x,my_position.y);
			temp_point.cols = i/row_num;	
			
		}

		else if(i%row_num!=0){
			//先ず座標変換する
			temp_point.ob_point = transCoordinate(obserPoint[i-1].ob_point,G_point, R);

			//objectとの角度を計算 angleに代入
			for(size_t j=0;j<tanob.size(); j++){
				obs_angle.push_back(calcAngle(temp_point.ob_point,G_point,tanob[j]));
				abs_obs_angle.push_back(abs(obs_angle[j]));
			}
			getSortOrder(abs_obs_angle,order,true);
			temp_point.occlusion_flag=0;
			if(abs_obs_angle.size()>=2){
				calc_sub_angle(obs_angle,sub_angle);
				getSortOrder(sub_angle,order2,false);
			
			}

			if(abs_obs_angle[order[0]]<max_angle){//物体が視野角に収まっている時

				if(obserPoint[i-1].rows != 0){
					temp_point.rows=obserPoint[i-1].rows - 1;
				}
				temp_point.cols = i/row_num;
				
			}
			else{//物体が視野角に収まっていない時
				for(size_t j=1;j<row_num;j++){
					if(i%row_num==j){
						for(size_t k=1;k<=j;k++){
							obserPoint[i-k].rows = k;
						}
					}
				}
				temp_point.ob_point = transCoordinate2(obserPoint[i-1].ob_point,G_point, R,row_num);
			//////////////////////////////////////////////////////////
				temp_point.rows=row_num;
				temp_point.change_flag = 1;
				temp_point.cols = i/row_num;
				//obserPoint.push_back(temp_point);
			}
			abs_obs_angle.clear();
			obs_angle.clear();
			sub_angle.clear();
			///////////////////////////////////test
		}

		//円周方向の観測地点の座標変換
		else{
			sum_change_flag=0;
			for(size_t j=1;j<=row_num;j++)
				sum_change_flag += obserPoint[i-j].change_flag;
			if(sum_change_flag==0){
				for(size_t j=1;j<=row_num;j++)
					obserPoint[i-j].rows = j;
			}
			angle_interval2 = -1 * angle_interval * PI/180;
			angle = angle + angle_interval;
				
			temp_point.ob_point= transRotCoordinate(obserPoint[i-row_num].ob_point, G_point,angle_interval2);

			temp_point.cols = i/row_num;

			for(size_t j=0;j<tanob.size(); j++){
				obs_angle.push_back(calcAngle(temp_point.ob_point,G_point,tanob[j]));
				abs_obs_angle.push_back(abs(obs_angle[j]));
			}
			getSortOrder(abs_obs_angle,order,true);
			temp_point.occlusion_flag=0;
			if(abs_obs_angle.size()>=2){
				calc_sub_angle(obs_angle,sub_angle);
				getSortOrder(sub_angle,order2,false);

			}
			if(abs_obs_angle[order[0]]<max_angle){//物体が視野角に収まっている時

				if(obserPoint[i-1].rows != 0){
					temp_point.rows=obserPoint[i-1].rows - 1;
				}
				temp_point.cols = i/row_num;
				//obserPoint.push_back(temp_point);

			}
			else{//物体が視野角に収まっていない時
				for(size_t j=1;j<row_num;j++){
					if(i%row_num==j){
						for(size_t k=1;k<=j;k++){
							obserPoint[i-k].rows = k;
						}
					}
				}
				temp_point.ob_point = transCoordinate2(temp_point.ob_point,G_point, R,row_num);
		//////////////////////////////////////////////////////////
				temp_point.rows=row_num;
				temp_point.change_flag = 1;
				temp_point.cols = i/row_num;
			}
			abs_obs_angle.clear();
			obs_angle.clear();
			sub_angle.clear();

		}
		if(i==row_num*N-1){
			sum_change_flag = temp_point.change_flag;
			for(size_t j=1;j<row_num;j++)
				sum_change_flag+=obserPoint[i-j].change_flag;
			if(sum_change_flag==0){
				temp_point.rows=1;
				for(size_t j=1;j<row_num;j++)
					obserPoint[i-j].rows = j+1;
			}
		}


		/////////////////////////////////
		///////////評価値計算////////////
		/////////////////////////////////
		
		if(i==5000){
			temp_point.F = 0;	
		}
		else{
			temp_point.estimatepositioninf2(tanob,G_point,zz);
			double func;
			double func1,eval_func;
			calKLD_multi(preface_body,func1);
			if(func1<0)
				func1=0.1;
			
			func=func1;
			fout<<func<<endl;
			tkl.push_back(func);
			double mef;
			mef=func;
			//cout<<"mef="<<mef<<endl;
			eval_func=exp(-mef/3.0);
			temp_point.F = eval_func;
			//cout<<"eval_func="<<eval_func<<endl;
			temp_F.push_back(eval_func);
			preface_body.clear();
	}
	
		temp_point.dist = sqrt(double(((my_position.x)-temp_point.ob_point.x)*((my_position.x)-temp_point.ob_point.x)+((my_position.y)-temp_point.ob_point.y)*((my_position.y)-temp_point.ob_point.y)));

			obserPoint.push_back(temp_point);

	}
	
	//fout.close();
//////////////////////////正規化F//////////////////////////////
	getSortOrder(temp_F,order3,false);

	//string filename4 ="正規化.csv";
	//ofstream fout4;
	//fout4.open(filename4.c_str());

	for(size_t i=0; i<obserPoint.size();i++){
		obserPoint[i].F = (obserPoint[i].F-temp_F[order3[0]])/(temp_F[order3[temp_F.size()-1]]-temp_F[order3[0]]);
		//fout4<<obserPoint[i].F<<endl;
	      
		yilunf.push_back(obserPoint[i].F);
	}
	   pingjunf.push_back(yilunf);
		yilunf.clear();
		//fout4.close();
	
	
	selectPoint_information(obserPoint,Goal_point);
	present_score=calcPresentScore(obserPoint);//現在地のスコアを計算

	for(size_t i=0; i<obserPoint.size();i++){///////////////色をつける
		obserPoint[i].color = hsv2rgb(obserPoint[i].F*119.9);
	}
	///////////////////////////voronoi_Imageへの書き込み/////////////////////////////////////

	for(size_t i=0;i<N*row_num;i++){

		if(obserPoint[i].ob_angle>60||obserPoint[i].ob_angle<-60)
			circle( voronoi_Image, obserPoint[i].ob_point, 7, obserPoint[i].color, cv::FILLED, 8, 0 );
		else
			circle( voronoi_Image, obserPoint[i].ob_point, 7, obserPoint[i].color, cv::FILLED, 8, 0 );

		if(obserPoint[i].occlusion_flag==1)
			circle( voronoi_Image, obserPoint[i].ob_point, 7, Scalar(211,0,148),cv::FILLED, 8, 0 );
		

		putText(voronoi_Image,"C",G_point,0,0.7,Scalar(0,0,0),1.9,cv::LINE_AA);
	}
	circle( voronoi_Image, obserPoint[0].ob_point, 12, Scalar(0,100,200), cv::FILLED, 10, 0 );
	putText(voronoi_Image,"S",Point( obserPoint[0].ob_point.x-7, obserPoint[0].ob_point.y+7),0,0.6,Scalar(0,0,0),2.0,cv::LINE_AA);
	circle( voronoi_Image,Point( Goal_point.x,Goal_point.y),11, Scalar(50,150,50), cv::FILLED, 8, 0 );
	putText(voronoi_Image,"G",Point(Goal_point.x-7,Goal_point.y+7),0,0.6,Scalar(0,0,0),2.0,cv::LINE_AA);
	
	for(size_t i=0; i<6;i++){
		Scalar temp_color=hsv2rgb(120*i/5);
		rectangle(voronoi_Image, Point(0,120-20*i),Point(30,140-20*i), temp_color, -1, 8, 0 );
	}

}
	
}


void DepthToGrayscale(Mat& pointCloudMap, Mat& depthMap)
{
	unsigned short  lv = 0;

	depthMap.create(pointCloudMap.size(),CV_8UC1);
	vector<Mat> planes;
	double max_val;
	split(pointCloudMap,planes);
	minMaxLoc(planes[2],0,&max_val);
	planes[2].convertTo(depthMap,CV_8U,max_val?255./max_val:0.,0);
}

void setPoints(Mat& ps, float depth_max_range, float depth_min_range) {
        // sets the pointCloud variable and also initialises points used to all zeros
        // sets the number of points
	Mat depth;
        depth = ps;
	int numberOfPoints;					// the number of points in the point cloud
        int pointsLeft;	
	int *pointsUsed;
		int c,r;
		r = depth.rows;
		c = depth.cols;
		vector<Point3f> pointCloud;
		vector<Point2i> pointIndices;
		
				
		Point3f p0(depth_max_range+0.3f,depth_max_range+0.3f,depth_max_range+0.3f);	
		pointCloud.clear();
		pointIndices.clear();
		for(int y=0;y<r;y++){
				for(int x=0;x<c;x++){
					Point3f &point = ((Point3f*)(depth.data+depth.step.p[0]*y))[x];
				
					if(point.z != 0 && point.z > depth_min_range && point.z < depth_max_range){
						pointIndices.push_back(Point(x,y));
						pointCloud.push_back(point);
						//cout<<"depth:"<<point.z<<"[m]"<<endl;
					}
					else{
						point = p0;
						//cout<<"depth:"<<point.z<<"[m]"<<endl;
					}
				}
			}
		numberOfPoints = pointCloud.size();

        pointsUsed = new int[numberOfPoints];
        for (int i = 0; i < numberOfPoints; i++) {
                pointsUsed[i] = 0;
        }
        pointsLeft = numberOfPoints;
}

double judgeOcclusion(vector<Object>& objects)
{
	double output;
	double ratio;
	vector<double> ratios;
	vector<size_t> order_ratio;
	ratios.clear();
	order_ratio.clear();
    for(size_t i=0;i<objects.size()-1;i++){
        //cout<<"i="<<i<<endl;
	for(size_t j=i+1;j<objects.size();j++){
	    //cout<<"j="<<j<<endl;
	    Rect rect = objects[i].boundingbox & objects[j].boundingbox;
	    if(objects[i].boundingbox.area()<objects[j].boundingbox.area())
		ratio=(double)(rect.area()*1.0/objects[i].boundingbox.area()*1.0);
	    else
		ratio=(double)(rect.area()*1.0/objects[j].boundingbox.area()*1.0);
	    if(ratio!=0)
	        ratios.push_back(ratio);
	}
    }
    if (ratios.size()!=0){
	getSortOrder(ratios,order_ratio,true);
	output = ratios[order_ratio[0]];
    }
    else 
	output = 0;
	
    return output;
}

double judgeOcclusion2(vector<Body_lan>& objects)
{
	double output;
	double ratio;
	vector<double> ratios;
	vector<size_t> order_ratio;
	ratios.clear();
	order_ratio.clear();
    for(size_t i=0;i<objects.size()-1;i++){
        //cout<<"i="<<i<<endl;
	for(size_t j=i+1;j<objects.size();j++){
	    //cout<<"j="<<j<<endl;
	    Rect rect = objects[i].boundingbox & objects[j].boundingbox;
	    if(objects[i].boundingbox.area()<objects[j].boundingbox.area())
		ratio=(double)(rect.area()*1.0/objects[i].boundingbox.area()*1.0);
	    else
		ratio=(double)(rect.area()*1.0/objects[j].boundingbox.area()*1.0);
	    if(ratio!=0)
	        ratios.push_back(ratio);
	}
    }
    if (ratios.size()!=0){
	getSortOrder(ratios,order_ratio,true);
	output = ratios[order_ratio[0]];
    }
    else 
	output = 0;
	
    return output;
}

void getSortOrder(const vector<double>& values, vector<size_t>& order, bool descending)
{
  
    vector<std::pair<size_t, vector<double>::const_iterator> > order_pair(values.size());

    size_t n = 0;
    for (vector<double>::const_iterator it = values.begin(); it != values.end(); ++it, ++n)
        order_pair[n] = make_pair(n, it);

    std::stable_sort(order_pair.begin(),order_pair.end(),orderingSorter());
    if (descending == false) std::reverse(order_pair.begin(),order_pair.end());

    vector<size_t>(order_pair.size()).swap(order);
    for (size_t i = 0; i < order_pair.size(); ++i)
    {
        order[i] = order_pair[i].first;
    }
}


// 行列と行列の足し算
Mtx mplus(const Mtx& a, const Mtx& b)
{
	Mtx m_plus;
	Vec1 m1;
	Vec1 m2;
	m1.push_back(a[0][0]+b[0][0]);
	m1.push_back(a[0][1]+b[0][1]);
	m2.push_back(a[1][0]+b[1][0]);
	m2.push_back(a[1][1]+b[1][1]);
	m_plus.push_back(m1);
	m_plus.push_back(m2);

	return m_plus;
}



	// 行列と行列の積
Mtx mul_k(const Mtx& a, const Mtx& b) 
	{
		vector <vector <double> > c(SZ(a), vector<double>(SZ(b[0])));
		for (int i = 0; i < SZ(a); i++)
		{
			for (int k = 0; k < SZ(b); k++)
			{
				for (int j = 0; j < SZ(b[0]); j++)
				{
					c[i][j] += a[i][k]*b[k][j];
				}
			}
		}
		return c;
	}

	// スカラーと行列の積
Mtx mulScalar_k(const Mtx& a, const double scalar)
	{
		Mtx ret(a);
		for (int i = 0; i < SZ(ret); i++)
		{
			for (int k = 0; k < SZ(ret[0]); k++)
			{
				ret[i][k] *= scalar;
			}
		}
		return ret;
	}

	// スカラーとベクトルの積
Vec1 mulScalar_k(const Vec1& a, const double scalar)
	{
		Vec1 ret(a);
		for (int i = 0; i < SZ(ret); i++)
		{
			ret[i] *= scalar;
		}
		return ret;
	}

	// ベクトルと行列(2X2)の積
Vec1 mulVecMul_k(const Vec1& ve, const Mtx& mu) 
	{
	    Vec1 re;
		re.clear();
	    double value0,value1;
		value0=ve[0]*mu[0][0]+ve[1]*mu[1][0];
		value1=ve[0]*mu[0][1]+ve[1]*mu[1][1];
		re.push_back(value0);
		re.push_back(value1);

		return re;
	}

	// 行列の転置
Mtx transpose_k( const Mtx& vs )
	{
		const int H = SZ(vs);
		const int W = SZ(vs[0]);

		Mtx ret(W, Vec1(H) );
		for (int y = 0; y < W; y++)
		{
			for (int x = 0; x < H; x++)
			{
				ret[y][x] = vs[x][y];
			}
		}

		return ret;
	}

	// 2*2の行列式を求める
double det_k(const Mtx& m)
	{
		return m[0][0]*m[1][1]-m[0][1]*m[1][0];
	}

	// 2*2の逆行列を求める
Mtx solve_k(const Mtx& m )
	{
		vector < vector <double> > ret(m);
		swap(ret[0][0],ret[1][1]);
		ret[0][1] = -ret[0][1];
		ret[1][0] = -ret[1][0];
		ret = mulScalar_k(ret,1.0/abs(det_k(m)));
		return ret;
	}


    // 平均、分散共分散行列を出力
void transfer(Object object,Vec1& mean, Mtx& vcm) //被積分関数
{
	////for mean
	mean.push_back(object.pixPoint.x);
	mean.push_back(object.pixPoint.y);

	////for Variance-covariance matrix
	double sigmax,sigmay;
	sigmax=object.boundingbox.width/2.58/2.0;
	sigmay=object.boundingbox.height/2.58/2.0;
	double s_sigx,s_sigy;
	s_sigx=sigmax*sigmax;
	s_sigy=sigmay*sigmay;
	Vec1 first;
	Vec1 second;
	first.push_back(s_sigx);
	first.push_back(0.0);
	second.push_back(0.0);
	second.push_back(s_sigy);
	vcm.push_back(first);
	vcm.push_back(second);
	
}

double trace(Mtx a)
{
	double result;
	result=a[0][0]+a[1][1];

	return result;
}

//////////////for single Gaussian 
double kl_calculation(Object object1,Object object2, double& kl)
{
	Vec1 mean1;
	Mtx vcm1;
	transfer(object1,mean1,vcm1);
	Vec1 mean2;
	Mtx vcm2;
	transfer(object2,mean2,vcm2);

	////for log calculation
	double value1,value2;
	double log_value;
	double ratio_value;
	value1=det_k(vcm1);
	value2=det_k(vcm2);
	ratio_value=value2/value1;
	log_value=log(ratio_value);

	////for trace
	Mtx inverse;
	inverse=solve_k(vcm2);
	Mtx multi;
	multi=mul_k(inverse,vcm1);
	double trace_value;
	trace_value=trace(multi);

	////others
	Vec1 delt;
	delt.push_back(mean1[0]-mean2[0]);
	delt.push_back(mean1[1]-mean2[1]);
	Vec1 smt;
	smt=mulVecMul_k(delt,inverse);
	double third_value;
	third_value=smt[0]*delt[0]+smt[1]*delt[1];

	kl=0.5*log_value+0.5*trace_value+0.5*third_value-1;

	return kl;
}

/////tab
void tab(Object a,Object b, double& tab_value)
{
	Vec1 mean1;
	Mtx vcm1;
	transfer(a,mean1,vcm1);
	Vec1 mean2;
	Mtx vcm2;
	transfer(b,mean2,vcm2);

	////second
	Mtx sum_vcm;
	sum_vcm=mplus(vcm1,vcm2);
	double second,second_value;
	second=det_k(sum_vcm);
	second_value=log(second);


	/////third
	Vec1 delt;
	delt.push_back(mean2[0]-mean1[0]);
	delt.push_back(mean2[1]-mean1[1]);
	Mtx inverse;
	inverse=solve_k(sum_vcm);
	Vec1 smt;
	smt=mulVecMul_k(delt,inverse);
	double third_value;
	third_value=smt[0]*delt[0]+smt[1]*delt[1];

	tab_value=-log(2*M_PI)-0.5*second_value-0.5*third_value;
	tab_value=exp(tab_value);
}


/////lower bound
void lowerbound(vector<Object> objects1,vector<Object> objects2,double& lower)
{
	double weight;
	double size;
	size=(double)(objects1.size()*1.0);
	weight=1.0/size;
	double sum_sum=0;
	for(size_t i=0;i<objects1.size();i++)
	{
		//////分子
		double sum_bun=0;
		for(size_t j=0;j<objects1.size();j++)
		{
			double klv;
			kl_calculation(objects1[i],objects1[j],klv);
			sum_bun=sum_bun+weight*exp(-klv);
		}

		////分母
		double sum_bubo=0;
		for(size_t j=0;j<objects2.size();j++)
		{
			double tab_value;
			tab(objects1[i],objects2[j],tab_value);
			sum_bubo=sum_bubo+weight*tab_value;
		}

		/////log
		double ratio;
		ratio=sum_bun/sum_bubo;
		double log_value;
		log_value=log(ratio);
		
		/////sum
		sum_sum=sum_sum+weight*log_value;
	}
	lower=sum_sum;
}

/////upper bound
void upperbound(vector<Object> objects1,vector<Object> objects2,double& upper)
{
	double weight;
	double size;
	size=(double)(objects1.size()*1.0);
	weight=1.0/size;
	double sum_sum=0;
	for(size_t i=0;i<objects1.size();i++)
	{
		//////分子
		double sum_bun=0;
		for(size_t j=0;j<objects1.size();j++)
		{
			double zaa_value;
			tab(objects1[i],objects1[j],zaa_value);
			sum_bun=sum_bun+weight*zaa_value;
		}
		////分母
		double sum_bubo=0;
		for(size_t j=0;j<objects2.size();j++)
		{
			double klv;
			kl_calculation(objects1[i],objects2[j],klv);
			sum_bubo=sum_bubo+weight*exp(-klv);
		}
		/////log
		double ratio;
		ratio=sum_bun/sum_bubo;
		double log_value;
		log_value=log(ratio);
		
		/////sum
		sum_sum=sum_sum+weight*log_value;
	}
	upper=sum_sum;
}

//////////kl mean
double kl_mean(vector<Object> objects1,vector<Object> objects2)
{
	double up,low;
	upperbound(objects1,objects2,up);
	lowerbound(objects1,objects2,low);
	double mean_kl;
	mean_kl=(up+low)/2.0;

	return mean_kl;
}


/////////KLD calculation for clusters
double calKLD_cluster(vector<Object> objects,double& result)
{
	vector <size_t> orderbx;
	vector<double> body_x;
	vector <size_t> orderby;
	vector<double> body_y;
	for(size_t b=0;b<objects.size();b++)
	{
		body_x.push_back(objects[b].pixPoint.x-objects[b].boundingbox.width/2);
		body_x.push_back(objects[b].pixPoint.x+objects[b].boundingbox.width/2);
		body_y.push_back(objects[b].pixPoint.y-objects[b].boundingbox.height/2);
		body_y.push_back(objects[b].pixPoint.y+objects[b].boundingbox.height/2);
	}
	getSortOrder(body_x,orderbx,true);
	getSortOrder(body_y,orderby,true);

	////////////cluster object generating
	double center_xll,center_yll;
	double r_widthll,r_heightll;
	center_xll=(body_x[orderbx[objects.size()-1]]+body_x[orderbx[0]])/2;
	center_yll=(body_y[orderby[objects.size()-1]]+body_y[orderby[0]])/2;
	r_widthll=body_x[orderbx[objects.size()-1]]-body_x[orderbx[0]];
	r_heightll=body_y[orderby[objects.size()-1]]-body_y[orderby[0]];

	Object cluster0;
	cluster0.boundingbox.width=r_widthll;
	cluster0.boundingbox.height=r_heightll;
	cluster0.pixPoint.x=center_xll;
	cluster0.pixPoint.y=center_yll;
	//////////////model cluster generating
	Object model0;

	model0.boundingbox.width=300;
	model0.boundingbox.height=220;
	model0.pixPoint.x=320;
	model0.pixPoint.y=240;

	//////////////kld calculation
	kl_calculation(cluster0,model0,result);

	return result;
}


////////KLD calculation for multi humans//////////rule of thirds
double calKLD_multi(vector<Object> objects,double& result)
{
	vector<Object> object0;
	Object mid_object;
	Object mid1,mid2;
	Object mid3;
	if(objects.size()==1)
	{
		if(objects[0].pixPoint.x<320)
		{
			mid_object.pixPoint.x=285;
			mid_object.pixPoint.y=240;
			mid_object.boundingbox.width=130;
			mid_object.boundingbox.height=236;
			// mid_object.boundingbox.width=80;
			// mid_object.boundingbox.height=160;
		}
		else
		{
			mid_object.pixPoint.x=571;
			mid_object.pixPoint.y=240;
			mid_object.boundingbox.width=130;
			mid_object.boundingbox.height=236;
			// mid_object.boundingbox.width=80;
			// mid_object.boundingbox.height=160;
		}
		//object0.push_back(mid_object);
		kl_calculation(objects[0],mid_object,result);
	}
	else
	{
		if(objects.size()==2)
		{
			mid1.boundingbox.width=90;
			mid1.boundingbox.height=160;
			// mid1.boundingbox.width=60;
			// mid1.boundingbox.height=100;
			mid1.pixPoint.x=285;
			mid1.pixPoint.y=240;
			mid2.boundingbox.width=90;
			mid2.boundingbox.height=160;
			// mid2.boundingbox.width=60;
			// mid2.boundingbox.height=100;
			mid2.pixPoint.x=571;
			mid2.pixPoint.y=240;
			object0.push_back(mid1);
			object0.push_back(mid2);
			result=kl_mean(objects,object0);

			// mid1.boundingbox.width=90;
			// mid1.boundingbox.height=160;
			// mid1.pixPoint.x=214;
			// mid1.pixPoint.y=240;
			// mid2.boundingbox.width=90;
			// mid2.boundingbox.height=160;
			// mid2.pixPoint.x=428;
			// mid2.pixPoint.y=240;
			// object0.push_back(mid1);
			// object0.push_back(mid2);
			// result=kl_mean(objects,object0);

			// mid1.boundingbox.width=90;
			// mid1.boundingbox.height=160;
			// mid1.pixPoint.x=100;
			// mid1.pixPoint.y=240;
			// mid2.boundingbox.width=90;
			// mid2.boundingbox.height=160;
			// mid2.pixPoint.x=300;
			// mid2.pixPoint.y=240;
			// object0.push_back(mid1);
			// object0.push_back(mid2);
			// result=kl_mean(objects,object0);
		}
		else
		{
			
			mid1.boundingbox.width=90;
			mid1.boundingbox.height=160;
			// mid1.boundingbox.width=60;
			// mid1.boundingbox.height=100;
			mid1.pixPoint.x=285;
			mid1.pixPoint.y=240;
			mid2.boundingbox.width=90;
			mid2.boundingbox.height=160;
			// mid2.boundingbox.width=60;
			// mid2.boundingbox.height=100;
			mid2.pixPoint.x=571;
			mid2.pixPoint.y=240;
			mid3.boundingbox.width=90;
			mid3.boundingbox.height=160;
			// mid3.boundingbox.width=60;
			// mid3.boundingbox.height=100;
			mid3.pixPoint.x=320;
			mid3.pixPoint.y=240;
			object0.push_back(mid1);
			object0.push_back(mid2);
			result=kl_mean(objects,object0);
		}
	}
	return result;
}


double calKLD_multi2(vector<Object> objects,double& result)
{
	vector<Object> object0;
	Object mid_object;
	Object mid1,mid2;
	if(objects.size()==1)
	{
		if(objects[0].pixPoint.x<428)
		{
			mid_object.pixPoint.x=285;
			mid_object.pixPoint.y=240;
			mid_object.boundingbox.width=130;
			mid_object.boundingbox.height=236;
		}
		else
		{
			mid_object.pixPoint.x=570;
			mid_object.pixPoint.y=240;
			mid_object.boundingbox.width=130;
			mid_object.boundingbox.height=236;
		}
		//object0.push_back(mid_object);
		kl_calculation(objects[0],mid_object,result);
	}
	else
	{
		if(objects.size()==2)
		{
			mid1.boundingbox.width=130;
			mid1.boundingbox.height=100;
			mid1.pixPoint.x=285;
			mid1.pixPoint.y=240;
			mid2.boundingbox.width=130;
			mid2.boundingbox.height=100;
			mid2.pixPoint.x=570;  //600
			mid2.pixPoint.y=240;  //240
			object0.push_back(mid1);
			object0.push_back(mid2);
			result=kl_mean(objects,object0);
		}
		else
		{
			vector <size_t> orderbx;
			vector<double> body_x;
			vector <size_t> orderby;
			vector<double> body_y;
			for(size_t b=0;b<objects.size();b++)
			{
				body_x.push_back(objects[b].pixPoint.x-objects[b].boundingbox.width/2);
				body_x.push_back(objects[b].pixPoint.x+objects[b].boundingbox.width/2);
				body_y.push_back(objects[b].pixPoint.y-objects[b].boundingbox.height/2);
				body_y.push_back(objects[b].pixPoint.y+objects[b].boundingbox.height/2);
			}
			getSortOrder(body_x,orderbx,true);
			getSortOrder(body_y,orderby,true);
		    //rectangle(colorImagelk,Point(body_x[orderbx[objects.size()-1]],body_y[orderby[objects.size()-1]]),Point(body_x[0],body_y[0]),Scalar(255,0,0),2,8,0);
			double center_x,center_y;
			double r_width,r_height;
			center_x=(body_x[orderbx[objects.size()-1]]+body_x[0])/2;
			center_y=(body_y[orderby[objects.size()-1]]+body_y[0])/2;
			r_width=body_x[0]-body_x[orderbx[objects.size()-1]];
			r_height=body_y[0]-body_y[orderby[objects.size()-1]];
			Object cluster0;
			cluster0.boundingbox.width=r_width;
			cluster0.boundingbox.height=r_height;
			cluster0.pixPoint.x=center_x;
			cluster0.pixPoint.y=center_y;
			//////////////model cluster generating
			Object model0;

			model0.boundingbox.width=350;
			model0.boundingbox.height=270;
			model0.pixPoint.x=320;
			model0.pixPoint.y=240;
			kl_calculation(cluster0,model0,result);
		}
	}
	return result;
}

void clac_weightlanclu(vector<Person> &obj)
{
	for(size_t i=0;i<obj.size();i++)
		obj[i].weight=1;

}

//cluster対象の視点選択
float evalFunctionlanclu(Mat& voronoi_Image, vector<Person>& objects,Point2f& G_point,Point2f& Goal_point,double& present_score, vector<double>& tkl)
{
	///////////変数の用意//////////////
	int	angle_interval=15;	//デフォルト::15
	float angle_interval2;
	int angle = 0;
	size_t N=360/angle_interval;//colの数
	vector<TanObject>	tanob;


	short int row_num=5;//同径方向の候補地点の数(デフォルト::4)
	float	min=0;
	float	max_angle=29.0f;
	float	occlusion_angle=5.0f;
	float	margin=100;
	float	R=50;//同径方向の距離(デフォルト::30)
	min = R+margin;
	vector<double> dist;
	vector<double> obs_angle;
	vector<double> abs_obs_angle;
	vector<double> sub_angle;
	
	vector<size_t> order;
	vector<size_t> order2;
	vector<size_t> order3;
	vector<size_t> order4;
	vector<int> obser_flag;

	

	Rect field(50,100,500,400);//フィールドの設定
	//その他
	int	sum_change_flag=0;
	const int image_dim=3;
	float test[image_dim];

	vector<ObservationPoint> obserPoint;
	int Goal_num=0;
	vector<double> temp_F;
	tanob.clear();
	//double new_y[50];
	///////////////////////////////////////////////////////////////////

	clac_weightlanclu(objects);                               //対象物の重み,全て1//////////////////////////////////
	vector<Point>	ellipse_pt;
	vector<double> zz;
	////////////////////////////表示用変数tanobの設定//////////////////////////////////////
	//////////////////composition map の対象物位置確定//////////////////////
	for(size_t i = 0;i< objects.size();i++){ 
		TanObject temp;
		temp.pix_center.x = my_position.x + (objects[i].facecenter.x*100.0) ;
		temp.pix_center.y = my_position.y - (objects[i].facecenter.y*100.0) ;
		temp.pixSaferect.width = 50;
		temp.pixSaferect.height = 50;//上から見ると正方形と仮定	
		temp.weight = objects[i].weight;
		temp.estimateboundingbox = objects[i].rectangle;
		double aaaaa;
		aaaaa=200;
		double r;
		r=objects[i].facecenter.z;
		zz.push_back(r);
		ellipse_pt.push_back(temp.pix_center);
		temp.initialD = sqrt((my_position.x - temp.pix_center.x)*(my_position.x - temp.pix_center.x) + (my_position.y - temp.pix_center.y)*(my_position.y - temp.pix_center.y));
		temp.initialA = objects[i].rectangle.area();
		temp.i_width=objects[i].face_width;
		temp.i_height=objects[i].face_height;
		//temp.initialy=objects[i].pixPoint.y;
		tanob.push_back(temp);
	}

	////////////////////////対象物体の描画//////////////////////////////
	for(size_t i=0;i<tanob.size();i++){
		ostringstream os;
		ostringstream xx;
		ostringstream gg;
		os.precision(3);
		xx.precision(3);
		os << i;
		//rectangle(voronoi_Image, Point((tanob[i].pixSaferect.x),(tanob[i].pixSaferect.y)),Point((tanob[i].pixSaferect.x)+tanob[i].pixSaferect.width,(tanob[i].pixSaferect.y)+tanob[i].pixSaferect.height),Scalar(250,210,100), cv::FILLED, 8, 0 );
		circle( voronoi_Image, Point(cvRound((tanob[i].pix_center.x)), cvRound((tanob[i].pix_center.y))), 5, Scalar(0,0,0), cv::FILLED, 8, 0 );
	}
	///////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////
	get_center_of_balance(tanob,G_point);//物体の重心の計算(ピクセル座標)
	circle( voronoi_Image,G_point, 5, Scalar(0,255,200), cv::FILLED, 8, 0 );
	//obserPoint.push_back(temp_ini);	//obserPoint[0]が現在位置
	///////////////////////////////////////////////////////////////////////////
	//////////////////////////環境周辺の評価値作成/////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	string filename1="tkl.csv";
	ofstream fout;
	fout.open(filename1.c_str());
	tkl.clear();
	///////////////////////////////////////////////////////////////////map calculation begin///////////////////////////////
	for(size_t i=0;i<N*row_num;i++){
	/////////////////////////////////
	/////////観測地点変換////////////
	/////////////////////////////////
		/*同径方向の観測地点の座標変換*/
		ObservationPoint temp_point;
		if(i==0){//自分の位置
			temp_point.ob_point = Point(my_position.x,my_position.y);
			temp_point.cols = i/row_num;	

		}

		else if(i%row_num!=0){
			//先ず座標変換する
			temp_point.ob_point = transCoordinate(obserPoint[i-1].ob_point,G_point, R);

			//objectとの角度を計算 angleに代入
			for(size_t j=0;j<tanob.size(); j++){
				obs_angle.push_back(calcAngle(temp_point.ob_point,G_point,tanob[j]));
				abs_obs_angle.push_back(abs(obs_angle[j]));
			}
			order4.clear();
			getSortOrder(abs_obs_angle,order4,true);
			temp_point.occlusion_flag=0;
			if(abs_obs_angle.size()>=2){
				calc_sub_angle(obs_angle,sub_angle);
				getSortOrder(sub_angle,order2,false);

			}
			
			if(abs_obs_angle[order4[0]]<max_angle){//物体が視野角に収まっている時

				if(obserPoint[i-1].rows != 0){
					temp_point.rows=obserPoint[i-1].rows - 1;
				}
				temp_point.cols = i/row_num;
				
			}
			
			else{//物体が視野角に収まっていない時
				for(size_t j=1;j<row_num;j++){
					if(i%row_num==j){
						for(size_t k=1;k<=j;k++){
							obserPoint[i-k].rows = k;
						}
					}
				}
				temp_point.ob_point = transCoordinate2(obserPoint[i-1].ob_point,G_point, R,row_num);

				temp_point.rows=row_num;
				temp_point.change_flag = 1;
				temp_point.cols = i/row_num;

			}
			abs_obs_angle.clear();
			obs_angle.clear();
			sub_angle.clear();

		}

		//円周方向の観測地点の座標変換
		else{
			sum_change_flag=0;
			for(size_t j=1;j<=row_num;j++)
				sum_change_flag += obserPoint[i-j].change_flag;
			if(sum_change_flag==0){
				for(size_t j=1;j<=row_num;j++)
					obserPoint[i-j].rows = j;
			}
			angle_interval2 = -1 * angle_interval * PI/180;
			angle = angle + angle_interval;
				
			temp_point.ob_point= transRotCoordinate(obserPoint[i-row_num].ob_point, G_point,angle_interval2);

			temp_point.cols = i/row_num;

			for(size_t j=0;j<tanob.size(); j++){
				obs_angle.push_back(calcAngle(temp_point.ob_point,G_point,tanob[j]));
				abs_obs_angle.push_back(abs(obs_angle[j]));
			}
			getSortOrder(abs_obs_angle,order,true);
			temp_point.occlusion_flag=0;
			if(abs_obs_angle.size()>=2){
				calc_sub_angle(obs_angle,sub_angle);
				getSortOrder(sub_angle,order2,false);

			}
	
			if(abs_obs_angle[order[0]]<max_angle){//物体が視野角に収まっている時

				if(obserPoint[i-1].rows != 0){
					temp_point.rows=obserPoint[i-1].rows - 1;
				}
				temp_point.cols = i/row_num;

			}
			
			else{//物体が視野角に収まっていない時
				for(size_t j=1;j<row_num;j++){
					if(i%row_num==j){
						for(size_t k=1;k<=j;k++){
							obserPoint[i-k].rows = k;
						}
					}
				}

				temp_point.ob_point = transCoordinate2(temp_point.ob_point,G_point, R,row_num);

				temp_point.rows=row_num;
				temp_point.change_flag = 1;
				temp_point.cols = i/row_num;
					

			}
			abs_obs_angle.clear();
			obs_angle.clear();
			sub_angle.clear();

		}
		if(i==row_num*N-1){
			sum_change_flag = temp_point.change_flag;
			for(size_t j=1;j<row_num;j++)
				sum_change_flag+=obserPoint[i-j].change_flag;
			if(sum_change_flag==0){
				temp_point.rows=1;
				for(size_t j=1;j<row_num;j++)
					obserPoint[i-j].rows = j+1;
			}
		}
	//	putText(voronoi_Image,temp_point.cols,Point( obserPoint[i].ob_point.x-7, obserPoint[i].ob_point.y+7),0,0.6,Scalar(0,0,0),2.0,cv::LINE_AA);


		/////////////////////////////////
		///////////評価値計算////////////
		/////////////////////////////////
		
		if(i==5000/*!field.contains(temp_point.ob_point)*/){
			temp_point.F = 0;	
		}
		else{
			preface_body.clear();
			temp_point.estimatepositionclu(tanob,G_point,zz);//zz//m////
			double func;
			////////////////////////////構図だけ///////////////////////
			double func1,eval_func;
			double f_kl;

			calKLD_multi2(preface_body,func1);
			if(func1<0)
				func1=0.1;
			func=func1;

			fout<<func<<endl;
			tkl.push_back(func);
			double mef;
			mef=func;
			eval_func=exp(-mef/3.0);

			temp_point.F = eval_func;
			temp_F.push_back(eval_func);

			preface_body.clear();
		
		}
		temp_point.dist = sqrt(double(((my_position.x)-temp_point.ob_point.x)*((my_position.x)-temp_point.ob_point.x)+((my_position.y)-temp_point.ob_point.y)*((my_position.y)-temp_point.ob_point.y)));
			obserPoint.push_back(temp_point);
	}
	
	fout.close();
//////////////////////////正規化F//////////////////////////////
	getSortOrder(temp_F,order3,false);/////////////from big to small

	string filename4 ="正規化.csv";
	ofstream fout4;
	fout4.open(filename4.c_str());
	for(size_t i=0; i<obserPoint.size();i++){
		obserPoint[i].F = (obserPoint[i].F-temp_F[order3[0]])/(temp_F[order3[temp_F.size()-1]]-temp_F[order3[0]]);
		fout4<<obserPoint[i].F<<endl;
	      
		yilunf.push_back(obserPoint[i].F);
	}
	    pingjunf.push_back(yilunf);
		yilunf.clear();
		fout4.close();
//////////////////////////////////////////////////////////////
	
	selectPoint_information(obserPoint,Goal_point);////////////評価値の最も高い最適地点を選ぶ
	//selectPoint_information2(obserPoint,Goal_point,Goal_point2);
	present_score=calcPresentScore(obserPoint);//現在地のスコアを計算

	for(size_t i=0; i<obserPoint.size();i++){///////////////色をつける
		obserPoint[i].color = hsv2rgb(obserPoint[i].F*119.9);
	}
	
	
	
	///////////////////////////voronoi_Imageへの書き込み/////////////////////////////////////

	for(size_t i=0;i<N*row_num;i++){
		ostringstream num;	
		ostringstream ok;
		ostringstream io;
		ostringstream yo;
		ostringstream to;
		ostringstream row;
		ostringstream col;
		ok.precision(4);
		io.precision(4);
		yo.precision(4);
		to.precision(4);
		row.precision(4);
		num.precision(4);
		col.precision(4);
		row<<obserPoint[i].rows;
		ok<<obserPoint[i].F ;
		col<<obserPoint[i].cols;
		num<<i;

	/*	if(obserPoint[i].ob_angle>60||obserPoint[i].ob_angle<-60)
			circle( voronoi_Image, obserPoint[i].ob_point, 7, obserPoint[i].color, cv::FILLED, 8, 0 );
		else*/
			circle( voronoi_Image, obserPoint[i].ob_point, 7, obserPoint[i].color, cv::FILLED, 8, 0 );
			
		putText(voronoi_Image,"C",G_point,0,0.7,Scalar(0,0,0),1.9,cv::LINE_AA);
	}
	circle( voronoi_Image, obserPoint[0].ob_point, 12, Scalar(0,100,200), cv::FILLED, 10, 0 );
	putText(voronoi_Image,"S",Point( obserPoint[0].ob_point.x-7, obserPoint[0].ob_point.y+7),0,0.6,Scalar(0,0,0),2.0,cv::LINE_AA);
	circle( voronoi_Image,Point( Goal_point.x,Goal_point.y),11, Scalar(50,150,50), cv::FILLED, 8, 0 );
	putText(voronoi_Image,"G",Point(Goal_point.x-7,Goal_point.y+7),0,0.6,Scalar(0,0,0),2.0,cv::LINE_AA);

	for(size_t i=0; i<6;i++){
		Scalar temp_color=hsv2rgb(120*i/5);
		rectangle(voronoi_Image, Point(0,120-20*i),Point(30,140-20*i), temp_color, -1, 8, 0 );
	}
	/////////////////////////////////////////////////////////////////////////////////////////

	return obserPoint[0].F;
	
}

int get_center_of_balancelan(vector<Person> obs)
{
	float w_x=0;
	float w_y=0;
	float w=0;
	int gpoint=0;
	for(size_t i=0;i<obs.size();i++)
	{
		w_x=w_x+obs[i].weight*obs[i].face_pixPoint.x;
		w=w+obs[i].weight;
	}
	gpoint=w_x/w;

	return gpoint;
}

