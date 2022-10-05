#include <opencv2/core.hpp>
//#include <opencv2/opencv.hpp>
#include <kl_evaluation/Object.h>
#include <kl_evaluation/Ransac.h>
#include <boost/math/special_functions/digamma.hpp>
using namespace std;
using namespace cv;
extern double k;
extern double kk;
extern int cishu;
extern vector<vector<double> > pingjunf;
extern vector<double> q;
extern vector<double> yilunf;
extern Point my_position;
extern Point2f Goalr;
//extern Point2f Goal_point;
extern vector<Object> preface_body;
using namespace std;
using namespace cv;
using namespace boost::math;
typedef vector <double> Vec1;
typedef vector <Vec1> Mtx;
#define SZ(a) ((int)((a).size()))
struct orderingSorter
{
    bool operator ()(std::pair<size_t, vector<double>::const_iterator> const& a, std::pair<size_t, vector<double>::const_iterator> const& b)
    {
        return (*a.second) > (*b.second);
    }
};
struct point{
    long int x;
    long int y;
};

Point3f extract3DPoint(Rect bb, Mat& pointCloudMap);
Point3f extract3DPointlan(Rect bb, Mat& pointCloudMap, Mat& mask);
void clac_weightlan(vector<Body_lan> &obj);
void get_center_of_balance(vector<TanObject> obs, Point2f& gpoint);
Point transRotCoordinate(Point before, Point center, double rot);
Point transCoordinate2(Point before, Point center, double R, int row_num);
Point transCoordinate(Point before, Point center ,double R);
double calcAngle(Point my_position, Point2f G_point, TanObject tanob);
void getSortOrder(const vector<double>& values, vector<size_t>& order, bool descending);
void calc_sub_angle(vector<double> angle, vector<double>& sub_angle);
Scalar hsv2rgb( float hue );
void cal_force(vector<Object> objects,double& result);
void apply_Filter(const unsigned int& type, const std::vector<double>& prv, const double& now, double& flt);
double Gaussian(double sigma, double x);
double Distancelan(Point2f p1, Point2f p2);
void selectPoint_information2(vector<ObservationPoint>& obserPoint,Point2f& Goal);
double calcPresentScore(vector<ObservationPoint>& obserPoint);
void DepthToGrayscale(Mat& pointCloudMap, Mat& depthMap);
void setPoints(Mat& ps, float depth_max_range, float depth_min_range);
double judgeOcclusion(vector<Object>& objects);
double judgeOcclusion2(vector<Body_lan>& objects);

void selectPoint_information(vector<ObservationPoint>& obserPoint,Point2f& Goal,Point2f& Goal2);
void evalFunctionlan21(Mat& voronoi_Image, vector<Body_lan>& objects,Point2f& G_point,Point2f& Goal_point,double& present_score,vector<double>& tkl);
void getSortOrder(const vector<double>& values, vector<size_t>& order, bool descending);
Mtx mplus(const Mtx& a, const Mtx& b);
Mtx mul_k(const Mtx& a, const Mtx& b);
Mtx mulScalar_k(const Mtx& a, const double scalar);
Vec1 mulScalar_k(const Vec1& a, const double scalar);
Vec1 mulVecMul_k(const Vec1& ve, const Mtx& mu);
double det_k(const Mtx& m);
Mtx transpose_k( const Mtx& vs );
Mtx solve_k(const Mtx& m );
void transfer(Object object,Vec1& mean, Mtx& vcm);
double trace(Mtx a);
double kl_calculation(Object object1,Object object2, double& kl);
void tab(Object a,Object b, double& tab_value);
void lowerbound(vector<Object> objects1,vector<Object> objects2,double& lower);
void upperbound(vector<Object> objects1,vector<Object> objects2,double& upper);
double kl_mean(vector<Object> objects1,vector<Object> objects2);
///////////////for cluster///////////////////////////////////
double calKLD_cluster(vector<Object> objects1,double& result);
double calKLD_multi(vector<Object> objects,double& result);
double calKLD_multi2(vector<Object> objects,double& result);

void clac_weightlanclu(vector<Person> &obj);
float evalFunctionlanclu(Mat& voronoi_Image, vector<Person>& objects,Point2f& G_point,Point2f& Goal_point,double& present_score, vector<double>& tkl);
int get_center_of_balancelan(vector<Person> obs);
