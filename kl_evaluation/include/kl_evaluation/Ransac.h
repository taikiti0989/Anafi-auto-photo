#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/persistence.hpp>
using namespace std;
using namespace cv;
class Ransac {
        // This class performs Ransac to find planes in a point cloud.
        // To do so, the Inlier Distance, the iterations and the points need to be set.
        // It will return some basic information about the plane.
        // The class is also able to convert a list of 0s and 1s into a list of numbers, where the new list contains the position of every elment that had a 1 before.
        // It also prints the elements into a file 


public:
        Ransac();
        void RansacFinishProcess();
        void setInlierDistance(float);
        void setIterations(int);
	void setSamplingRatio(float);
        void setPoints(Mat&, float, float);
        void findBestPlane(int, vector<int>&, Point3f&, Point3f *);
	void removePlane(vector<int>& inlierIndices);
	int  getLeftPoints(){return pointsLeft;}

	 Mat point_cloud;							// Point Cloud Map 

private:
        float inlierDistance;				// The distance between the plane spanned and a point, which will cause the point to be considered an inlier.
        int iterations;						// the number of iterations that Ransac should be performed
        Mat depth;							// Point Cloud Map 
	vector<Point3f> pointCloud;			// contains all point cloud
	vector<Point2i> pointIndices;		// 
        int *pointsUsed;		// contains information on which points have been used. 1 for has been used, 0 otherwise
		//int *inlierIndices;					// contains inlier points
		int maxInliers;
        int numberOfPoints;					// the number of points in the point cloud
        int pointsLeft;						// the number of points that have not been allocated to a plane yet
		float samplingRatio;				// randam sampling ratio

};


