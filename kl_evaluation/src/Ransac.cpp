#include "kl_evaluation/Ransac.h"
Ransac::Ransac(){

        //intialise random number generator
        //srand(time(0));

}

void Ransac::RansacFinishProcess(){
        // free the memory of all the memory acquired by this class
        delete[] pointsUsed;
		//delete[] inlierIndices;
}

void Ransac::setInlierDistance(float dis) {
        // sets the distance from the plane when a point will still be considered an inlier
        inlierDistance = dis;
}

void Ransac::setPoints(Mat& ps, float depth_max_range, float depth_min_range) {
        // sets the pointCloud variable and also initialises points used to all zeros
        // sets the number of points
        depth = ps;

		int c,r;
		r = depth.rows;
		c = depth.cols;
		
		
		Point3f p0(depth_max_range+0.3f,depth_max_range+0.3f,depth_max_range+0.3f);	
		pointCloud.clear();
		pointIndices.clear();
		for(int y=0;y<r;y++){
				for(int x=0;x<c;x++){
					Point3f &point = ((Point3f*)(depth.data+depth.step.p[0]*y))[x];
				
					if(point.z != 0 && point.z > depth_min_range && point.z < depth_max_range){
						pointIndices.push_back(Point(x,y));
						pointCloud.push_back(point);
					}else
						point = p0;
				}
			}
		numberOfPoints = pointCloud.size();

        pointsUsed = new int[numberOfPoints];
        for (int i = 0; i < numberOfPoints; i++) {
                pointsUsed[i] = 0;
        }
        pointsLeft = numberOfPoints;
}

void Ransac::setSamplingRatio(float rt){
	// sets random sampling ratio
	samplingRatio = rt;
}

void Ransac::setIterations(int is) {
        // sets the number of iterations Ransac should take to find the best plane
        iterations = is;
}

void Ransac::findBestPlane(int planeNumber, vector<int>& inlierIndices, Point3f &normal, Point3f *bestPoints) {
    //finds the best plane from the sampled point cloud
    //input: the plane number
    //output: the index of inlier points, the planes normal
    //iterations = pointsLeft / 30;


	//random sampling
	vector<size_t> leftPointsIndices;
	for(int i=0;i<numberOfPoints;++i){
		if(pointsUsed[i] == 0)
			leftPointsIndices.push_back(i);
	}

	RNG& rng = theRNG();
	vector<Point3f> sampledPointCloud;	//
	int numberOfsampledPoints;             // the number of points in the sampled point cloud
	int pointsToExtract = static_cast<int>(samplingRatio*static_cast<float>(pointsLeft));	//
	vector<char> usedMask(pointsLeft, false);
	fill(usedMask.end()-pointsToExtract,usedMask.end(),true);
	//random_shuffle(usedMask.begin(),usedMask.end());
	for(int i=0;i<pointsLeft;i++)
	{
		int i1 = rng(pointsLeft),i2 = rng(pointsLeft);
		char tmp= usedMask[i1]; usedMask[i1] = usedMask[i2]; usedMask[i2] = tmp;
	}
	sampledPointCloud.clear();
	for(int i=0; i<pointsLeft; i++)
	{
		if(usedMask[i] )
			sampledPointCloud.push_back(pointCloud[leftPointsIndices[i]]);
	}
	numberOfsampledPoints = sampledPointCloud.size();

	
	//find a plane from the sampled point cloud
    int inlier = 0;
	maxInliers = 0;
    int pos1, pos2, pos3;
    Point3f P1,  P2, P3, P;
    float A,B,C,D;
    float dis;
    for (int i=0; i < iterations; i++) {
            // get Random Points
            pos1 = rng(numberOfsampledPoints);
            pos2 = rng(numberOfsampledPoints);
            pos3 = rng(numberOfsampledPoints);
            // ensure all points are different and that they have not been used before
            while ( pos1 == pos2 || pos2 == pos3 || pos1 == pos3 ) {
                    pos1 = rng( numberOfsampledPoints);
					pos2 = rng( numberOfsampledPoints);
					pos3 = rng( numberOfsampledPoints);	
            }
            P1 = sampledPointCloud[pos1];
			P2 = sampledPointCloud[pos2];
			P3 = sampledPointCloud[pos3];

            // calculate A, B, C and D, for the plane spanned by points pos1, pos2 and pos 3
			A = P1.y * (P2.z - P3.z) + P2.y * (P3.z - P1.z) + P3.y * (P1.z - P2.z);
            B = P1.z * (P2.x - P3.x) + P2.z * (P3.x - P1.x) + P3.z * (P1.x - P2.x);
            C = P1.x * (P2.y - P3.y) + P2.x * (P3.y - P1.y) + P3.x * (P1.y - P2.y);
            D = -(P1.x * (P2.y * P3.z - P3.y * P2.z) + P2.x * (P3.y * P1.z - P1.y* P3.z) + P3.x * (P1.y * P2.z - P2.y * P1.z));
                
            // set inliers to 0
            inlier = 0;
    
            for (int j = 0; j < numberOfsampledPoints; j++) {
                    // calculate distance between point and plane
                    P = sampledPointCloud[j];
                    // calculate the distance between the point and the plane
					if(P.z !=0){
						dis = abs(A*P.x + B*P.y + C*P.z + D) / sqrt(A*A + B*B + C*C);
          
						if (dis < inlierDistance){
								inlier++;
						}
					}
            }
            // if the inlier number is better than the best plane found so far, then we take these points to be considered the new best plane
            if (inlier > maxInliers) {
                    maxInliers = inlier;
                    bestPoints[0] = P1;
                    bestPoints[1] = P2;
                    bestPoints[2] = P3;
            }
                
    }


    // use the best points, recreate the plane
    P1 = bestPoints[0];
    P2 = bestPoints[1];
    P3 = bestPoints[2];
	A = P1.y * (P2.z - P3.z) + P2.y * (P3.z - P1.z) + P3.y * (P1.z - P2.z);
	B = P1.z * (P2.x - P3.x) + P2.z * (P3.x - P1.x) + P3.z * (P1.x - P2.x);
	C = P1.x * (P2.y - P3.y) + P2.x * (P3.y - P1.y) + P3.x * (P1.y - P2.y);
	D = -(P1.x * (P2.y * P3.z - P3.y * P2.z) + P2.x * (P3.y * P1.z - P1.y* P3.z) + P3.x * (P1.y * P2.z - P2.y * P1.z));
    
        
	int pointsPlaced = 0;
    for (int i = 0; i < pointsLeft; i++) {
        // calculate distance between point and plane
		int j = leftPointsIndices[i];
        P = pointCloud[j];
		float dis = abs(A*P.x + B*P.y + C*P.z + D) / sqrt(A*A + B*B + C*C);
                
		// find all inliers and mark them
		if (dis < inlierDistance){
			inlierIndices.push_back(j);
			pointsPlaced++;
			pointsUsed[j] = planeNumber;
		}
				 
    }	
	maxInliers = pointsPlaced;
    pointsLeft -= maxInliers;
        
    // calculate the normal of the plane
	normal = (P1-P2).cross(P3-P1);
		
}


void Ransac::removePlane(vector<int>& inlierIndices)
{
	Point3f p0(4.3f,4.3f,4.3f);
	int y, x;
	for(size_t i=0;i<inlierIndices.size();i++){
		y=pointIndices[inlierIndices[i]].y;
		x=pointIndices[inlierIndices[i]].x;
		Point3f &point = ((Point3f*)(depth.data+depth.step.p[0]*y))[x];
		point = p0;
	}

}
