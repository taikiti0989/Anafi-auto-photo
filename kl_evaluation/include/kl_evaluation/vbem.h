#include <iostream>
#include <math.h>
#include <boost/math/special_functions/digamma.hpp>
// #include <cv.h>
#include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;
typedef vector <double> Vec1;
typedef vector <Vec1> Mtx;
#define SZ(a) ((int)((a).size()))

extern Vec1 alphae;
extern Vec1 betae;
extern Vec1 nyue;
extern Mtx me;
extern vector<Mtx> We;

using namespace std;
using namespace cv;
using namespace boost::math;

//yokomatsu
extern string file_data_free1;
extern ofstream fout_data_free1;
// Mtx gagamama;

struct orderingSorterEM
{
    bool operator ()(std::pair<size_t, vector<double>::const_iterator> const& a, std::pair<size_t, vector<double>::const_iterator> const& b)
    {
        return (*a.second) > (*b.second);
    }
};
class VBEM_GMM
{
private:
	int N; // 
	int D; // 
	int K; //
	Mtx m_xx;                 // 
	double alpha0;             //alpha: 
	double beta0;              //
	double nyu0;               //
	Vec1 m0;                    //
	Mtx W0;                    //    W[D][D]
	Mtx gamma_nk;       // 	gamma_nk[N][K]
	vector<double> alpha;
	vector<double> beta;
	vector<double> nyu;
	Mtx m;
	vector<Mtx> W;
public:

	void initialize(const Mtx& dataFrame, int numClasses)
	{
		N = SZ(dataFrame);
		//D = SZ(dataFrame[0]);
		D=2;
		K = numClasses;

		assert(N>=1);
		assert(D==2);
		assert(K>=1);

		
		m_xx = dataFrame;
		//normalize(m_xx);   
		normalize2(m_xx);  

		
		alpha0=0.001;
		beta0=0.1;
		nyu0=2;
		W0=Mtx(D,Vec1(D));
		for(int row=0;row<D;++row)
		{
			for(int col=0;col<D;++col)
			{
				if(row==col)
					W0[row][col]= 1; 
				else
					W0[row][col]=0;
			}
		}
		m0=Vec1(D);

			for(int col=0;col<D;++col)
			{
					m0[col]=0;
			}

	
		double a,b,v;
		Vec1 mm;
		mm.clear();
		mm=Vec1(D);
		Mtx WW;
		WW.clear();
		WW=Mtx(D,Vec1(D));
		
		alpha.clear();
		beta.clear();
		nyu.clear();
		m=Mtx(K,Vec1(D));
		W=vector<Mtx>(K,Mtx(D,Vec1(D)));
		m.clear();
		W.clear();
		for(int k=1;k<K+1;++k)
		{
			a=alpha0+N/K;
			alpha.push_back(a);
			b=beta0+N/K;
			beta.push_back(b);
			v=nyu0+N/K;
			nyu.push_back(v);
			mm.clear();
			mm.push_back(getNormRand());
		    mm.push_back(getNormRand());
			m.push_back(mm);
			WW=W0;
			W.push_back(WW);
		}
	}


	void initialize2(const Mtx& dataFrame, int numClasses,int flag)
	{
		N = SZ(dataFrame);
		D=2;
		K = numClasses;

		assert(N>=1);
		assert(D==2);
		assert(K>=1);

		
		m_xx = dataFrame;
		//normalize(m_xx);                        
		normalize(m_xx);//kesitaradame
		// fout_data_free1<<m_xx<<endl;
		//normalize3(m_xx);                       //

		//alpha, beta, nyu, m, 
		alpha0=0.001;
		beta0=0.1;
		nyu0=2;
		W0=Mtx(D,Vec1(D));
		for(int row=0;row<D;++row)
		{
			for(int col=0;col<D;++col)
			{
				if(row==col)
					W0[row][col]= 1; 
				else
					W0[row][col]=0;
			}
		}
		m0=Vec1(D);

			for(int col=0;col<D;++col)
			{
					m0[col]=0;
			}

	if(flag==1)
	{
		
		double a,b,v;
		Vec1 mm;
		mm.clear();
		mm=Vec1(D);
		Mtx WW;
		WW.clear();
		WW=Mtx(D,Vec1(D));
		
		alpha.clear();
		beta.clear();
		nyu.clear();
		m=Mtx(K,Vec1(D));
		W=vector<Mtx>(K,Mtx(D,Vec1(D)));
		m.clear();
		W.clear();
		for(int k=1;k<K+1;++k)
		{
			a=alpha0+N/K;
			alpha.push_back(a);
			b=beta0+N/K;
			beta.push_back(b);
			v=nyu0+N/K;
			nyu.push_back(v);
			mm.clear();
			mm.push_back(getNormRand());
		    mm.push_back(getNormRand());
			m.push_back(mm);
			WW=W0;
			W.push_back(WW);
		}
	}
	else
      {
		alpha=alphae; 
		beta=betae;
		nyu=nyue;
		m=me; 
		W=We;
	  }
	}

		
	void run(const int& numLoops,int& count)
	{
	
		Vec1 new_alpha;
		Vec1 new_beta;
		Vec1 new_nyu;
		Mtx new_m;
		vector<Mtx> new_W;	
		for (int i = 0; i < numLoops; i++)
		{
	
		gamma_nk.clear();
		// gagamama.clear();
		gamma_nk = calcEstep(m_xx, alpha, beta, nyu,m,W);
		// gagamama = gamma_nk;
		// fout_data_free1<<gagamama[0][0]<<endl;

		int groupsnum=0;
		for(int k=0;k<K;k++)
		{
			double N_km=0;
			for(int n=0;n<N;n++)
			{
				N_km=N_km+gamma_nk[n][k];
			}
			if(N_km!=0)
				groupsnum++;
		}
		
		calcMstep(new_alpha, new_beta, new_nyu, new_m,new_W,m_xx, gamma_nk);
	if(alpha[1]==new_alpha[1]&&alpha[2]==new_alpha[2]&&alpha[3]==new_alpha[3]&&alpha[0]==new_alpha[0])
		{
			alpha = new_alpha;
			beta  = new_beta;
			nyu   = new_nyu;
			m = new_m;
			W =  new_W;
			count=i+1;
			break;
		}	
		else
			{
			alpha = new_alpha;
			beta  = new_beta;
			nyu   = new_nyu;
			m = new_m;
			W =  new_W;
			count=i+1;
		}	
		}
		alphae = new_alpha;
		betae  = new_beta;
		nyue   = new_nyu;
		me = new_m;
		We =  new_W;
	}

	void printGammaNK(int& groups,vector<int>& which,vector<Point>& centers_lan,vector<Point>&sizes_lan,vector<int>& angles_lan, vector<double>& gggammma) const
	{

		groups=0;
		which.clear();
		for(int k=0;k<K;k++)
		{
			double N_km=0;
			for(int n=0;n<N;n++)
			{
				N_km=N_km+gamma_nk[n][k];
				gggammma.push_back(N_km);
			}
			//cout<<N_km<<endl;
			if(N_km!=0)
			{
				groups++;
				which.push_back(k);
				Point pts_lan;
			    	pts_lan.x=m[k][0]*120.0+428;
				pts_lan.y=m[k][1]*120.0+240;
				//pts_lan.y=480-pts_lan.y;
				centers_lan.push_back(pts_lan);

				Mtx xiefang;
				xiefang=W[k];
				double alk,blk;
				alk=xiefang[0][0];
				blk=xiefang[1][1];
		        	double alkk,blkk;
				alkk=sqrt(alk);
				blkk=sqrt(blk);
				double costh;
				costh=xiefang[1][0]/(alkk*blkk);
				double thetaa;
				thetaa=acos(costh);
				thetaa=thetaa/3.14*180;
				int an_lk;
				an_lk=(int)(thetaa);
				an_lk=-an_lk;
				angles_lan.push_back(an_lk);
				int chang_lan,kuan_lan;
				chang_lan=(int)(alkk*120*3);
				kuan_lan=(int)(blkk*120*3);
				Point sizes;
				sizes.x=chang_lan;
				sizes.y=kuan_lan;
				sizes_lan.push_back(sizes);
			}
		}
	}

	Mtx outputGamma(Mtx& output)
	{
		vector<double> linel;
		for(int n=0;n<N;n++)
		{
			linel.clear();
			for(int k=0;k<K;k++)
			{
				linel.push_back(gamma_nk[n][k]);
			}
			output.push_back(linel);
		}
		return output;
	}


private:
	// 
void normalize( Mtx& xx ) const
	{
		for (int col=0;col<D;++col)
		{
			double mean = 0.0;	
			for (int row=0;row<N;++row)
			{
				mean += xx[row][col];
			}
			mean /= N;

			double sd	= 0.0;	
			for (int row=0;row<N;++row)
			{
				const double diff = xx[row][col]-mean;
				sd += diff * diff;
			}
			sd	/= (N-1);	
			sd	= sqrt(sd);

		
			for (int row=0;row<N;++row)
			{
				xx[row][col] = (xx[row][col]-mean)/sd;
			}
		}
	}

void normalize2( Mtx& xx ) const
	{
		for (int row=0;row<N;++row)
		{
			xx[row][0]=xx[row][0]-320;
			xx[row][0]=xx[row][0]/120.0;
			xx[row][1]=xx[row][1]-240;
			xx[row][1]=xx[row][1]/120.0;
	}
}
void normalize3( Mtx& xx ) const
	{
		string filelan="x.txt";
		ofstream fout;
		fout.open(filelan.c_str());
		string filelan2="y.txt";
		ofstream fout2;
		fout2.open(filelan2.c_str());
	
		
		double meanx = 0.0;	//
		double meany = 0.0;  // 
		double sdx	= 0.0;	
		double sdy	= 0.0;	
		double varx=0.0;     
		double vary=0.0;     
		for (int col=0;col<D;++col)
		{
			if(col==0){
			for (int row=0;row<N;++row)
			{
				meanx += xx[row][col];
			}
			meanx /= N;
			}
			else
			{
				for (int row=0;row<N;++row)
				{
					meany += xx[row][col];
				}
				meany /= N;
			}

			if(col==0){
			for (int row=0;row<N;++row)
			{
				const double diff = xx[row][col]-meanx;
				sdx += diff * diff;
			}
			varx=sdx/N;      
			sdx	/= (N);	
			sdx	= sqrt(sdx);
			}
			else
			{
				for (int row=0;row<N;++row)
				{
					const double diff = xx[row][col]-meany;
					sdy += diff * diff;
				}
				vary=sdy/N;     
				sdy	/= (N);	
				sdy	= sqrt(sdy);
			}

		}

		double convar=0.0; 
		for (int row=0;row<N;++row)
		{
			convar=convar+(xx[row][0]-meanx)*(xx[row][1]-meany);
		}
		convar=convar/N;

	
		Mtx covar;
		covar=Mtx(D,Vec1(D));
		covar[0][0]=varx;
		covar[0][1]=convar;
		covar[1][0]=convar;
		covar[1][1]=vary;

	
	    if(convar!=0)
		{ 
	

 
		double a,b,c;
	
		a=varx;
		b=vary;
		c=convar;
		double delta;
		delta=(a-b)*(a-b)+4*c*c;
		double lam1,lam2;
		lam1=0.5*(a+b+sqrt(delta));
		lam2=0.5*(a+b-sqrt(delta));	
        	double m1,n1,m2,n2;
		m1=1;
		n1=(lam1-a)/c;
		m2=1;
		n2=(lam2-a)/c;
		Mtx S;
		S=Mtx(D,Vec1(D));
		double mn1,mn2;
		mn1=sqrt(m1*m1+n1*n1);
		mn2=sqrt(m2*m2+n2*n2);
		S[0][0]=m1/mn1;
		S[1][0]=n1/mn1;
		S[0][1]=m2/mn2;
		S[1][1]=n2/mn2;
		Mtx St;
		St=transpose(S);

		Mtx la;
		la=Mtx(D,Vec1(D));
		la[0][0]=sqrt(lam1);
		la[1][1]=sqrt(lam2);
		la[0][1]=0.0;
		la[1][0]=0.0;
		Mtx lak;
		lak=Mtx(D,Vec1(D));
		lak[0][0]=1.0/sqrt(lam1);
		lak[1][1]=1.0/sqrt(lam2);
		lak[0][1]=0.0;
		lak[1][0]=0.0;
		Mtx midle;
		midle=mul(lak,St);
		double a00,b00;
		for(int row=0;row<N;++row)
		{
			a00=xx[row][0];
			b00=xx[row][1];
		
			xx[row][0]=midle[0][0]*(a00-meanx)+midle[0][1]*(b00-meany);
			xx[row][1]=midle[1][0]*(a00-meanx)+midle[1][1]*(b00-meany);
		}

	
		double a0,b0;
		for(int row=0;row<N;++row)
		{
			a0=xx[row][0];
			b0=xx[row][1];
			xx[row][0]=a0*S[0][0]+b0*S[0][1];
			xx[row][1]=a0*S[1][0]+b0*S[1][1];
		}
		}
		for(int row=0;row<N;++row)
		{
			fout<<xx[row][0]<<endl;
			fout2<<xx[row][1]<<endl;
		}
		fout.close();
		fout2.close();
	}
	
	inline Mtx mul(const Mtx& a, const Mtx& b) const
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


	inline Mtx mulScalar(const Mtx& a, const double scalar) const
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


	inline Vec1 mulScalar(const Vec1& a, const double scalar) const
	{
		Vec1 ret(a);
		for (int i = 0; i < SZ(ret); i++)
		{
			ret[i] *= scalar;
		}
		return ret;
	}


	inline Vec1 mulVecMul(const Vec1& ve, const Mtx& mu) const
	{
	    Vec1 re;
		re.clear();
	    double value0,value1;
		value0=ve[0]*mu[0][0]+ve[1]*mu[1][0];
		value1=ve[0]*mu[0][1]+ve[1]*mu[1][1];
		// value0=ve[0]*mu[0][0]+ve[1]*mu[0][1];
		// value1=ve[0]*mu[1][0]+ve[1]*mu[1][1];
		re.push_back(value0);
		re.push_back(value1);

		return re;
	}

	inline Mtx transpose( const Mtx& vs ) const
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

	//
	inline double det(const Mtx& m) const
	{
		return m[0][0]*m[1][1]-m[0][1]*m[1][0];
	}

	// 
	inline Mtx solve(const Mtx& m ) const
	{
		vector < vector <double> > ret(m);
		swap(ret[0][0],ret[1][1]);
		ret[0][1] = -ret[0][1];
		ret[1][0] = -ret[1][0];
		ret = mulScalar(ret,1.0/abs(det(m)));
		return ret;
	}

	// 
	double getDMNorm(const Vec1& x, const Vec1& mu, const Mtx& sig) const
	{
		Vec1 x_mu(x);	// x - mu
		for (int i = 0; i < SZ(x_mu); i++)
		{
			x_mu[i] -= mu[i];
		}

		const Mtx inv = solve(sig);

		// 
		const double C = x_mu[0]*(x_mu[0]*inv[0][0]+x_mu[1]*inv[1][0]) + x_mu[1]*(x_mu[0]*inv[0][1]+x_mu[1]*inv[1][1]);

		double ret = 1/(sqrt(pow(2.0*M_PI,D) * det(sig))) * exp( -0.5 * C );

		return ret;
	}

	//
	Mtx calcEstep( const Mtx& xx, const vector<double> &Alpha,const vector<double> &Beta,const vector<double> &Nyu,const Mtx &M, const vector<Mtx> &Ww) const
	{
		Mtx ret;
		ret.clear();
		//(10.65)
		Vec1 ln_lambda;
		Vec1 ln_pi;
		ln_lambda.clear();
		ln_pi.clear();
		double lnlam;
		double lnpi;
		double dg_x;
		double sum_alpha=0;
		for(int k=0;k<K;k++)
		{
			sum_alpha=sum_alpha+Alpha[k];
		}

		for(int k=0;k<K;k++)
		{
			double sum_digam=0;
			for(int i=1;i<D+1;i++)
			{
				dg_x=(Nyu[k]+1-i)*0.5;
				sum_digam=sum_digam+digamma(dg_x);
			}
			lnlam=sum_digam+D*log(2.0)+log(det(Ww[k]));
			ln_lambda.push_back(lnlam);                                         ////////////(10.65)
			lnpi=digamma(Alpha[k])-digamma(sum_alpha);
			ln_pi.push_back(lnpi);                                                    ////////////
		}

		
		Vec1 mid;                       
		double midd;
		Vec1 mimu;
		double exp_n;
		double exp_sum;
		
		
		for(int row=0;row<N;++row)
		{
			vector<double> number(K);
			number.clear();
			double sum_rnk=0;
			for(int k=0;k<K;k++)
			{
				mid.clear();
				for(int col=0;col<D;col++)
				{
					midd=xx[row][col]-M[k][col];
					mid.push_back(midd);              
				}
				mimu.clear();
				mimu=mulVecMul(mid,Ww[k]);
			    exp_n=mimu[0]*mid[0]+mimu[1]*mid[1];
				exp_sum=ln_lambda[k]/2+ln_pi[k]-D/(2*Beta[k])-exp_n*Nyu[k]/2;
				exp_sum=exp(exp_sum);
				sum_rnk=sum_rnk+exp_sum;
				number.push_back(exp_sum);
			}
			number=mulScalar(number,1.0/sum_rnk);
			ret.push_back(number);
		}

		return ret;
	}

	
	void calcMstep(Vec1& new_alpha,Vec1& new_beta, Vec1& new_nyu, Mtx& new_m, vector < Mtx >& new_W, const Mtx& xx, const Mtx& gamma_nk) const	
	{
		new_alpha.clear();
		new_nyu.clear();
		new_beta.clear();
		new_m.clear();
		new_W.clear();

		
		vector<double> N_k(K);    
		Mtx x_ba(K,Vec1(D));    
		vector<Mtx> S;
		S=vector<Mtx>(K,Mtx(D,Vec1(D)));
		N_k.clear();
		x_ba.clear();
		//x_ba=Mtx(K,Vec1(D));
		//N_s
		double n_s;
		Point2f pt;
		for(int k=0;k<K;k++)
		{
			double sum_n=0;
			double sum_xx=0;
			double sum_xy=0;
			Vec1 x_k(D);
			x_k.clear();
			for(int n=0;n<N;n++)
			{
				sum_n=sum_n+gamma_nk[n][k];
				sum_xx=sum_xx+xx[n][0]*gamma_nk[n][k];
				sum_xy=sum_xy+xx[n][1]*gamma_nk[n][k];
			}

			n_s=sum_n;  /////N_k
			if(sum_n!=0.0)
			{
				x_k.push_back(sum_xx/sum_n);//x_k_x
				x_k.push_back(sum_xy/sum_n);//x_k_y
			}
			else
			{
				x_k.push_back(0.0);//x_k_x
				x_k.push_back(0.0);//x_k_y
			}
			N_k.push_back(n_s);      
			x_ba.push_back(x_k);
		}
			//S_k
		for(int k=0;k<K;k++)
		{
			for(int n=0;n<N;n++)
			{
				pt.x=xx[n][0]-x_ba[k][0];
				pt.y=xx[n][1]-x_ba[k][1];

				S[k][0][0]=S[k][0][0]+gamma_nk[n][k]*pt.x*pt.x;
				S[k][0][1]=S[k][0][1]+gamma_nk[n][k]*pt.x*pt.y;
				S[k][1][0]=S[k][1][0]+gamma_nk[n][k]*pt.y*pt.x;
				S[k][1][1]=S[k][1][1]+gamma_nk[n][k]*pt.y*pt.y;
			}
			if(N_k[k]!=0)
				S[k] = mulScalar(S[k], 1.0/ N_k[k]);
			else
				S[k] = mulScalar(S[k], N_k[k]);
		}


		
		double alpha_k;
		double beta_k;
		double nyu_k;
		Vec1 m_k(D);
	
		Vec1 m_k1;
		Vec1 m_k2;
		for(int k=0;k<K;k++)
		{
			alpha_k=alpha0+N_k[k];
			beta_k=beta0+N_k[k];
			nyu_k=nyu0+N_k[k];
			new_alpha.push_back(alpha_k);
			new_beta.push_back(beta_k);
			new_nyu.push_back(nyu_k);
		}
		for(int k=0;k<K;k++)
		{
			m_k.clear();
			m_k1.clear();
			m_k2.clear();
			double m_kx;
			double m_ky;
			m_k1=mulScalar(m0,beta0);
			m_k2=mulScalar(x_ba[k],N_k[k]);
			m_kx=m_k1[0]+m_k2[0];
			m_ky=m_k1[1]+m_k2[1];
			m_k.push_back(m_kx);
			m_k.push_back(m_ky);
			m_k=mulScalar(m_k,1.0/new_beta[k]);
			new_m.push_back(m_k);
		}

		Mtx W_ki;
		W_ki=Mtx(D,Vec1(D));
		Point2f pts;
		Mtx W_0i;
		W_0i=Mtx(D,Vec1(D));
		Mtx W_k;
		W_k=Mtx(D,Vec1(D));
		for(int k=0;k<K;k++)
		{
			pts.x=x_ba[k][0]-m0[0];
			pts.y=x_ba[k][1]-m0[1];

			W_ki[0][0]=pts.x*pts.x;
			W_ki[0][1]=pts.x*pts.y;
			W_ki[1][0]=pts.y*pts.x;
			W_ki[1][1]=pts.y*pts.y;
			W_ki=mulScalar(W_ki,beta0*N_k[k]/(beta0+N_k[k]));
			W_0i=solve(W0);
			W_ki[0][0]=W_ki[0][0]+W_0i[0][0]+N_k[k]*S[k][0][0];
			W_ki[0][1]=W_ki[0][1]+W_0i[0][1]+N_k[k]*S[k][1][0];
			W_ki[1][0]=W_ki[1][0]+W_0i[1][0]+N_k[k]*S[k][0][1];
			W_ki[1][1]=W_ki[1][1]+W_0i[1][1]+N_k[k]*S[k][1][1];

			W_k=solve(W_ki);
			new_W.push_back(W_k);
		}

	}

	// 
	double getNormRand() const
	{
		double ret = 0.0;
		for( int i = 0; i < 12;i++ ){
			ret += (double)rand()/RAND_MAX;
		}
		return ret-6.0;
	}

};

void getSortOrderEM(const vector<double>& values, vector<size_t>& order, bool descending);
