//using pgi compiler (openacc) on linux workstation with GPU (NVIDIA GeForce GTX 1080 Ti)
//following compile option is used (mt19937ar.cc and mt19937ar.h are for Mersenne twister)
//pgc++ macropinocytosis_Fig2.cc mt19937ar.cc -acc -Minfo -ta=tesla,cc60 -fast -Bstatic_pgi
//run: "a.out > data.dat"


#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <iostream>
#include "mt19937ar.h"
#include "algorithm"
#include <algorithm>
#include <vector>
using namespace std;


#define pi 3.1415926536

//system size
#define L 250
#define Lz 430

//spatial and temporal resolutions
#define dx 0.1
#define dt 0.0004
#define cutoff 0.001

//parameters for membrane 
#define ep 0.8
#define eta 0.5
#define Mem 5.0
#define r0 10.0 //
#define beta_sig 100.0 //
#define theta_sig 0.105 //
#define tau 10.0

//parameters for protruding force
#define F0 2.0
#define K 0.005
#define Kd 0.25

//parameters for reaction-diffusion
#define Da 0.1
#define Di 0.01
#define ki1 0
#define ki2 0
#define at 2.8
#define alpha 1.0 // Use a negative alpha value for alpha=infty (Eq.(8))
#define gamma2 10.0
#define xmax 50.0 //upper limit of the concentration


//parameters for the initial condition
#define init_amp 5.0
#define init_rng 15
#define seed 4

//noise parameters
#define sigma 7.0
#define theta 0.0

//total iteration is maxT*maxTT
//int maxT=500;
int maxT=50;
#define maxTT 10000


//definition of grobal variables
double vol; //cell volume 
double area;//cell surface area
double intA;//total amount of A divided by cell surface area
double at_n; // at_n is normalized value of at and defined as at_n=at*4pi*R0*R0/area, where area is total surface area of the cell


//definition of local variables
int SIZE=L*L*Lz;
double* ph=(double*)std::malloc(sizeof(double)*(SIZE)); //phi
double* phn=(double*)std::malloc(sizeof(double)*(SIZE));
double* ps=(double*)std::malloc(sizeof(double)*(SIZE)); //psi
double* psn=(double*)std::malloc(sizeof(double)*(SIZE));
double* A=(double*)std::malloc(sizeof(double)*(SIZE)); //concentration of 'A' molecule 
double* An=(double*)std::malloc(sizeof(double)*(SIZE));
double* I=(double*)std::malloc(sizeof(double)*(SIZE));// concentration of 'I' molecule 
double* In=(double*)std::malloc(sizeof(double)*(SIZE));
double* vamp=(double*)std::malloc(sizeof(double)*(SIZE));//the amplitude of the velocity
double* vx=(double*)std::malloc(sizeof(double)*(SIZE));// x component of the velocity vector
double* vy=(double*)std::malloc(sizeof(double)*(SIZE));// y component of the velocity vector
double* vz=(double*)std::malloc(sizeof(double)*(SIZE));// z component of the velocity vector



//function for caluclating psi from phi
double sigmoid(double phi){ 
  return 1.0/(1.0+exp(-beta_sig*(phi-theta_sig)));
}


double Fpoly(double x1,double x2, double y1,double y2){
  int nh=3;
  double ret=0;
  if(x1>0){
    ret=(pow(x1,nh)/(pow(y1,nh)+pow(x1,nh)))*(pow(y2,nh)/(pow(y2,nh)+pow(x2,nh)));
  }  
  return ret;
}


//functions for spatial differentiation
double d1th_x(double *x, int i,int j,int k){
  return (x[(i+1)*L*Lz+j*Lz+k]-x[(i-1)*L*Lz+j*Lz+k])/(2.0*dx);
}
double d1th_y(double *x, int i,int j, int k){
  return (x[i*L*Lz+(j+1)*Lz+k]-x[i*L*Lz+(j-1)*Lz+k])/(2.0*dx);
}
double d1th_z(double *x, int i,int j, int k){
  return (x[i*L*Lz+j*Lz+k+1]-x[i*L*Lz+j*Lz+k-1])/(2.0*dx);
}
double d1th_x_3v(double *x,double *y,double *z, int i,int j, int k){
  return (x[(i+1)*L*Lz+j*Lz+k]*y[(i+1)*L*Lz+j*Lz+k]*z[(i+1)*L*Lz+j*Lz+k]-x[(i-1)*L*Lz+j*Lz+k]*y[(i-1)*L*Lz+j*Lz+k]*z[(i-1)*L*Lz+j*Lz+k])/(2.0*dx);
}
double d1th_y_3v(double *x,double *y,double *z, int i,int j,int k){
  return (x[i*L*Lz+(j+1)*Lz+k]*y[i*L*Lz+(j+1)*Lz+k]*z[i*L*Lz+(j+1)*Lz+k]-x[i*L*Lz+(j-1)*Lz+k]*y[i*L*Lz+(j-1)*Lz+k]*z[i*L*Lz+(j-1)*Lz+k])/(2.0*dx);
}
double d1th_z_3v(double *x,double *y,double *z, int i,int j,int k){
  return (x[i*L*Lz+j*Lz+k+1]*y[i*L*Lz+j*Lz+k+1]*z[i*L*Lz+j*Lz+k+1]-x[i*L*Lz+j*Lz+k-1]*y[i*L*Lz+j*Lz+k-1]*z[i*L*Lz+j*Lz+k-1])/(2.0*dx);
}
double d1th_abs(double *x, int i,int j,int k){
  double DX=d1th_x(x,i,j,k);
  double DY=d1th_y(x,i,j,k);
  double DZ=d1th_z(x,i,j,k);
  return sqrt(DX*DX+DY*DY+DZ*DZ);
}
double d2th(double *x, int i, int j,int k){
 return (x[(i+1)*L*Lz+j*Lz+k]+x[(i-1)*L*Lz+j*Lz+k]+x[(i)*L*Lz+(j+1)*Lz+k]+x[(i)*L*Lz+(j-1)*Lz+k]+x[i*L*Lz+j*Lz+k+1]+x[i*L*Lz+j*Lz+k-1]-6.0*x[i*L*Lz+j*Lz+k])/(dx*dx);
}


//function for calulating Eq.1
double func_vamp(double *AA,double *II,double *p, int i, int j,int k, double VOL){
  double V0=  4.0*pi*r0*r0*r0/3.0;
  return (eta*(d2th(p,i,j,k)-16.0*p[i*L*Lz+j*Lz+k]*(1.0-p[i*L*Lz+j*Lz+k])*(1.0-2.0*p[i*L*Lz+j*Lz+k])/(ep*ep))+(-Mem*(VOL-V0)+F0*Fpoly(AA[i*L*Lz+j*Lz+k],((ki1>0) ? AA[i*L*Lz+j*Lz+k]*AA[i*L*Lz+j*Lz+k]*ki1/ki2:AA[i*L*Lz+j*Lz+k]*AA[i*L*Lz+j*Lz+k]),K,Kd) )*d1th_abs(p,i,j,k))/tau;

}

//functions for calulating reaction-diffusion
double func_A(double *AA,double *II,double AT, double inta,int i, int j,int k){
  if(alpha>0){return  AA[i*L*Lz+j*Lz+k]*AA[i*L*Lz+j*Lz+k]*(AT-inta)/((1.0+II[i*L*Lz+j*Lz+k])*(1.0+AA[i*L*Lz+j*Lz+k]*AA[i*L*Lz+j*Lz+k]/(alpha*alpha)))-AA[i*L*Lz+j*Lz+k];}
  else{return  AA[i*L*Lz+j*Lz+k]*AA[i*L*Lz+j*Lz+k]*(AT-inta)/((1.0+II[i*L*Lz+j*Lz+k]))-AA[i*L*Lz+j*Lz+k];}
}
double func_I(double *AA,double *II, int i, int j, int k){
  return ki1*AA[i*L*Lz+j*Lz+k]*AA[i*L*Lz+j*Lz+k]-ki2*II[i*L*Lz+j*Lz+k];
}



//functions for boundary condition (periodic B.C.)
void BC_ph(double *x){

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int i=0;i<L;++i){
   for(int j=0;j<L;++j){
     x[i*L*Lz+j*Lz+0]=x[i*L*Lz+j*Lz+Lz-2];
     x[i*L*Lz+j*Lz+Lz-1]=x[i*L*Lz+j*Lz+1];
   }
 }

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2) 
 for(int j=0;j<L;++j){
   for(int k=0;k<Lz;++k){
     x[0*L*Lz+j*Lz+k]=x[(L-2)*L*Lz+j*Lz+k];
     x[(L-1)*L*Lz+j*Lz+k]=x[(1)*L*Lz+j*Lz+k];
   }
 }

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int i=0;i<L;++i){
   for(int k=0;k<Lz;++k){
     x[i*L*Lz+0*Lz+k]=x[i*L*Lz+(L-2)*Lz+k];
     x[i*L*Lz+(L-1)*Lz+k]=x[i*L*Lz+1*Lz+k];
   }
 }

}

void BC(double *x){
#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int i=0;i<L;++i){
   for(int j=0;j<L;++j){
     x[i*L*Lz+j*Lz+0]   =x[i*L*Lz+j*Lz+Lz-2];
     x[i*L*Lz+j*Lz+Lz-1]=x[i*L*Lz+j*Lz+1];
   }
 }

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int j=0;j<L;++j){
   for(int k=0;k<Lz;++k){
     x[0*L*Lz+j*Lz+k]=x[(L-2)*L*Lz+j*Lz+k];
     x[(L-1)*L*Lz+j*Lz+k]=x[(1)*L*Lz+j*Lz+k];
   }
 }

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int i=0;i<L;++i){
   for(int k=0;k<Lz;++k){
     x[i*L*Lz+0*Lz+k]=x[i*L*Lz+(L-2)*Lz+k];
     x[i*L*Lz+(L-1)*Lz+k]=x[i*L*Lz+1*Lz+k];
   }
 }

}


//functions for update variables
void update_ph(double *x, double *xn){
#pragma acc parallel vector_length(256)
#pragma acc loop collapse(3)
  for(int i=1;i<L-1;++i){
    for(int j=1;j<L-1;++j){
      for(int k=1;k<Lz-1;++k){
	x[i*L*Lz+j*Lz+k]=(xn[i*L*Lz+j*Lz+k]>=0)? xn[i*L*Lz+j*Lz+k]:0;
      }
    }
  }
  

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int i=0;i<L;++i){
   for(int j=0;j<L;++j){
     x[i*L*Lz+j*Lz+0]   =x[i*L*Lz+j*Lz+Lz-2];
     x[i*L*Lz+j*Lz+Lz-1]=x[i*L*Lz+j*Lz+1];
   }
 }

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2) 
 for(int j=0;j<L;++j){
   for(int k=0;k<Lz;++k){
     x[0*L*Lz+j*Lz+k]=x[(L-2)*L*Lz+j*Lz+k];
     x[(L-1)*L*Lz+j*Lz+k]=x[(1)*L*Lz+j*Lz+k];
   }
 }

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int i=0;i<L;++i){
   for(int k=0;k<Lz;++k){
     x[i*L*Lz+0*Lz+k]=x[i*L*Lz+(L-2)*Lz+k];
     x[i*L*Lz+(L-1)*Lz+k]=x[i*L*Lz+1*Lz+k];
   }
 }
  
}

void update(double *x, double *xn){
#pragma acc parallel vector_length(256)
#pragma acc loop collapse(3)
  for(int i=0;i<L;++i){
    for(int j=0;j<L;++j){
      for(int k=0;k<Lz;++k){
	x[i*L*Lz+j*Lz+k]=(xn[i*L*Lz+j*Lz+k]>=0)? xn[i*L*Lz+j*Lz+k]:0;
	if(xn[i*L*Lz+j*Lz+k]>xmax){x[i*L*Lz+j*Lz+k]=xmax;}
      }
    }
  }
#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int i=0;i<L;++i){
   for(int j=0;j<L;++j){
     x[i*L*Lz+j*Lz+0]   =x[i*L*Lz+j*Lz+Lz-2];
     x[i*L*Lz+j*Lz+Lz-1]=x[i*L*Lz+j*Lz+1];
   }
 }

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int j=0;j<L;++j){
   for(int k=0;k<Lz;++k){
     x[0*L*Lz+j*Lz+k]=x[(L-2)*L*Lz+j*Lz+k];
     x[(L-1)*L*Lz+j*Lz+k]=x[(1)*L*Lz+j*Lz+k];
   }
 }

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(2)
 for(int i=0;i<L;++i){
   for(int k=0;k<Lz;++k){
     x[i*L*Lz+0*Lz+k]=x[i*L*Lz+(L-2)*Lz+k];
     x[i*L*Lz+(L-1)*Lz+k]=x[i*L*Lz+1*Lz+k];
   }
 }

}



//initial condition
void set_init(){

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(3)  
  for(int i=0;i<L;++i){
    for(int j=0;j<L;++j){
      for(int k=0;k<Lz;++k){
	double r1=sqrt((i-L/2)*(i-L/2)/(r0*r0)+(j-L/2)*(j-L/2)/(r0*r0)+(k-Lz/2)*(k-Lz/2)/(r0*r0))*dx;
	ph[i*L*Lz+j*Lz+k]=(1.0+tanh(2.0*(r0*(1.0-r1))/(ep)))/2.0;
	A[i*L*Lz+j*Lz+k]=0;
	I[i*L*Lz+j*Lz+k]=0;
      }
    }
  }


#pragma acc parallel loop
  for(int i=1;i<L-1;++i){
#pragma acc loop independent
    for(int j=1;j<L-1;++j){
#pragma acc loop independent
      for(int k=1;k<Lz-1;++k){
	ps[i*L*Lz+j*Lz+k]=sigmoid(ph[i*L*Lz+j*Lz+k]*(1.0-ph[i*L*Lz+j*Lz+k]));
      }
    }
  }

  BC(ps);

  vol=0;
  intA=0;
  area=0;
  
#pragma acc parallel vector_length(256)
#pragma acc loop collapse(3)  
  for(int i=1;i<L-1;++i){
    for(int j=1;j<L-1;++j){
      for(int k=1;k<Lz-1;++k){
	vol+=ph[i*L*Lz+j*Lz+k]*dx*dx*dx;
	area+=ps[i*L*Lz+j*Lz+k]*dx*dx*dx/ep;
	intA+=ps[i*L*Lz+j*Lz+k]*A[i*L*Lz+j*Lz+k]*dx*dx*dx;
      }
    }
  }


  
}

void paramet_logs(){
  /*
  time_t t = time(NULL);
  char filename[50];
  sprintf(filename,"%s/log_code/%s_%ld",getenv("HOME"),__FILE__,t);
    FILE *fp,*fpw;
  fpw = fopen(filename, "w");
  fp = fopen(__FILE__, "r");
  char cha; 
  while( ( cha = fgetc(fp) ) != EOF ) {
    fprintf(fpw,"%c", cha);
  }
  fclose(fp);
  fclose(fpw);
  */
  
  cout<<" L Lz dx dt cutoff ep Mem r0 tau F0 gamma2 beta_sig theta_sig eta K K' Da Di ki1 ki2 at"<<endl;
  //  cout<<"#"<<filename<<" "<<L<<" "<<Lz<<" "<<dx<<" "<<dt<<" "<<cutoff<<" "<<ep<<" "<<Mem<<" "<<r0<<" "<<tau<<" "<<F0<<" "<<gamma2<<" "<<beta_sig<<" "<<theta_sig<<" "<<eta<<" "<<K<<" "<<Kd<<" "<<Da<<" "<<Di<<" "<<ki1<<" "<<ki2<<" "<<at<<" "<<endl;
  cout<<"#"<<" "<<L<<" "<<Lz<<" "<<dx<<" "<<dt<<" "<<cutoff<<" "<<ep<<" "<<Mem<<" "<<r0<<" "<<tau<<" "<<F0<<" "<<gamma2<<" "<<beta_sig<<" "<<theta_sig<<" "<<eta<<" "<<K<<" "<<Kd<<" "<<Da<<" "<<Di<<" "<<ki1<<" "<<ki2<<" "<<at<<" "<<endl;

}


//function for the initial patch nucleation
void set_initA(){
  for(int i=0;i<L;++i){
    for(int j=0;j<L;++j){
      for(int k=0;k<Lz;++k){
	if( sqrt((i-L/2)*(i-L/2)+(j-L/2)*(j-L/2))<=init_rng && k>=Lz/2 && ps[i*L*Lz+j*Lz+k]>cutoff){
	  A[i*L*Lz+j*Lz+k]=init_amp*genrand_real3();
	}
      }
    }
  }
}

   
int main(){
  init_genrand(seed);
  paramet_logs();
  set_init();
  for(int T=0;T<maxT;++T){

      double t=1.0*T*maxTT*dt;

      //output data (only values for psi > 100*cutoff is recorded for compression of data)
      cout<<"#T="<<T<< " t="<<t<<" "<<vol<<" "<<area<<endl;
      for(int i=0;i<L;++i){
	for(int j=0;j<L;++j){
	  for(int k=0;k<Lz;++k){
	    if(ps[i*L*Lz+j*Lz+k]>100.0*cutoff){cout<<i*L*Lz+j*Lz+k<<" "<<ph[i*L*Lz+j*Lz+k]<<" "<<ps[i*L*Lz+j*Lz+k]<<" "<<A[i*L*Lz+j*Lz+k]<<" "<<I[i*L*Lz+j*Lz+k]<<endl;}
	  }
	}
      }
      

#pragma acc data create(vamp[0:L*L*Lz]),create(vx[0:L*L*Lz]),create(vy[0:L*L*Lz]),create(vz[0:L*L*Lz]),create(phn[0:L*L*Lz]),create(psn[0:L*L*Lz]),create(An[0:L*L*Lz]),create(In[0:L*L*Lz]),copy(ph[0:L*L*Lz]),copy(ps[0:L*L*Lz]),copy(A[0:L*L*Lz]),copy(I[0:L*L*Lz])
      for(int TT=0;TT<maxTT;++TT){

	
	vol=0;
	intA=0;
	area=0;

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(3)
	for(int i=1;i<L-1;++i){
	  for(int j=1;j<L-1;++j){
	    for(int k=1;k<Lz-1;++k){
	      vol+=ph[i*L*Lz+j*Lz+k]*dx*dx*dx;
	      area+=ps[i*L*Lz+j*Lz+k]*dx*dx*dx/ep;
	      intA+=ps[i*L*Lz+j*Lz+k]*A[i*L*Lz+j*Lz+k]*dx*dx*dx;
	    }
	  }
	}
	intA=intA/(1.0*area);
	at_n=at*4.0*pi*r0*r0/(area);

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(3)
	for(int i=1;i<L-1;++i){
	  for(int j=1;j<L-1;++j){
	    for(int k=1;k<Lz-1;++k){
	      phn[i*L*Lz+j*Lz+k]=ph[i*L*Lz+j*Lz+k]+dt*func_vamp(A,I,ph,i,j,k,vol);
	      if(d1th_abs(ph,i,j,k)>cutoff){
		vamp[i*L*Lz+j*Lz+k]=func_vamp(A,I,ph,i,j,k,vol)/d1th_abs(ph,i,j,k);
		vx[i*L*Lz+j*Lz+k]=-d1th_x(ph,i,j,k)/d1th_abs(ph,i,j,k);
		vy[i*L*Lz+j*Lz+k]=-d1th_y(ph,i,j,k)/d1th_abs(ph,i,j,k);
		vz[i*L*Lz+j*Lz+k]=-d1th_z(ph,i,j,k)/d1th_abs(ph,i,j,k);
	      }
	      else{
		vamp[i*L*Lz+j*Lz+k]=0;
	      }
	    }
	  }
	}
		
	BC(phn);BC(vamp);
	BC(vx);	BC(vy);	BC(vz);

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(3)
	for(int i=1;i<L-1;++i){
	  for(int j=1;j<L-1;++j){
	    for(int k=1;k<Lz-1;++k){
	      psn[i*L*Lz+j*Lz+k]=sigmoid(phn[i*L*Lz+j*Lz+k]*(1.0-phn[i*L*Lz+j*Lz+k]));
	    }
	  }
	}

#pragma acc parallel vector_length(256)
#pragma acc loop collapse(3)
	for(int i=1;i<L-1;++i){
	  for(int j=1;j<L-1;++j){
	    for(int k=1;k<Lz-1;++k){
	      if(ps[i*L*Lz+j*Lz+k]>cutoff){
		An[i*L*Lz+j*Lz+k]=A[i*L*Lz+j*Lz+k]+dt*(-(d1th_x_3v(A,vamp,vx,i,j,k)+d1th_y_3v(A,vamp,vy,i,j,k)+d1th_z_3v(A,vamp,vz,i,j,k)) +beta_sig*(1.0-ps[i*L*Lz+j*Lz+k])*(1.0-2.0*ph[i*L*Lz+j*Lz+k])*(Da*(d1th_x(ph,i,j,k)*d1th_x(A,i,j,k)+d1th_y(ph,i,j,k)*d1th_y(A,i,j,k)+d1th_z(ph,i,j,k)*d1th_z(A,i,j,k)))+Da*d2th(A,i,j,k)+ func_A(A,I,at_n,intA,i,j,k));
		In[i*L*Lz+j*Lz+k]=I[i*L*Lz+j*Lz+k]+dt*(-(d1th_x_3v(I,vamp,vx,i,j,k)+d1th_y_3v(I,vamp,vy,i,j,k)+d1th_z_3v(I,vamp,vz,i,j,k)) +beta_sig*(1.0-ps[i*L*Lz+j*Lz+k])*(1.0-2.0*ph[i*L*Lz+j*Lz+k])*(Di*(d1th_x(ph,i,j,k)*d1th_x(I,i,j,k)+d1th_y(ph,i,j,k)*d1th_y(I,i,j,k)+d1th_z(ph,i,j,k)*d1th_z(I,i,j,k)))+Di*d2th(I,i,j,k)+ func_I(A,I,i,j,k));
	      }
	      else{
		An[i*L*Lz+j*Lz+k]=A[i*L*Lz+j*Lz+k]-dt*gamma2*A[i*L*Lz+j*Lz+k];
		In[i*L*Lz+j*Lz+k]=I[i*L*Lz+j*Lz+k]-dt*gamma2*I[i*L*Lz+j*Lz+k];
	      }

	    }
	  }
	}

	update(ph,phn);
	update(ps,psn);
	update(A,An);
	update(I,In);
	

      }//close TT loop
      
      if(T==3){set_initA();} // initial nucleation of the patch


      
      if(genrand_real2()<theta && T>3){ //noise
	int i=genrand_int32()%(L-4)+2;
	int j=genrand_int32()%(L-4)+2;
	int k=genrand_int32()%(Lz-4)+2;
	
	for(int ii=0;ii<8;++ii){
	  for(int jj=0;jj<8;++jj){
	    for(int kk=0;kk<8;++kk){
	      double rnd=sigma;
	      if(i+ii>0 && i+ii<L-1&& j+jj>0 && j+jj<L-1 && k+kk>0 && k+kk<Lz-1 && ps[(i+ii)*L*Lz+(j+jj)*L+k+kk]>cutoff){
		A[(i+ii)*L*Lz+(j+jj)*L+k+kk]+=rnd*genrand_real2();
	      }
	    }
	  }
	}
      }
      

          
     
  }//close T loop



  free(ph);
  free(phn);
  free(ps);
  free(psn);
  free(A);
  free(An);
  free(I);
  free(In);
  free(vamp);
  free(vx);
  free(vy);
  free(vz);

}



