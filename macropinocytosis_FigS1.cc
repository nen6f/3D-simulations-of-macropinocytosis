//using c++ compiler (e.g., g++ or icc) on linux workstation without GPU 
//following compile option is used (mt19937ar.cc and mt19937ar.h are for Mersenne twister)
//icc macropinocytosis_FigS1.cc mt19937ar.cc -lm  -O3
//run: "./a.out 2.0 2.8 0.005 0.25 $2 > datfile.dat"
//1st - 5th arguments are F, at, K, K', seed


#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <iostream>
#include "mt19937ar.h"
#include "algorithm"
#include <algorithm>
#include <vector>
#define _POSIX_SOURCE
#include <unistd.h>
using namespace std;


#define pi 3.1415926536
//system size
#define Lx 200
#define Ly 600


//spatial and temporal resolutions
#define dx 0.1
#define dt 0.0004
#define cutoff 0.001

//parameters for membrane 
#define ep 0.8
#define eta 0.5
#define Mem 5.0
#define R0 10.0 //
#define beta_sig 100.0 //
#define theta_sig 0.105 //
#define tau 10.0

//parameters for reaction-diffusion
#define Da 0.1
#define Di 0.01
#define ki1 0
#define ki2 0
#define alpha 1.0 // Use a negative alpha value for alpha=infty (Eq.(8))
#define gamma2 10.0
#define xmax 50.0 //upper limit of the concentration

//parameters for the initial condition
#define init_amp 5.0
#define init_rng 15

//the following paremeters are inputted as arguments of the main function
double F0,at,K,Kd;
int seed;

int maxT=15;
#define maxTT 40000

//defining grobal variables
double Vol,area,intA,at_n;

//defining local variables
double ph[(Lx)*(Ly)],phn[(Lx)*(Ly)]; //phi
double ps[(Lx)*(Ly)],psn[(Lx)*(Ly)]; //psi
double A[(Lx)*(Ly)],An[(Lx)*(Ly)];
double I[(Lx)*(Ly)],In[(Lx)*(Ly)];
double vamp[Lx*Ly];       
double vx[Lx*Ly],vy[Lx*Ly]; // direction of velocity:  the velocity field is expressed as (vamp*vx, vamp*vy)
#define  Npath 120
int path[Npath];

#define Nlog 20
int path_log[Nlog][Npath];
int Lat_log[Nlog];
int t_log[Nlog];
double engV_log[Nlog];

double engV,engV_b;
int eng;


void paramet_logs(){
  time_t t = time(NULL);
  char filename[50];
  //sprintf(filename,"%s/log_code/%ld_%s",getenv("HOME"),t,__FILE__);
sprintf(filename,"%s/log_code/%s_%ld_%d",getenv("HOME"),__FILE__,t,getpid());
    FILE *fp,*fpw;
  fpw = fopen(filename, "w");
  fp = fopen(__FILE__, "r");
  char cha; 
  while( ( cha = fgetc(fp) ) != EOF ) {
    fprintf(fpw,"%c", cha);
  }
  fclose(fp);
  fclose(fpw);

  cout<<"#"<<filename<<" "<<Lx<<" "<<Ly<<" "<<dx<<" "<<dt<<" "<<cutoff<<" "<<ep<<" "<<Mem<<" "<<R0<<" "<<F0<<" "<<eta<<" "<<at<<" "<<K<<" "<<Kd<<" "<<seed<<endl;
}



//function for caluclating psi from phi
double sigmoid(double x){
  return 1.0/(1.0+exp(-beta_sig*(x-theta_sig)));
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
double d1th_x(double *x, int i,int j){
  return (x[(i+1)*Ly+j]-x[(i-1)*Ly+j])/(2.0*dx);
}
double d1th_y(double *x, int i,int j){
  return (x[i*Ly+j+1]-x[i*Ly+j-1])/(2.0*dx);
}
double d1th_x_2v(double *x,double *y, int i,int j){
  return (x[(i+1)*Ly+j]*y[(i+1)*Ly+j]-x[(i-1)*Ly+j]*y[(i-1)*Ly+j])/(2.0*dx);
}
double d1th_y_2v(double *x,double *y, int i,int j){
  return (x[i*Ly+j+1]*y[i*Ly+j+1]-x[i*Ly+j-1]*y[i*Ly+j-1])/(2.0*dx);
}
double d1th_x_3v(double *x,double *y,double *z, int i,int j){
  return (x[(i+1)*Ly+j]*y[(i+1)*Ly+j]*z[(i+1)*Ly+j]-x[(i-1)*Ly+j]*y[(i-1)*Ly+j]*z[(i-1)*Ly+j])/(2.0*dx);
}
double d1th_y_3v(double *x,double *y,double *z, int i,int j){
  return (x[i*Ly+j+1]*y[i*Ly+j+1]*z[i*Ly+j+1]-x[i*Ly+j-1]*y[i*Ly+j-1]*z[i*Ly+j-1])/(2.0*dx);
}
double d1th_abs(double *x, int i,int j){
  double DX=d1th_x(x,i,j);
  double DY=d1th_y(x,i,j);
  return sqrt(DX*DX+DY*DY);
}
double d2th(double *x, int i, int j){
  return (x[(i+1)*Ly+j]+x[(i-1)*Ly+j]+x[i*Ly+j+1]+x[i*Ly+j-1]-4.0*x[i*Ly+j])/(dx*dx);
}



//function for calulating Eq.1
double func_vamp(double *AA,double *II,double *p, int i, int j, double vol,double f, double FB, double FI){
  return (eta*(d2th(p,i,j)+d1th_x(p,i,j)/(1.0*i*dx)-16.0*p[i*Ly+j]*(1.0-p[i*Ly+j])*(1.0-2.0*p[i*Ly+j])/(ep*ep))+(-Mem*(vol-4.0*pi*R0*R0*R0/3.0)+f*Fpoly(AA[i*Ly+j],((ki1>0)? AA[i*Ly+j]*AA[i*Ly+j]*ki1/ki2:AA[i*Ly+j]*AA[i*Ly+j]) ,FB,FI))*d1th_abs(p,i,j))/tau;
}

//functions for calulating reaction-diffusion
double func_A(double *AA,double *II,double AT, double inta,int i, int j){
  return AA[i*Ly+j]*AA[i*Ly+j]*(AT-inta)/((1.0+II[i*Ly+j])*(1.0+AA[i*Ly+j]*AA[i*Ly+j]/(alpha*alpha)))-AA[i*Ly+j];
}
double func_I(double *AA,double *II, int i, int j){
  return ki1*AA[i*Ly+j]*AA[i*Ly+j]-ki2*II[i*Ly+j];
}



//functions for boundary condition (periodic B.C.)
void BC_ph(double *x){
  for(int i=0;i<Lx;++i){
    x[i*Ly+0]=x[i*Ly+Ly-2];
    x[i*Ly+Ly-1]=x[i*Ly+1];
  }
  for(int i=0;i<Ly;++i){
     x[0*Ly+i]=x[1*Ly+i];
     x[(Lx-1)*Ly+i]=x[(Lx-2)*Ly+i];
  }
}

void BC(double *x){
  for(int i=0;i<Lx;++i){
    x[i*Ly+0]=x[i*Ly+Ly-2];
    x[i*Ly+Ly-1]=x[i*Ly+1];
  }
  for(int i=0;i<Ly;++i){
     x[0*Ly+i]=x[1*Ly+i];
     x[(Lx-1)*Ly+i]=x[(Lx-2)*Ly+i];
  }
}

void BC_vx(double *x){
  for(int i=0;i<Lx;++i){
    x[i*Ly+0]=x[i*Ly+Ly-2];
    x[i*Ly+Ly-1]=x[i*Ly+1];
  }
  for(int i=0;i<Ly;++i){
     x[0*Ly+i]=0;
     x[(Lx-1)*Ly+i]=x[(Lx-2)*Ly+i];
  }
}


//functions for update variables
void update_ph(double *x, double *xn){
  for(int i=1;i<Lx-1;++i){
    for(int j=1;j<Ly-1;++j){
      x[i*Ly+j]=(xn[i*Ly+j]>=0)? xn[i*Ly+j]:0;
    }
  }
 
  for(int i=0;i<Lx;++i){
    x[i*Ly+0]=x[i*Ly+Ly-2];
    x[i*Ly+Ly-1]=x[i*Ly+1];
  }
  for(int i=0;i<Ly;++i){
     x[0*Ly+i]=x[1*Ly+i];
     x[(Lx-1)*Ly+i]=x[(Lx-2)*Ly+i];
  }
  
}


void update(double *x, double *xn){
  for(int i=1;i<Lx-1;++i){
    for(int j=1;j<Ly-1;++j){
      x[i*Ly+j]=(xn[i*Ly+j]>=0)? xn[i*Ly+j]:0;
      if(xn[i*Ly+j]>xmax){x[i*Ly+j]=xmax;}
    }
  }

  for(int i=0;i<Lx;++i){
    x[i*Ly+0]=x[i*Ly+Ly-2];
    x[i*Ly+Ly-1]=x[i*Ly+1];
  }
  for(int i=0;i<Ly;++i){
     x[0*Ly+i]=x[1*Ly+i];
     x[(Lx-1)*Ly+i]=x[(Lx-2)*Ly+i];
  }

}





//initial condition
void set_init(){
  int cy0 = Ly/3;
  for(int i=0;i<Lx;++i){
    for(int j=0;j<Ly;++j){
      double r1=sqrt((i*dx)*(i*dx)+(j-cy0)*(j-cy0)*dx*dx);
      ph[i*Ly+j]=(1.0+tanh(2.0*(R0-r1)/ep))/2.0;
      A[i*Ly+j]=0;
      I[i*Ly+j]=0;     
    }
  }

  double tmp2=0;
  for(int i=cy0;i<=Ly-2;++i){//find max
    if(d1th_abs(ph,3,i)>tmp2){tmp2=d1th_abs(ph,3,i);path[0]=3*Ly+i;path[1]=4*Ly+i;}
  }
  engV=0;
  engV_b=0;
  for(int i=0;i<Nlog;++i){t_log[i]=0;}
  for(int i=0;i<Nlog;++i){Lat_log[i]=0;}
}



void set_initA(){
  int cy0 = Ly/3;
  for(int i=0;i<Lx;++i){
    for(int j=0;j<Ly;++j){
      if( i<init_rng   && ps[i*Ly+j]>cutoff && j>cy0){
	A[i*Ly+j]=init_amp*genrand_real3();
      }
    }
  }
}


//estimating volume of engulfment (i.e., macropinosome)
double cal_engV(double* p, int* Path,int *LAT){
  int ph0=0;
  int counter=0;
  int jt=-1;
  int jb=-1;
  double eV=0;
  int index=0;
  if(p[1*Ly+Ly-1]>0.5){ return -1;}
  else{
    for(int j=Ly-1;j>0;--j){ //counting number of transition \phi=0 -> 1 and  \phi= 1 -> 0 
      if(ph0==0 && p[1*Ly+j]>0.5){
	counter+=1;ph0=1;
	if(jt==-1){jt=j;}  // jt*dt is y value of the top point of the cell
	else{jb=j;} // jb*dt is y value of the bottom point of the cup
      }
      else if(ph0==1 && p[1*Ly+j]<0.5){counter+=1;ph0=0;}
    }
    
    if(counter==2){
      return 0;
    }
    else{ // engulfment detected !!!
      for(int j=jb;j<=jt;++j){  //jt*dt is y value of the top point of the cell
	if(p[1*Ly+j]<0.5){ //ph=0 at i=1 
	  int ph_i=0;
	  for(int i=2;i<Lx;++i){
	    if(ph_i==0 && p[i*Ly+j]>0.5){ph_i=1;Path[index]=i*Ly+j;index+=1;eV+=pi*(i*dx)*(i*dx)*dx;}
	  }
	}
	else{ //ph=1 at i=1 
	  int ph_i1=0;
	  int ph_i2=0;
	  int i1,i2;
	  for(int i=2;i<Lx;++i){
	    if(ph_i1==0 && p[i*Ly+j]<0.5){ph_i1=1;Path[index]=i*Ly+j;index+=1;i1=i;}
	    else if(ph_i1==1 && ph_i2==0 && p[i*Ly+j]>0.5){ph_i2=1;Path[index]=i*Ly+j;index+=1;i2=i;}
	  }
	  if(ph_i1!=0 && ph_i2!=0){eV+=pi*((i2*dx)*(i2*dx)-(i1*dx)*(i1*dx))*dx;}
	  else{index-=1;}
	}
      }
      Path[index]=-1;
    }
  }
  *LAT=index;
  return eV;
}

int Lat;

int main(int argc, char *argv[]){
  if(argc!=6){
    cout<<"argumentation error"<<endl;
    return 1;
  }
  F0=atof(argv[1]);
  at=atof(argv[2]);
  K=atof(argv[3]);
  Kd=atof(argv[4]);
  seed=atoi(argv[5]);

  paramet_logs();
  init_genrand(seed);
  int flg=0;
  set_init();
  

  int log=1;
  
  for(int T=0; T<maxT;++T){
  double t=1.0*T*dt;
  
  if(T==5){set_initA();} // initial nucleation of the patch
      
  for(int TT=0;TT<maxTT;++TT){
	
	Vol=0;
	for(int i=1;i<Lx-1;++i){
	  for(int j=1;j<Ly-1;++j){
	    Vol+=2.0*pi*(i*dx)*ph[i*Ly+j]*dx*dx;
	  }
	}

	area=0;
	intA=0;
	
	for(int i=1;i<Lx-1;++i){
	  for(int j=1;j<Ly-1;++j){
	    area+=2.0*pi*(i*dx)*ps[i*Ly+j]*dx*dx/ep;
	    intA+=2.0*pi*(i*dx)*ps[i*Ly+j]*A[i*Ly+j]*dx*dx;
	  }
	}
	intA=intA/area;
	at_n=at*4.0*pi*R0*R0/area;
	

	for(int i=1;i<Lx-1;++i){
	  for(int j=1;j<Ly-1;++j){
	    double temp=func_vamp(A,I,ph,i,j,Vol,F0,K,Kd);
	      phn[i*Ly+j]=ph[i*Ly+j]+dt*temp;
	      if(d1th_abs(ph,i,j)>cutoff){
		vamp[i*Ly+j]=temp/(d1th_abs(ph,i,j));
		vx[i*Ly+j]=-d1th_x(ph,i,j)/d1th_abs(ph,i,j);
		vy[i*Ly+j]=-d1th_y(ph,i,j)/d1th_abs(ph,i,j);
	      }
	      else{
		vamp[i*Ly+j]=0;
	      }
	  }
	}
	
	
	BC_ph(phn);
	BC(vamp);
	BC_vx(vx);
	BC(vy);
	
	for(int i=1;i<Lx-1;++i){
	  for(int j=1;j<Ly-1;++j){
	    psn[i*Ly+j]=sigmoid(phn[i*Ly+j]*(1.0-phn[i*Ly+j]));
	  }
	}


	
	for(int i=1;i<Lx-1;++i){
	  for(int j=1;j<Ly-1;++j){
	    if(ps[i*Ly+j]>cutoff){An[i*Ly+j]=A[i*Ly+j]+dt*(-(d1th_x_3v(A,vamp,vx,i,j)+d1th_y_3v(A,vamp,vy,i,j))-A[i*Ly+j]*vx[i*Ly+j]*vamp[i*Ly+j]/(1.0*i*dx) +beta_sig*(1.0-ps[i*Ly+j])*(1.0-2.0*ph[i*Ly+j])*(Da*(d1th_x(ph,i,j)*d1th_x(A,i,j)+d1th_y(ph,i,j)*d1th_y(A,i,j)))+Da*(d2th(A,i,j)+d1th_x(A,i,j)/(1.0*i*dx))+ func_A(A,I,at_n,intA,i,j));}
	    else{An[i*Ly+j]=A[i*Ly+j]-dt*gamma2*A[i*Ly+j];}
	    
	    if(ps[i*Ly+j]>cutoff){In[i*Ly+j]=I[i*Ly+j]+dt*(-(d1th_x_3v(I,vamp,vx,i,j)+d1th_y_3v(I,vamp,vy,i,j))-I[i*Ly+j]*vx[i*Ly+j]*vamp[i*Ly+j]/(1.0*i*dx) +beta_sig*(1.0-ps[i*Ly+j])*(1.0-2.0*ph[i*Ly+j])*(Di*(d1th_x(ph,i,j)*d1th_x(I,i,j)+d1th_y(ph,i,j)*d1th_y(I,i,j)))+Di*(d2th(I,i,j)+d1th_x(I,i,j)/(1.0*i*dx))+ func_I(A,I,i,j));}
	    else{In[i*Ly+j]=I[i*Ly+j]-dt*gamma2*I[i*Ly+j];}
	  }
	}

	        
	
      
	update_ph(ph,phn);
	update(ps,psn);
	update(A,An);
	update(I,In);
	
	if(TT%1==0){
	  int Lat_tmp;
	  double tmp=cal_engV(ph, path,&Lat_tmp);
	  if(tmp>engV){
	    engV=tmp;
	    for(int i=0;i<Npath;++i){path_log[0][i]=path[i];}
	    Lat_log[0]=Lat_tmp;
	    t_log[0]=T;
	    engV_log[0]=engV;
	  }
	}
	
  }
  //output data (only values for psi > 100*cutoff is recorded for compression of data)
  cout<<"#"<<" "<<T<<" "<<F0<<" "<<at<<" "<<K<<" "<<Kd<<" "<<Vol<<" "<<area/(4.0*pi*R0*R0)<<" "<<engV/(4.0*pi*R0*R0*R0/3.0)<<" "<<Lat_log[0]<<endl;
	for(int i=2;i<Lx-2;++i){
	  for(int j=2;j<Ly-2;++j){
	    if(ph[i*Ly+j]>10.0*cutoff ){cout<<i*Ly+j<<" "<<ph[i*Ly+j]<<" "<<ps[i*Ly+j]<<" "<<A[i*Ly+j]<<" "<<I[i*Ly+j]<<" "<<endl;}
	  }
	}
	cout<<endl;

      if(ph[3*Ly+Ly-2]>0.9){T=maxT;}	
  }

  /*
  cout<<"! "<<Lat_log[0]<<endl;
    for(int i=0;i<Lat_log[0]+1;++i){
      cout<<path_log[0][i]/Ly<<" "<<(path_log[0][i]%Ly)<<" "<<(path_log[0][i+1]/Ly)<<" "<<(path_log[0][i+1]%Ly)<<" "<<endl;
  }
  */

    cout<<endl;
    cout<<endl;
  for(int n=0;n<1;++n){
    if(t_log[n]>0){
      cout<<t_log[n]<<" "<<engV_log[n]/(4.0*pi*R0*R0*R0/3.0)<<" "<<Lat_log[n]<<endl;
    }
  }

  
}






