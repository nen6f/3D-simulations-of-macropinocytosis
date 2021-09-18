
//need to install libpng to generate png files
//This code has been tested on macOS X.
//compile option
//g++ visualize3D.cc -framework GLUT -framework OpenGL -mmacosx-version-min=10.8 -lpng
//run: ./a.out


//#include <GL/glut.h> //use this for Linux
#include <GLUT/glut.h>
#include "glext.h"
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <math.h>
#include <string.h>
#include <iostream>
#include "algorithm"
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <png.h>
#include <sys/stat.h>
using namespace std;

// if mode_recoding 1, png files are generated
#define mode_recording 0

#define L 250 // use this for data generated by macropinocytosis_Fig2.cc
//#define L 430 // use this for data generated by macropinocytosis_Fig6.cc
#define Lz 430
#define sc 0.005
#define dx 0.1
#define ep 0.8
#define R0 10.0




char pngdir[250];
//defining input file name that is generated by macropinocytosis_Fig2.cc  
#define datfile "data_macro_Fig2.dat"

ifstream ifs(datfile);

void init(void)
{
  glClearColor(1.0,1.0, 1.0, 1.0);
  if(mode_recording==1){
    time_t timestamp=time(NULL);
    sprintf(pngdir,"./%s_%ld",datfile,timestamp);
    mkdir(pngdir, S_IRWXU|S_IRGRP|S_IXGRP);
    char filename[250];
    
    sprintf(filename,"%s/%s_%ld",pngdir,__FILE__,timestamp);
    FILE *fp,*fpw;
    fpw = fopen(filename, "w");
    fp = fopen(__FILE__, "r");
    char cha; 
    while( ( cha = fgetc(fp) ) != EOF ) {
      fprintf(fpw,"%c", cha);
    }
    fclose(fp);
    fclose(fpw);
  }
}

int SIZE=L*L*Lz;
double* ph=(double*)std::malloc(sizeof(double)*(SIZE));
double* phn=(double*)std::malloc(sizeof(double)*(SIZE));
double* ps=(double*)std::malloc(sizeof(double)*(SIZE));
double* psn=(double*)std::malloc(sizeof(double)*(SIZE));
double* A=(double*)std::malloc(sizeof(double)*(SIZE));
double* An=(double*)std::malloc(sizeof(double)*(SIZE));
double* I=(double*)std::malloc(sizeof(double)*(SIZE));
double* In=(double*)std::malloc(sizeof(double)*(SIZE));
double* vamp=(double*)std::malloc(sizeof(double)*(SIZE));
double* vx=(double*)std::malloc(sizeof(double)*(SIZE));
double* vy=(double*)std::malloc(sizeof(double)*(SIZE));
double* vz=(double*)std::malloc(sizeof(double)*(SIZE));


void capture(char* filename)
{
    png_bytep raw1D;
    png_bytepp raw2D;
    int i;
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    height-=height%2;

    
    //Allocating the structures
    FILE *fp = fopen(filename, "wb");
    png_structp pp = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop ip = png_create_info_struct(pp);
    // preparation
    png_init_io(pp, fp);
    png_set_IHDR(pp, ip, width, height,
        8, // 8bit
	PNG_COLOR_TYPE_RGBA, // RGBA
        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    // Allocating pixcel regions
    raw1D = (png_bytep)malloc(height * png_get_rowbytes(pp, ip));
    raw2D = (png_bytepp)malloc(height * sizeof(png_bytep));
    for (i = 0; i < height; i++)
        raw2D[i] = &raw1D[i*png_get_rowbytes(pp, ip)];
    // capture of the image
    glReadBuffer(GL_FRONT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height,GL_RGBA,GL_UNSIGNED_BYTE,(void*)raw1D);
    //Up and down reversal
    for (i = 0; i < height/ 2; i++){
        png_bytep swp = raw2D[i];
        raw2D[i] = raw2D[height - i - 1];
        raw2D[height - i - 1] = swp;
    }
    //writing
    png_write_info(pp, ip);
    png_write_image(pp, raw2D);
    png_write_end(pp, ip);
    //free
    png_destroy_write_struct(&pp, &ip);
    fclose(fp);
    free(raw1D);
    free(raw2D);

    printf("write out screen capture to '%s'\n", filename);
}


	
GLdouble vertex[][3] = {
  { 0.0, 0.0, 0.0 },
  { 1.0, 0.0, 0.0 },
  { 1.0, 1.0, 0.0 },
  { 0.0, 1.0, 0.0 },
  { 0.0, 0.0, 1.0 },
  { 1.0, 0.0, 1.0 },
  { 1.0, 1.0, 1.0 },
  { 0.0, 1.0, 1.0 }
};

int edge[][2] = {
  { 0, 1 },
  { 1, 2 },
  { 2, 3 },
  { 3, 0 },
  { 4, 5 },
  { 5, 6 },
  { 6, 7 },
  { 7, 4 },
  { 0, 1 },
  { 1, 5 },
  { 5, 4 },
  { 4, 0 },
  { 3, 2 },
  { 2, 6 },
  { 6, 7 },
  { 7, 3 },
  { 0, 4 },
  { 4, 7 },
  { 7, 3 },
  { 3, 0 }
};

void shift(double *x,double *x0, double dz,double *out){
  out[0]=(x0[0]+x[0])*dz;
  out[1]=(x0[1]+x[1])*dz;
  out[2]=(x0[2]+x[2])*dz;
}

void render_string(float x, float y, const char* string)
{
float z = -1.0f;
glRasterPos3f(x, y, z);
char* p = (char*) string;
while (*p != '\0') glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *p++);
}


int mode=0; // 0: normal view, 1:cross-section view
//the origin of initail view point
int ci=L/2; 
int cj=L/2;
int ck=Lz/2+100;

//parameters for time resolutions : Use the same value as in macropinocytosis_Fig2.cc  
double dt=0.0004;
int maxTT=10000;     

//initializing iteration step
int T=0;



void display(){

  string str;
  
  int a;
  double b,c,d,e;
  if(T==0){
    getline(ifs, str);
    getline(ifs, str);
    cout<<"!!!"<<endl;
  }

  //  initializing variables
    for(int i=0;i<L*L*Lz;++i){
      ph[i]=0;
      ps[i]=0;
      A[i]=0;
      I[i]=0;
    }

  int  flag=0;
  char strT[40];
  while(flag==0 &&     getline(ifs, str)) {
    if((int)str.find("#")==0){sscanf(str.data(), "%s", strT);flag=1;}
    else{
      sscanf(str.data(), "%d %lf %lf %lf %lf", &a, &b, &c, &d, &e);
      ph[a]=b;
      ps[a]=c;
      A[a]=d;
      I[a]=e;
    }
  }

  if(ifs.eof()==true){ 
    exit(0);
    ifs.clear();
    ifs.seekg(0, std::ios::beg);
    T=0;
    flag=0;
  }
  
  
  if(flag==1){
      //Drawing
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

     glEnable(GL_BLEND);
     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
     int M=L*L*Lz;
     for(int i=0;i<L;++i){
       for(int j=0;j<L;++j){
	 for(int k=0;k<Lz;++k){

	   int X=(i+ci-L/2);
	   int Y=(j+cj-L/2);
	   int Z=(k+ck-Lz/2);
	   if(Z>Lz-2){Z=Z-Lz+2;}
	   int ijk=((X)%L)*L*Lz+(Y)*Lz+Z;

	   if(ps[ijk]>0 && (mode==0 ||(i-ci+L/2)%(L)<L/2 )){
	     glColor4f(10.0*A[ijk]*ps[ijk]+0.3*ps[ijk],0.8*ps[ijk] , 10.0*I[ijk]*ps[ijk], 1.0*ps[ijk]*ps[ijk]);
	     glBegin(GL_QUAD_STRIP);

	     
	      double x1[3];
	      double x0[3]={i,j,k};
	      for (int ii = 0; ii < 10; ++ii) {
		shift(vertex[edge[ii][0]],x0,sc,x1);glVertex3dv(x1);
		shift(vertex[edge[ii][1]],x0,sc,x1);glVertex3dv(x1);
	      }
	      glEnd();
	     	
	      }
	     
	      
	      
	   }
	   
	   
	 }
       }


     glColor3f(0,0,0);
	
     char ch[50];     
     sprintf(ch,"t = %.0f",(1.0*(T-4))*maxTT*dt);
		render_string(-1.5, -0.5, ch);
	glFlush();

	if(mode_recording==1){
	  char ch2[150];
	  sprintf(ch2,"./%s/out_%04d.png",pngdir,T);
	  capture(ch2);
	}
	
	glutSwapBuffers();


	
	double vol=0;
	double area=0;

	for(int i=2;i<L-2;++i){
	  for(int j=2;j<L-2;++j){
	    for(int k=2;k<Lz-2;++k){
	      vol+=ph[i*L*Lz+j*Lz+k]*dx*dx*dx;
	      area+=ps[i*L*Lz+j*Lz+k]*dx*dx*dx/ep;
	    }
	  }
	}
      
	  
	
	T+=1;
	cout<<strT<<" "<<T<<" area="<<area/(4.0*3.141592*R0*R0)<<endl;

  }
  glutPostRedisplay();
      
      
}

void resize(int w, int h)
{
  
  glViewport(0, 0, w, h);
  glLoadIdentity();
  gluPerspective(7.0, (double)w / (double)h, 10.0, 100.0);
  gluLookAt(20.0, sc*L/2.0, 8.0, sc*L/2.0,sc*L/2.0, 250.0*sc, 0.0, 0, 1.0);  //center is sc*L/2,sc*L/2,sc*Lz/2
  glTranslated(sc*0.5*L, sc*0.5*L, sc*0.5*L);
  glRotated(0, 0,0,1.0);
  glTranslated(-sc*0.5*L, -sc*0.5*L, -sc*0.5*L);
  
  glTranslated(sc*0.5*L, sc*0.5*L, sc*0.5*L);
  glRotated(40,0,1.0, 0); 
  glTranslated(-sc*0.5*L, -sc*0.5*L, -sc*0.5*L);
  
  glTranslated(sc*0.5*L, sc*0.5*L, sc*0.5*L);
    //   glRotated(90, 0,0, 1.0); 
    //      glRotated(30, 1.0,0, 0);
  glTranslated(-sc*0.5*L, -sc*0.5*L, -sc*0.5*L);

  glTranslated(-sc*0.2*L, 0, 0);    
   
}


int main(int argc, char *argv[])
{
  int temp=0;
  glutInit(&temp,NULL);
  glutInitWindowSize(Lz*3.0,Lz*3.0);
  glutInitDisplayString("double rgb depth=0 samples=2");
  glutCreateWindow(argv[0]);
  glutReshapeFunc(resize);
  init();
  glutDisplayFunc(display);

  string str;

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  

  glutMainLoop();
  return 0;
}



