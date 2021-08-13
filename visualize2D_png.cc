//need to install libpng to generate png files
//This code has been tested on macOS X.
//compile option
//g++ visualize3D.cc -framework GLUT -framework OpenGL -mmacosx-version-min=10.8 -lpng
//run: ./a.out


//#include <GL/glut.h> //use this for Linux
#include <GLUT/glut.h>
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

#define pi 3.14159265

#define Lx 200
#define Ly 600
double dt=0.0004;
int maxTT=40000;     

#define dx 0.1
#define ep 0.8
double R0=10.0;


//definition of variables 
double ph[Lx*Ly],phn[Lx*Ly]; //phi
double ps[Lx*Ly],psn[Lx*Ly]; //psi
double A[Lx*Ly],An[Lx*Ly];
double I[Lx*Ly],In[Lx*Ly];

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


void render_string(float x, float y, const char* string)
{
float z = -1.0f;
glRasterPos3f(x, y, z);
char* p = (char*) string;
while (*p != '\0') glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *p++);
}


int T=0;



char pngdir[250];
//input file name that is generated by macropinocytosis_FigS1.cc  
#define datfile "dat_Fig.S1_F2.0at2.8.dat"

ifstream ifs(datfile);

void init(void)
{
  glClearColor(1.0, 1.0, 1.0, 1.0);
  
  if(mode_recording==1){
    time_t timestamp=time(NULL);
    sprintf(pngdir,"./%s_%ld",datfile,timestamp);
    mkdir(pngdir, S_IRWXU|S_IRGRP|S_IXGRP);
    //    cout<<pngdir<<endl;

    char filename[250];

    //copying read_3D_ogl**.cc at the directory pngdir
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

void display(){

  string str;
  
  int a;
  double b,c,d,e,f;
  if(T==0){
    cout<<"!!!"<<endl;
    getline(ifs, str);
    getline(ifs, str);
    //    cout<<"!!!"<<endl;
    for(int i=0;i<Lx*Ly;++i){ph[i]=0;}
  }

    for(int i=0;i<Lx*Ly;++i){
      ps[i]=0;
      A[i]=0;
      I[i]=0;
    }

  int  flag=0;
  char strT[40];
  char str1[40],str2[40],str3[40],str4[40],str5[40],str6[40],str7[40],str8[40],str9[40];
  
  
  while(flag==0 &&     getline(ifs, str)) {
    if((int)str.find("#")==0){sscanf(str.data(), "%s %s %s %s %s %s %s %s %s %s", strT,str1,str2,str3,str4,str5,str6,str7,str8,str9);flag=1;}
    else{
      sscanf(str.data(), "%d %lf %lf %lf %lf", &a, &b, &c, &d, &e);
      ph[a]=b;
      ps[a]=c;
      A[a]=d;
      I[a]=e;
    }
  }

  if(ifs.eof()==true){exit(0);}
  
      //Drawing
	for(int i=0;i<Lx;++i){
	  for(int j=0;j<Ly;++j){
	    glColor3f(10.0*ps[i*Ly+j]*A[i*Ly+j],ph[i*Ly+j],0);	  
	    glBegin(GL_QUADS);
	    glVertex2f((-1.0+2.0*i/(1.0*Lx)),(-1.0+2.0*j/(1.0*Ly)));
	    glVertex2f((-1.0+2.0*(i+1)/(1.0*Lx)),(-1.0+2.0*j/(1.0*Ly)));
	    glVertex2f((-1.0+2.0*(i+1)/(1.0*Lx)),(-1.0+2.0*(j+1)/(1.0*Ly)));
	    glVertex2f((-1.0+2.0*i/(1.0*Lx)),(-1.0+2.0*(j+1)/(1.0*Ly)));
	    glEnd();
	  }
	}

	glColor3f(1.0,1.0,1.0);
	char ch[50];
	sprintf(ch,"t=%.1f",1.0*(T-5)*maxTT*dt);
	render_string((0.1), (0.9), ch);
	glFlush();

	if(mode_recording==1){
	  char ch2[150];
	  sprintf(ch2,"./%s/out_%04d.png",pngdir,T);
	  capture(ch2);
	}
	
	glutSwapBuffers();

	
	
	
	T+=1;
	cout<<strT<<" "<<str1<<" "<<str2<<" "<<str3<<" "<<str4<<" "<<str5<<" "<<str6<<" "<<str7<<" "<<str8<<" "<<str9<<" "<<endl;
	
	glutPostRedisplay();
    
     
      
	
}



int main(int argc, char *argv[])
{
  int temp=0;
  glutInit(&temp,NULL);
  glutInitWindowSize(Lx,Ly);
  glutInitDisplayMode(GLUT_RGBA);
  glutCreateWindow(argv[0]);
  init();
  glutDisplayFunc(display);
  glutMainLoop();
  return 0;
}