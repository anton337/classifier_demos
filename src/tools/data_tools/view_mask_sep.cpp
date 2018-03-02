#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <GL/glut.h>
using namespace std;

float * dat = NULL;

long offset = 0;

long selection = 0;

long nx,ny,nz;

float max_val = -1e20;
float min_val =  1e20;

void read_sep(long p_nx,long p_ny,long p_nz,std::string filename,std::string output)
{
    nx = p_nx;
    ny = p_ny;
    nz = p_nz;
    ifstream file (filename.c_str() , ios::binary);
    file.seekg (0, ios::beg);
    char * dat_c = new char[4*ny*nz];
    dat = new float[ny*nz*nx];
    offset = ny*nz;
    long k=0;
    for(long x=0;x<nx;x++)
    {
        file . read(dat_c, 4*ny*nz);
        for(long i=0;i<ny*nz;i++,k++)
        {
            dat[k] = *reinterpret_cast<float*>(&dat_c[4*i]);
            max_val = (dat[k] > max_val)?dat[k]:max_val;
            min_val = (dat[k] < min_val)?dat[k]:min_val;
        }
        std::cout << "x=" << x << std::endl;
    }
    file . close ();
}

void draw(void)
{
        float min_z = -1, max_z = 1;
        float min_y = -1, max_y = 1;
        float dz = (max_z - min_z)/nz;
        float dy = (max_y - min_y)/ny;
        float val;
        glBegin(GL_QUADS);
        for(long y=0,k=offset*selection;y<ny;y++)
        {
            for(long z=0;z<nz;z++,k++)
            {
                val = (dat[k]-min_val)/(max_val-min_val);
                glColor3f(val,val,val);
                glVertex3f( min_y + dy*y
                          , min_z + dz*(nz-1-z)
                          , 0
                          );
                glVertex3f( min_y + dy*y
                          , min_z + dz*(nz-1-z+1)
                          , 0
                          );
                glVertex3f( min_y + dy*(y+1)
                          , min_z + dz*(nz-1-z+1)
                          , 0
                          );
                glVertex3f( min_y + dy*(y+1)
                          , min_z + dz*(nz-1-z)
                          , 0
                          );
            }
        }
        glEnd();
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  draw();
  glutSwapBuffers();
}

void idle(void)
{
  usleep(100);
  glutPostRedisplay();
}

void init(void)
{
  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 2.75,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */
}

void keyboard(unsigned char Key, int x, int y)
{
  switch(Key)
  {
    case 27:
      {
        exit(1);
        break;
      }
    case 'w':selection++;if(selection>=nx)selection=nx-1;std::cout << "selection=" << selection << std::endl;break;
    case 's':selection--;if(selection<0)selection=0;std::cout << "selection=" << selection << std::endl;break;
    default : break;
  };
}

int main(int argc,char ** argv)
{
    long nx = 232;
    long ny = 4087;
    long nz = 1601;
    read_sep(nx,ny,nz,"/home/antonk/data/geopress.sep","/home/antonk/data/geopress.mask");
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutCreateWindow("Mask");
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    init();
    glutMainLoop();
    return 0;
}

