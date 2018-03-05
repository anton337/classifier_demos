#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <unistd.h>
#include <GL/glut.h>
#include "visualize_data_array.h"
#include "visualize_data_array_color.h"
#include "visualize_model_output.h"
#include "visualize_rbm_reconstruction.h"
#include "visualize_activation_probe.h"
#include "visualize_cnn_activation_probe.h"
#include "visualize_cnn_convolution_probe.h"
#include "visualize_crbm_visible_probe.h"
#include "visualize_crbm_hidden_probe.h"
#include "visualize_crbm_kernel_probe.h"

bool change_pos_index  = false;
bool change_neg_index  = false;
bool change_up_index   = false;
bool change_down_index = false;
bool CONTINUE = true;

#ifndef FLOAT_DISPLAY
std::vector < Display < double > * > displays;
#else
std::vector < Display < float  > * > displays;
#endif

template<typename T>
void addDisplay(Display<T> * display)
{
    displays.push_back(display);
}

void draw(void)
{

    for(long i=0;i<displays.size();i++)
    {
        displays[i] -> signal ( change_pos_index
                              , change_neg_index
                              , change_up_index
                              , change_down_index
                              );
    }

    for(long i=0;i<displays.size();i++)
    {
        displays[i] -> update ();
    }

    change_pos_index = false;
    change_neg_index = false;
    change_up_index = false;
    change_down_index = false;

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
  glLoadIdentity();
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
    case ' ':CONTINUE=false;break;
    case 'd':change_pos_index=true;break;
    case 'a':change_neg_index=true;break;
    case 'w':change_up_index=true;break;
    case 's':change_down_index=true;break;
    case 'm':scale_factor*=1.1;std::cout << scale_factor << std::endl;break;
    case 'n':scale_factor/=1.1;std::cout << scale_factor << std::endl;break;
    case 'k':scale_factor_2*=1.1;std::cout << scale_factor_2 << std::endl;break;
    case 'j':scale_factor_2/=1.1;std::cout << scale_factor_2 << std::endl;break;
    case 27:
      {
        exit(1);
        break;
      }
  };
}

bool key_up     = false;
bool key_down   = false;
bool key_left   = false;
bool key_right  = false;

bool key_UP     = false;
bool key_DOWN   = false;
bool key_LEFT   = false;
bool key_RIGHT  = false;

void specialInput(int key,int x,int y)
{
    switch(key)
    {
        case GLUT_KEY_UP    : 
            {
                if(glutGetModifiers() == GLUT_ACTIVE_SHIFT) 
                    key_UP    = true; 
                else
                    key_up    = true;
                break;
            }
        case GLUT_KEY_DOWN  : 
            {
                if(glutGetModifiers() == GLUT_ACTIVE_SHIFT) 
                    key_DOWN  = true;
                else
                    key_down  = true;
                break;
            }
        case GLUT_KEY_LEFT  : 
            {
                if(glutGetModifiers() == GLUT_ACTIVE_SHIFT) 
                    key_LEFT  = true;
                else
                    key_left  = true;
                break;
            }
        case GLUT_KEY_RIGHT : 
            {
                if(glutGetModifiers() == GLUT_ACTIVE_SHIFT) 
                    key_RIGHT = true;
                else
                    key_right = true;
                break;
            }
        default: break;
    }
}

int mouse_x = -1;
int mouse_y = -1;
bool left_selected = false;
bool right_selected = false;

void OnMouseClick(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) 
  { 
    std::cout << x << '\t' << y << std::endl;
    mouse_x = x;
    mouse_y = y;
    left_selected = true;
  } 
  if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) 
  { 
    mouse_x = x;
    mouse_y = y;
    right_selected = true;
  } 
}

void startGraphics(int argc,char**argv,std::string title,int winx=400,int winy=400)
{
  glutInit(&argc, argv);
  glutInitWindowSize(winx,winy);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow(title.c_str());
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(specialInput);
  glutMouseFunc(OnMouseClick);
  init();
  glutMainLoop();
}

#endif
