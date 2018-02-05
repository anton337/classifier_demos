#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <unistd.h>
#include <GL/glut.h>
#include "visualize_data_array.h"

void draw(void)
{
    viz_selection++;
    if(viz_selection*n_vars >= n_elems)
    {
        viz_selection = 0;
    }
    visualize_data_array ( viz_selection
                         , n_elems
                         , n_vars
                         , n_x
                         , n_y
                         , viz_dat
                         );
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  draw();
  glutSwapBuffers();
}

void idle(void)
{
  usleep(100000);
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
  gluLookAt(0.0, 0.0, 3,  /* eye is at (0,0,5) */
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
  };
}

void startGraphics(int argc,char**argv,std::string title)
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow(title.c_str());
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutKeyboardFunc(keyboard);
  init();
  glutMainLoop();
}

#endif
