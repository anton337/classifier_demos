#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <unistd.h>
#include <GL/glut.h>

void drawBox(void)
{
  //std::cout << "drawBox" << std::endl;
  //float max_err = 0;
  //for(long k=0;k<errs.size();k++)
  //{
  //  if(max_err<errs[k])max_err=errs[k];
  //}
  //glColor3f(1,1,1);
  //glBegin(GL_LINES);
  //for(long k=0;k+1<errs.size();k++)
  //{
  //  glVertex3f( -1 + 2*k / ((float)errs.size()-1)
  //            , errs[k] / max_err
  //            , 0
  //            );
  //  glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
  //            , errs[k+1] / max_err
  //            , 0
  //            );
  //  glVertex3f( -1 + 2*k / ((float)errs.size()-1)
  //            , 0
  //            , 0
  //            );
  //  glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
  //            , 0
  //            , 0
  //            );
  //  glVertex3f( -1 + 2*k / ((float)errs.size()-1)
  //            , 0
  //            , 0
  //            );
  //  glVertex3f( -1 + 2*k / ((float)errs.size()-1)
  //            , errs[k] / max_err
  //            , 0
  //            );
  //  glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
  //            , 0
  //            , 0
  //            );
  //  glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
  //            , errs[k+1] / max_err
  //            , 0
  //            );
  //}
  //glEnd();
  //if(rbm)
  //{
  //  float max_W = -1000;
  //  float min_W =  1000;
  //  for(long i=0,k=0;i<rbm->v;i++)
  //    for(long j=0;j<rbm->h;j++,k++)
  //    {
  //      if(rbm->W[k]>max_W)max_W=rbm->W[k];
  //      if(rbm->W[k]<min_W)min_W=rbm->W[k];
  //    }
  //  float fact_W = 1.0 / (max_W - min_W);
  //  float col;
  //  glBegin(GL_QUADS);
  //  float d=3e-3;
  //  for(long x=0;x<WIN;x++)
  //  {
  //    for(long y=0;y<WIN;y++)
  //    {
  //      for(long i=0;i<rbm->v/WIN;i++)
  //      {
  //        for(long j=0;j<rbm->h/WIN;j++)
  //        {
  //          col = 0.5f + 0.5f*(rbm->W[(i+x)*rbm->h+j+y]-min_W)*fact_W;
  //          glColor3f(col,col,col);
  //          glVertex3f(  -1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,  -1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
  //          glVertex3f(d+-1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,  -1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
  //          glVertex3f(d+-1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,d+-1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
  //          glVertex3f(  -1+(1.15*WIN*i+x)/(1.15*(float)rbm->v) ,d+-1+(1.15*WIN*j+y)/(1.15*(float)rbm->h),0);
  //        }
  //      }
  //    }
  //  }
  //  glEnd();
  //}
  //{
  //  float d = 5e-1;
  //  float col;
  //  glBegin(GL_QUADS);
  //  for(long y=0,k=0;y<WIN;y++)
  //  {
  //    for(long x=0;x<WIN;x++,k++)
  //    {
  //      glColor3f(vis_preview[k]
  //               ,vis_previewG[k]
  //               ,vis_previewB[k]
  //               );
  //      glVertex3f(      (x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(d/WIN+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(d/WIN+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(      (x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
  //    }
  //  }
  //  for(long y=0,k=0;y<WIN;y++)
  //  {
  //    for(long x=0;x<WIN;x++,k++)
  //    {
  //      glColor3f(vis0_preview[k]
  //               ,vis0_previewG[k]
  //               ,vis0_previewB[k]
  //               );
  //      glVertex3f(      0.5f+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,      -1+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(      0.5f+(x)/(2.0*WIN) ,d/WIN+-1+(2*WIN-1-y)/(2.0*WIN),0);
  //    }
  //  }
  //  for(long y=0,k=0;y<WIN;y++)
  //  {
  //    for(long x=0;x<WIN;x++,k++)
  //    {
  //      glColor3f(vis1_preview[k]
  //               ,vis1_previewG[k]
  //               ,vis1_previewB[k]
  //               );
  //      glVertex3f(      0.5f+(x)/(2.0*WIN) ,      -1-0.5f+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,      -1-0.5f+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(d/WIN+0.5f+(x)/(2.0*WIN) ,d/WIN+-1-0.5f+(2*WIN-1-y)/(2.0*WIN),0);
  //      glVertex3f(      0.5f+(x)/(2.0*WIN) ,d/WIN+-1-0.5f+(2*WIN-1-y)/(2.0*WIN),0);
  //    }
  //  }
  //  glEnd();
  //}
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawBox();
  glutSwapBuffers();
}

void idle(void)
{
  usleep(10000);
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
