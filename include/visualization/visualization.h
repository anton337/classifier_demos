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
#include "visualize_crbm_convolution_probe.h"

bool change_pos_index  = false;
bool change_neg_index  = false;
bool change_up_index   = false;
bool change_down_index = false;

std::vector < Display < double > * > displays;

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

#if 0

#ifdef VISUALIZE_ACTIVATION_PROBE

#endif

#ifdef VISUALIZE_RBM_RECONSTRUCTION
    if(change_pos_index)
    {
        rbm_selection++;
        if(rbm_selection*rbm_vars >= rbm_elems)
        {
            rbm_selection = 0;
        }
    }
    if(change_neg_index)
    {
        rbm_selection--;
        if(rbm_selection < 0)
        {
            rbm_selection = 0;
        }
    }
    if(change_down_index)
    {
        rbm_selection+=rbm_stride;
        if(rbm_selection*rbm_vars >= rbm_elems)
        {
            rbm_selection = 0;
        }
    }
    if(change_up_index)
    {
        rbm_selection-=rbm_stride;
        if(rbm_selection < 0)
        {
            rbm_selection = 0;
        }
    }
    visualize_rbm_reconstruction < double > ( viz_rbm 
                                            , rbm_max_layer
                                            , rbm_selection 
                                            , rbm_elems 
                                            , rbm_vars 
                                            , rbm_nx 
                                            , rbm_ny 
                                            , rbm_dat 
                                            , -1 
                                            , 0 
                                            , -1 
                                            , 1 
                                            );
#endif

#ifdef VISUALIZE_SIGNAL
    visualize_signal ( sig_elems , sig_dat );
    visualize_reconstruction ( sig_perceptron , sig_elems , sig_dat , sig_prct , sig_num_in , sig_start_elem );
#endif

#ifdef VISUALIZE_DATA_ARRAY
    if(change_pos_index)
    {
        viz_selection++;
        if(viz_selection*n_vars >= n_elems)
        {
            viz_selection = 0;
        }
    }
    if(change_neg_index)
    {
        viz_selection--;
        if(viz_selection < 0)
        {
            viz_selection = 0;
        }
    }
    if(change_down_index)
    {
        viz_selection+=n_stride;
        if(viz_selection*n_vars >= n_elems)
        {
            viz_selection = 0;
        }
    }
    if(change_up_index)
    {
        viz_selection-=n_stride;
        if(viz_selection < 0)
        {
            viz_selection = 0;
        }
    }
    visualize_data_array < double > ( viz_selection
                                    , n_elems
                                    , n_vars
                                    , n_x
                                    , n_y
                                    , viz_dat
                                    , 0 
                                    , 1 
                                    , -1 
                                    , 1 
                                    );
#endif

#ifdef VISUALIZE_MODEL_OUTPUT
    visualize_model_output ( mod_perceptron
                           , mod_min_x , mod_max_x , mod_n_x
                           , mod_min_y , mod_max_y , mod_n_y
                           );
#endif

#endif

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
