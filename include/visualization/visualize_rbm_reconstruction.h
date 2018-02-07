#ifndef VISUALIZE_RBM_RECONSTRUCTION_H
#define VISUALIZE_RBM_RECONSTRUCTION_H

#include <RBM.h>

std::vector < RBM<double> * > viz_rbm;

double * rbm_dat = NULL;

long rbm_selection = 0;

long rbm_elems = 0;

long rbm_vars = 0;

long rbm_nx = 0;

long rbm_ny = 0;

long rbm_max_layer = 1;

template<typename T>
void set_rbm_data ( std::vector < RBM<T> * > rbm
                  , long num_elems
                  , long num_vars
                  , long nx
                  , long ny
                  , T * dat
                  )
{
    viz_rbm = rbm;
    rbm_elems = num_elems;
    rbm_vars  = num_vars;
    rbm_nx = nx;
    rbm_ny = ny;
    rbm_dat = dat;
}

template<typename T>
void visualize_rbm_reconstruction ( std::vector < RBM<T> * > rbm
                                  , long max_layer
                                  , long selection
                                  , long num_elems
                                  , long num_vars
                                  , long nx
                                  , long ny
                                  , T * dat
                                  , T min_x = -1
                                  , T max_x = 1
                                  , T min_y = -1
                                  , T max_y = 1
                                  )
{

    if(dat == NULL)
    {
        std::cout << "Data array not initialized" << std::endl;
        exit(1);
    }

    if(num_elems % num_vars != 0)
    {
        std::cout << "Number of elements is not divisible by number of variables" << std::endl;
        exit(1);
    }

    if(nx*ny != num_vars)
    {
        std::cout << "nx * ny is not equal to number of variables" << std::endl;
        exit(1);
    }

    if(selection*num_vars >= num_elems)
    {
        std::cout << "selection index is outside of data range" << std::endl;
        exit(1);
    }

    double ** out = new double*[2*rbm.size()+2];
    for(int i=0;i<2*rbm.size()+2;i++)
    {
        out[i] = NULL;
    }
    int layer = 0;
    for(int r=0;r<rbm.size()&&r<max_layer;r++)
    {
        if(r == 0)
        {
            layer = 1;
            out[0] = new T[rbm[0]->v];
            out[1] = new T[rbm[0]->h];
            for(long y=0,k=0;y<ny;y++)
            {
                for(long x=0;x<nx;x++,k++)
                {
                    {
                        out[0][k] = dat[selection*num_vars+k];
                    }
                }
            }
            rbm[0]->vis2hid_simple(out[0],out[1]);
        }
        else
        {
            layer = r+1;
            out[r+1] = new T[rbm[r]->h];
            rbm[r]->vis2hid_simple(out[r],out[r+1]);
        }
    }
    int start_layer=rbm.size()-1;
    if(max_layer-1<start_layer)
    {
        start_layer = max_layer-1;
    }
    for(int r=start_layer;r>=0;r--)
    {
        if(r == 0)
        {
          out[layer+1] = new T[rbm[r]->v];
          rbm[r]->hid2vis_simple(out[layer],out[layer+1]);
          float val;
          T dx = (max_x-min_x)/nx;
          T dy = (max_y-min_y)/ny;
          glBegin(GL_QUADS);
          for(long y=0,k=0;y<ny;y++)
          {
              for(long x=0;x<nx;x++,k++)
              {
                  val = out[layer+1][k];
                  glColor3f(val,val,val);
                  glVertex3f( min_x + dx*x
                            , min_y + dy*y
                            , 0
                            );
                  glVertex3f( min_x + dx*(x+1)
                            , min_y + dy*y
                            , 0
                            );
                  glVertex3f( min_x + dx*(x+1)
                            , min_y + dy*(y+1)
                            , 0
                            );
                  glVertex3f( min_x + dx*x
                            , min_y + dy*(y+1)
                            , 0
                            );
              }
          }
          glEnd();
        }
        else
        {
          out[layer+1] = new T[rbm[r]->v];
          rbm[r]->hid2vis_simple(out[layer],out[layer+1]);
        }
        layer++;
    }

    for(int i=0;i<2*rbm.size()+2;i++)
    {
        if(out[i] != NULL)
        {
            delete [] out[i];
        }
    }
    delete [] out;

}

#endif

