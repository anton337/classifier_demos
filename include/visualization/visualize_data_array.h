#ifndef VISUALIZE_DATA_ARRAY_H
#define VISUALIZE_DATA_ARRAY_H

#include "display.h"

template<typename T>
struct VisualizeDataArray : public Display < T >
{

    double * viz_dat;
    
    long viz_selection;
    
    long n_elems;
    
    long n_vars;
    
    long n_stride;
    
    long n_x;
    
    long n_y;

    VisualizeDataArray ( long num_elems
                       , long num_vars
                       , long stride
                       , long nx
                       , long ny
                       , T * dat
                       , T min_x
                       , T max_x
                       , T min_y
                       , T max_y
                       )
    : Display < T > ( min_x 
                    , max_x 
                    , min_y 
                    , max_y 
                    )
    {
        set_viz_data ( num_elems
                     , num_vars
                     , stride
                     , nx
                     , ny
                     , dat
                     );
        viz_selection = 0;
    }
    
    void set_viz_data ( long num_elems
                      , long num_vars
                      , long stride
                      , long nx
                      , long ny
                      , T * dat
                      )
    {
        n_elems = num_elems;
        n_vars  = num_vars;
        n_stride = stride;
        n_x = nx;
        n_y = ny;
        viz_dat = dat;
    }

    virtual void signal ( bool pos 
                        , bool neg 
                        , bool up 
                        , bool down 
                        )
    {
        if(pos)
        {
            viz_selection++;
            if(viz_selection*n_vars >= n_elems)
            {
                viz_selection = 0;
            }
        }
        if(neg)
        {
            viz_selection--;
            if(viz_selection < 0)
            {
                viz_selection = 0;
            }
        }
        if(down)
        {
            viz_selection+=n_stride;
            if(viz_selection*n_vars >= n_elems)
            {
                viz_selection = 0;
            }
        }
        if(up)
        {
            viz_selection-=n_stride;
            if(viz_selection < 0)
            {
                viz_selection = 0;
            }
        }
    }

    void update ()
    {
        visualize_data_array ( viz_selection
                             , n_elems
                             , n_vars
                             , n_x
                             , n_y
                             , viz_dat
                             , Display<T>::min_x
                             , Display<T>::max_x
                             , Display<T>::min_y
                             , Display<T>::max_y
                             );
    }
    
    void visualize_data_array ( long selection
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
        
        float val;
        T dx = (max_x-min_x)/nx;
        T dy = (max_y-min_y)/ny;
        glBegin(GL_QUADS);
        for(long y=0,k=selection*num_vars;y<ny;y++)
        {
            for(long x=0;x<nx;x++,k++)
            {
                val = dat[k];
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

};

#endif

