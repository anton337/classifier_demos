#ifndef VISUALIZE_MODEL_OUTPUT_H
#define VISUALIZE_MODEL_OUTPUT_H

#include "Perceptron.h"

Perceptron < double > * mod_perceptron = NULL;

bool visualize_model_output_toggle = false;

double mod_min_x = 0;

double mod_max_x = 1;

long mod_n_x = 1;

double mod_min_y = 0;

double mod_max_y = 1;

long mod_n_y = 1;

template<typename T>
void set_mod_data ( Perceptron<T> * perceptron
                  , double min_x , double max_x , long nx
                  , double min_y , double max_y , long ny
                  )
{
    visualize_model_output_toggle = true;
    mod_perceptron = perceptron;
    mod_min_x = min_x;
    mod_max_x = max_x;
    mod_n_x = nx;
    mod_min_y = min_y;
    mod_max_y = max_y;
    mod_n_y = ny;
}

template<typename T>
void visualize_model_output ( Perceptron<T> * perceptron
                            , T min_x , T max_x , long nx
                            , T min_y , T max_y , long ny
                            )
{

    if(visualize_model_output_toggle == false)return;

    if(perceptron == NULL)
    {
        std::cout << "Perceptron is NULL" << std::endl;
        return;
    }
    
    float val;
    T dx = 2.0/nx;
    T dy = 2.0/ny;
    glBegin(GL_QUADS);
    float min_val = 10;
    float max_val = -10;
    for(long y=0;y<ny;y++)
    {
        for(long x=0;x<nx;x++)
        {
            T * in = new T[2];
            in[0] = min_x + (max_x - min_x)*((double)x/nx);
            in[1] = min_y + (max_y - min_y)*((double)y/ny);
            T * out = perceptron->model(2,1,in);
            val = out[0];
            min_val = (val<min_val)?val:min_val;
            max_val = (val>max_val)?val:max_val;
            delete [] in;
            delete [] out;
            glColor3f(val,val,val);
            glVertex3f( -1 + dx*x
                      , -1 + dy*y
                      , 0
                      );
            glVertex3f( -1 + dx*(x+1)
                      , -1 + dy*y
                      , 0
                      );
            glVertex3f( -1 + dx*(x+1)
                      , -1 + dy*(y+1)
                      , 0
                      );
            glVertex3f( -1 + dx*x
                      , -1 + dy*(y+1)
                      , 0
                      );
        }
    }
    glEnd();
    //std::cout << min_val << '\t' << max_val << std::endl;

}

#endif

