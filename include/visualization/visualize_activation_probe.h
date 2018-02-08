#ifndef VISUALIZE_ACTIVATION_PROBE_H
#define VISUALIZE_ACTIVATION_PROBE_H

#include "Perceptron.h"
#include "ActivationProbe.h"
#include "display.h"

template<typename T>
struct VisualizeActivationProbe : public Display < T >
{

    Perceptron < double > * probe_perceptron;
    ActivationProbe < double > * activation_probe;
    
    long   probe_in_nx;
    long  probe_out_nx;
    long   probe_in_ny;
    long  probe_out_ny;

    VisualizeActivationProbe ( Perceptron<T> * perceptron
                             , ActivationProbe<T> * activation_probe
                             , long in_nx 
                             , long out_nx 
                             , long in_ny 
                             , long out_ny 
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
        set_probe_data ( perceptron
                       , activation_probe
                       , in_nx
                       , out_nx
                       , in_ny
                       , out_ny
                       );
    }
    
    void set_probe_data ( Perceptron<T> * perceptron
                        , ActivationProbe<T> * probe
                        , long in_nx 
                        , long out_nx 
                        , long in_ny 
                        , long out_ny 
                        )
    {
        probe_perceptron = perceptron;
        activation_probe = probe;
         probe_in_nx =  in_nx;
        probe_out_nx = out_nx;
         probe_in_ny =  in_ny;
        probe_out_ny = out_ny;
    }

    void update ()
    {
        visualize_activation_probe ( probe_perceptron
                                   , activation_probe
                                   , probe_in_nx
                                   , probe_out_nx
                                   , probe_in_ny
                                   , probe_out_ny
                                   , Display<T>::min_x
                                   , Display<T>::max_x
                                   , Display<T>::min_y
                                   , Display<T>::max_y
                                   );
    }
    
    void visualize_activation_probe ( Perceptron<T> * perceptron
                                    , ActivationProbe<T> * activation_probe
                                    , long in_nx 
                                    , long out_nx 
                                    , long in_ny 
                                    , long out_ny 
                                    , T min_x 
                                    , T max_x 
                                    , T min_y 
                                    , T max_y 
                                    )
    {
    
        if(perceptron == NULL)
        {
            std::cout << "Perceptron is NULL" << std::endl;
            return;
        }
    
        if(activation_probe == NULL)
        {
            std::cout << "Activation probe is NULL" << std::endl;
            return;
        }
    
        activation_probe -> get_neuron_inputs ( perceptron , 0 );

        activation_probe -> get_neuron_outputs ( perceptron , 0 );
    
        activation_probe -> set_input_grid ( in_nx , in_ny );
    
        activation_probe -> set_output_grid ( out_nx , out_ny );
        
        float val;
        float val2;
        T  in_dx = 0.9 / activation_probe ->  input_grid_nx;
        T  in_dy = 0.9 / activation_probe ->  input_grid_ny;
        T out_dx = 1.0 / activation_probe -> output_grid_nx;
        T out_dy = 1.0 / activation_probe -> output_grid_ny;

        glBegin(GL_QUADS);
        for(long oy=0,k=0,o=0;oy<activation_probe->output_grid_ny;oy++)
        {
            for(long ox=0;ox<activation_probe->output_grid_nx;ox++,o++)
            {
                for(long iy=0;iy<activation_probe->input_grid_ny;iy++)
                {
                    for(long ix=0;ix<activation_probe->input_grid_nx;ix++,k++)
                    {
                        val  = 0.5f + activation_probe ->  input_dat [ k ] ;
                        val2 = activation_probe -> output_dat [ o ] ;
                        glColor3f(val,val,val2);
                        glVertex3f( min_x + (max_x-min_x)*(out_dx * ox + out_dx * in_dx* ix     )
                                  , min_y + (max_y-min_y)*(out_dy * oy + out_dy * in_dy* iy     )
                                  , 0
                                  );
                        glVertex3f( min_x + (max_x-min_x)*(out_dx * ox + out_dx * in_dx*(ix+1)  )
                                  , min_y + (max_y-min_y)*(out_dy * oy + out_dy * in_dy* iy     )
                                  , 0
                                  );
                        glVertex3f( min_x + (max_x-min_x)*(out_dx * ox + out_dx * in_dx*(ix+1)  )
                                  , min_y + (max_y-min_y)*(out_dy * oy + out_dy * in_dy*(iy+1)  )
                                  , 0
                                  );
                        glVertex3f( min_x + (max_x-min_x)*(out_dx * ox + out_dx * in_dx* ix     )
                                  , min_y + (max_y-min_y)*(out_dy * oy + out_dy * in_dy*(iy+1)  )
                                  , 0
                                  );
                    }
                }
            }
        }
        glEnd();
    
    }

};

#endif

