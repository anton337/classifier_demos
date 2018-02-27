#ifndef VISUALIZE_CNN_ACTIVATION_PROBE_H
#define VISUALIZE_CNN_ACTIVATION_PROBE_H

#include "ConvolutionalNN.h"
#include "CNNActivationProbe.h"
#include "display.h"

template<typename T>
struct VisualizeCNNActivationProbe : public Display < T >
{

    ConvolutionalNeuralNetwork < double > * cnn;
    CNNActivationProbe < double > * cnn_activation_probe;
    
    long   probe_in_nx;
    long  probe_out_nx;
    long   probe_in_ny;
    long  probe_out_ny;

    VisualizeCNNActivationProbe ( ConvolutionalNeuralNetwork<T> * p_cnn
                                , CNNActivationProbe<T> * p_cnn_activation_probe
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
        set_probe_data ( p_cnn
                       , p_cnn_activation_probe
                       );
    }

    VisualizeCNNActivationProbe ( ConvolutionalNeuralNetwork<T> * p_cnn
                                , CNNActivationProbe<T> * p_cnn_activation_probe
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
        set_probe_data ( p_cnn
                       , p_cnn_activation_probe
                       , in_nx
                       , out_nx
                       , in_ny
                       , out_ny
                       );
    }
    
    void set_probe_data ( ConvolutionalNeuralNetwork<T> * p_cnn
                        , CNNActivationProbe<T> * p_cnn_activation_probe
                        )
    {
        cnn = p_cnn;
        cnn_activation_probe = p_cnn_activation_probe;
         probe_in_nx = 0;
        probe_out_nx = 0;
         probe_in_ny = 0;
        probe_out_ny = 0;
    }
    
    void set_probe_data ( ConvolutionalNeuralNetwork<T> * p_cnn
                        , CNNActivationProbe<T> * p_cnn_activation_probe
                        , long in_nx 
                        , long out_nx 
                        , long in_ny 
                        , long out_ny 
                        )
    {
        cnn = p_cnn;
        cnn_activation_probe = p_cnn_activation_probe;
         probe_in_nx =  in_nx;
        probe_out_nx = out_nx;
         probe_in_ny =  in_ny;
        probe_out_ny = out_ny;
    }

    void update ()
    {
        visualize_activation_probe ( cnn
                                   , cnn_activation_probe
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
    
    void visualize_activation_probe ( ConvolutionalNeuralNetwork<T> * p_cnn
                                    , CNNActivationProbe<T> * p_cnn_activation_probe
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
    
        if(p_cnn == NULL)
        {
            std::cout << "CNN is NULL" << std::endl;
            return;
        }
    
        if(p_cnn_activation_probe == NULL)
        {
            std::cout << "CNN activation probe is NULL" << std::endl;
            return;
        }
   
        switch ( p_cnn_activation_probe -> layer_type )
        {
          case FULLY_CONNECTED_LAYER :
            {
              //std::cout << "Fully Connected Layer" << std::endl;
              p_cnn_activation_probe -> get_neuron_inputs  ( p_cnn , p_cnn_activation_probe -> layer );
              //p_cnn_activation_probe -> get_neuron_outputs ( p_cnn , p_cnn_activation_probe -> layer );
              
              p_cnn_activation_probe -> set_input_grid ( in_nx , in_ny );
              p_cnn_activation_probe -> set_output_grid ( out_nx , out_ny );
              
              float val;
              float val2;
              T  in_dx = 0.9 / p_cnn_activation_probe ->  input_grid_nx;
              T  in_dy = 0.9 / p_cnn_activation_probe ->  input_grid_ny;
              T out_dx = 1.0 / p_cnn_activation_probe -> output_grid_nx;
              T out_dy = 1.0 / p_cnn_activation_probe -> output_grid_ny;

              //std::cout << p_cnn_activation_probe -> input_grid_nx << '\t';
              //std::cout << p_cnn_activation_probe -> input_grid_ny << '\t';
              //std::cout << p_cnn_activation_probe -> output_grid_nx << '\t';
              //std::cout << p_cnn_activation_probe -> output_grid_ny << '\t';
              //std::cout << std::endl;

              glBegin(GL_QUADS);
              for(long oy=0,k=0,o=0;oy<p_cnn_activation_probe->output_grid_ny;oy++)
              {
                  for(long ox=0;ox<p_cnn_activation_probe->output_grid_nx;ox++,o++)
                  {
                      for(long iy=0;iy<p_cnn_activation_probe->input_grid_ny;iy++)
                      {
                          for(long ix=0;ix<p_cnn_activation_probe->input_grid_nx;ix++,k++)
                          {
                              val  = p_cnn_activation_probe ->  input_dat [ k ] ;
                              //val2 = p_cnn_activation_probe -> output_dat [ o ] ;
                              val += 0.5;
                              glColor3f(val,val,val);
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
              break;
            }
          case CONVOLUTIONAL_LAYER :
            {
              //std::cout << "Convolutional Layer" << std::endl;
              p_cnn_activation_probe -> get_neuron_inputs  ( p_cnn , p_cnn_activation_probe -> layer );
              //p_cnn_activation_probe -> get_neuron_outputs ( p_cnn , p_cnn_activation_probe -> layer );
              
              float val;
              float val2;
              T  in_dx = 0.9 / p_cnn_activation_probe ->  input_grid_nx;
              T  in_dy = 0.9 / p_cnn_activation_probe ->  input_grid_ny;
              T out_dx = 1.0 / p_cnn_activation_probe -> output_grid_nx;
              T out_dy = 1.0 / p_cnn_activation_probe -> output_grid_ny;

              //std::cout << p_cnn_activation_probe -> input_grid_nx << '\t';
              //std::cout << p_cnn_activation_probe -> input_grid_ny << '\t';
              //std::cout << p_cnn_activation_probe -> output_grid_nx << '\t';
              //std::cout << p_cnn_activation_probe -> output_grid_ny << '\t';
              //std::cout << std::endl;

              glBegin(GL_QUADS);
              for(long oy=0,k=0,o=0;oy<p_cnn_activation_probe->output_grid_ny;oy++)
              {
                  for(long ox=0;ox<p_cnn_activation_probe->output_grid_nx;ox++,o++)
                  {
                      for(long iy=0;iy<p_cnn_activation_probe->input_grid_ny;iy++)
                      {
                          for(long ix=0;ix<p_cnn_activation_probe->input_grid_nx;ix++,k++)
                          {
                              val  = p_cnn_activation_probe ->  kernel_dat [ k ] ;
                              //val2 = p_cnn_activation_probe -> output_dat [ o ] ;
                              val += 0.5;
                              glColor3f(val,val,val);
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
              break;
            }
          default :
            {
              break;
            }
        }
    
    }

};

#endif

