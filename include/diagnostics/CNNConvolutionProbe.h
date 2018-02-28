#ifndef CNN_CONVOLUTION_PROBE_H
#define CNN_CONVOLUTION_PROBE_H

template<typename T>
struct CNNConvolutionProbe
{

    LayerType layer_type;

    long layer;

    long num_inputs;
    long  input_grid_nx;
    long  input_grid_ny;

    long num_outputs;
    long output_grid_nx;
    long output_grid_ny;

    T * input_dat;
    T * output_dat;

    long num_convolution;
    long num_convolution_nx;
    long num_convolution_ny;

    long num_output_conv;
    long num_output_conv_nx;
    long num_output_conv_ny;

    long num_convolution_dx;
    long num_convolution_dy;

    long num_output_N;

    T * convolution_dat;
    T * output_conv_dat;

    CNNConvolutionProbe(ConvolutionalNeuralNetwork<T> * cnn,long p_layer)
    {

        layer = p_layer;

        layer_type = cnn -> n_layer_type[layer];

        switch ( layer_type )
        {
          case RELU_LAYER :
            {
              convolution_dat = NULL;
              num_convolution = 0;
              long M = cnn->n_features[layer];
              long N = cnn->n_features[layer+1];
              long dx = cnn->nx[layer];
              long dy = cnn->ny[layer];
              output_conv_dat = new T[N*dx*dy];
              num_convolution_dx = dx;
              num_convolution_dy = dy;
              num_output_N = N;
              break;
            }

          case FULLY_CONNECTED_LAYER :
            {
               input_dat = NULL;
              output_dat = NULL;
              if(input_dat == NULL)
              {
                  input_dat = new T[cnn->n_nodes[layer+1]*cnn->n_nodes[layer]];
              }
              num_inputs = cnn->n_nodes[layer];
              if(output_dat == NULL)
              {
                  output_dat = new T[cnn->n_nodes[layer+1]];
              }
              num_outputs = cnn->n_nodes[layer+1];
              break;
            }

          case CONVOLUTIONAL_LAYER :
            {
              long M = cnn->n_features[layer];
              long N = cnn->n_features[layer+1];
              long wx = (cnn->kx[layer]/2);
              long wy = (cnn->ky[layer]/2);
              long dx = cnn->nx[layer] - wx*2;
              long dy = cnn->ny[layer] - wy*2;
              convolution_dat = NULL;
              output_conv_dat = NULL;
              if(convolution_dat == NULL)
              {
                  convolution_dat = new T[N*dx*dy];
              }
              num_convolution = N*dx*dy;
              if(output_conv_dat == NULL)
              {
                  output_conv_dat = new T[N*(dx)*(dy)];
              }
              num_output_conv = N*(dx)*(dy);
              num_convolution_dx = dx;
              num_convolution_dy = dy;
              num_output_N = N;
              input_grid_nx = dx;
              input_grid_ny = dy;
              output_grid_nx = 1;
              output_grid_ny = N;
              break;
            }
          case MAX_POOLING_LAYER :
          case MEAN_POOLING_LAYER :
            {
              convolution_dat = NULL;
              num_convolution = 0;
              long M = cnn->n_features[layer];
              long N = cnn->n_features[layer+1];
              long dx = cnn->nx[layer] / cnn->pooling_factorx[layer+1];
              long dy = cnn->ny[layer] / cnn->pooling_factory[layer+1];
              output_conv_dat = new T[N*dx*dy];
              num_convolution_dx = dx;
              num_convolution_dy = dy;
              num_output_N = N;
              break;
            }

          default :
            {
              break;
            }
        }
    }

    void set_input_grid(int nx,int ny)
    {
        input_grid_nx = nx;
        input_grid_ny = ny;
        if(check_input_grid() == false)
        {
            std::cout << "Input grid inconsistent: " << nx << "x" << ny << "!=" << num_inputs << std::endl;
            exit(1);
        }
    }

    void set_output_grid(int nx,int ny)
    {
        output_grid_nx = nx;
        output_grid_ny = ny;
        if(check_output_grid() == false)
        {
            std::cout << "Output grid inconsistent: " << nx << "x" << ny << "!=" << num_outputs << std::endl;
            exit(1);
        }
    }

    bool check_input_grid()
    {
        if(num_inputs%input_grid_nx==0&&num_inputs%input_grid_ny==0)
        {
            return true;
        }
        return false;
    }

    bool check_output_grid()
    {
        if(num_outputs%output_grid_nx==0&&num_outputs%output_grid_ny==0)
        {
            return true;
        }
        return false;
    }

    void get_neuron_inputs(ConvolutionalNeuralNetwork<T> * cnn,long layer)
    {
        switch ( layer_type )
        {
          case FULLY_CONNECTED_LAYER :
            {
              std::cout << "convolutional probe applied to fully connected layer" << std::endl;
              exit(1);
              break;
            }
          case CONVOLUTIONAL_LAYER :
            {
              long M = cnn->n_features[layer];
              long N = cnn->n_features[layer+1];
              long wx = (cnn->kx[layer]/2);
              long wy = (cnn->ky[layer]/2);
              long dx = cnn->nx[layer] - wx*2;
              long dy = cnn->ny[layer] - wy*2;
              long kx = cnn->kx[layer];
              long ky = cnn->ky[layer];
              long nx = cnn->nx[layer];
              long ny = cnn->ny[layer];
              T factor = 1.0 / (kx*ky);
              {
                  for(long n=0,i=0;n<N;n++)
                  {
                      for(long oy=0;oy<dy;oy++)
                      for(long ox=0;ox<dx;ox++,i++)
                      {
                          T sum = 0;//cnn->weights_bias[layer][i];
                          long ix = ox+wx;
                          long iy = oy+wy;
                          for(long m=0;m<M;m++)
                          {
                              for(long fy=-wy,ty=0;fy<=wy;fy++,ty++)
                              for(long fx=-wx,tx=0;fx<=wx;fx++,tx++)
                              {
                                  // W * y
                                  sum += cnn->weights_neuron[layer][ky*n+ty][kx*m+tx]
                                       * cnn->activation_values1[layer][(nx*ny)*m + nx*(iy+fy) + (ix+fx)]
                                       * factor;
                              }
                          }
                          convolution_dat[i] = sigmoid(sum,0);
                      }
                  }
              }
              break;
            }
          case RELU_LAYER :
          case MAX_POOLING_LAYER :
          case MEAN_POOLING_LAYER :
            {
              break;
            }
          default :
            {
              break;
            }
        }
    }

    void get_neuron_outputs(ConvolutionalNeuralNetwork<T> * cnn,long layer)
    {
        std::cout << "convolutional probe get neuron outputs not defined" << std::endl;
        exit(1);
    }

};

#endif

