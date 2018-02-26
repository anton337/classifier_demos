#ifndef CNN_ACTIVATION_PROBE_H
#define CNN_ACTIVATION_PROBE_H

template<typename T>
struct CNNActivationProbe
{

    LayerType layer_type;

    long num_inputs;
    long  input_grid_nx;
    long  input_grid_ny;

    long num_outputs;
    long output_grid_nx;
    long output_grid_ny;

    T * input_dat;
    T * output_dat;

    long num_kernel;
    long num_kernel_nx;
    long num_kernel_ny;

    long num_output_conv;
    long num_output_conv_nx;
    long num_output_conv_ny;

    long num_kernel_kx;
    long num_kernel_ky;

    long num_output_N;

    T * kernel_dat;
    T * output_conv_dat;

    CNNActivationProbe(ConvolutionalNeuralNetwork<T> * cnn,long layer)
    {

        layer_type = cnn -> n_layer_type[layer];

        switch ( layer_type )
        {
          case RELU_LAYER :
            {
              kernel_dat = NULL;
              num_kernel = 0;
              long M = n_features[layer];
              long N = n_features[layer+1];
              long dx = nx[layer];
              long dy = ny[layer];
              output_conv_dat = new T[N*dx*dy];
              num_kernel_kx = kx[layer];
              num_kernel_ky = ky[layer];
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
              kernel_dat = NULL;
              output_conv_dat = NULL;
              if(kernel_dat == NULL)
              {
                  kernel_dat = new T[N*M*cnn->kx[layer]*cnn->ky[layer]];
              }
              num_kernel = N*M*cnn->kx[layer]*cnn->ky[layer];
              if(output_conv_dat == NULL)
              {
                  output_conv_dat = new T[N*(dx)*(dy)];
              }
              num_output_conv = N*(dx)*(dy);
              num_kernel_kx = cnn->kx[layer];
              num_kernel_ky = cnn->ky[layer];
              num_output_N = N;
              break;
            }
          case MAX_POOLING_LAYER :
          case MEAN_POOLING_LAYER :
            {
              kernel_dat = NULL;
              num_kernel = 0;
              long M = cnn->n_features[layer];
              long N = cnn->n_features[layer+1];
              long dx = cnn->nx[layer] / cnn->pooling_factorx[layer+1];
              long dy = cnn->ny[layer] / cnn->pooling_factory[layer+1];
              output_conv_dat = new T[N*dx*dy];
              num_kernel_kx = cnn->kx[layer];
              num_kernel_ky = cnn->ky[layer];
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

    void get_neuron_inputs(ConvolutionalNeuralNetwork<T> * perceptron,long layer)
    {
        switch ( layer_type )
        {
          case FULLY_CONNECTED_LAYER :
            {
              for(long i=0,k=0;i<perceptron->n_nodes[layer+1];i++)
              {
                  for(long j=0;j<perceptron->n_nodes[layer];j++,k++)
                  {
                      input_dat[k] = perceptron->weights_neuron[layer][i][j];
                  }
              }
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
              for(long n=0,k=0;n<N;n++)
              {
                  for(long m=0;m<M;m++)
                  {
                      for(long ty=0;ty<ky;ty++)
                      {
                          for(long tx=0;tx<kx;tx++,k++)
                          {
                              kernel_dat[k] = cnn->weights_neuron[layer][ky*n+ty][kx*m+tx];
                          }
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

    void get_neuron_outputs(ConvolutionalNeuralNetwork<T> * perceptron,long layer)
    {
        switch ( layer_type )
        {
          case FULLY_CONNECTED_LAYER :
            {
              for(long i=0;i<perceptron->n_nodes[layer+1];i++)
              {
                  T sum = 0;
                  for(long j=0;j<perceptron->n_nodes[layer];j++)
                  {
                      sum += perceptron->activation_values1[layer]   [j] 
                           * perceptron->   weights_neuron [layer][i][j];
                  }
                  output_dat[i] = sigmoid(sum,0);
              }
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
              T factor = 1.0 / (kx*ky);
              {
                  for(long n=0,i=0;n<N;n++)
                  {
                      for(long oy=0;oy<dy;oy++)
                      for(long ox=0;ox<dx;ox++,i++)
                      {
                          T sum = 0.0;//g->weights_bias[layer][i];
                          long ix = ox+wx;
                          long iy = oy+wy;
                          for(long m=0;m<M;m++)
                          {
                              for(long fy=-wy,ty=0;fy<=wy;fy++,ty++)
                              for(long fx=-wx,tx=0;fx<=wx;fx++,tx++)
                              {
                                  // W * y
                                  sum += g->weights_neuron[layer][ky*n+ty][kx*m+tx]
                                       * g->activation_values[layer][(nx*ny)*m + nx*(iy+fy) + (ix+fx)]
                                       * factor;
                              }
                          }
                          output_conv_dat[i] = sigmoid(sum,0);
                      }
                  }
              }
              break;
            }
          case MAX_POOLING_LAYER :
            {
              long M = g->n_features[layer];
              long nx = g->nx[layer];
              long ny = g->ny[layer];
              long factorx = g->pooling_factorx[layer];
              long factory = g->pooling_factory[layer];
              long dx = nx / factorx;
              long dy = ny / factory;
              T tmp_val,mean_val;
              for(long m=0,i=0;m<M;m++)
              {
                  for(long y=0,oy=0;y<ny;y+=factory,oy++)
                  for(long x=0,ox=0;x<nx;x+=factorx,ox++,i++)
                  {
                      max_val = -100000000;
                      for(long ty=0;ty<factory;ty++)
                      for(long tx=0;tx<factorx;tx++)
                      {
                          tmp_val = g->activation_values[layer][m*nx*nx+nx*y+x];
                          if(tmp_val>max_val)
                          {
                            max_val = tmp_val;
                          }
                      }
                      output_conv_dat[i] = max_val;
                  }
              }
              break;
            }
          case MEAN_POOLING_LAYER :
            {
              long M = g->n_features[layer];
              long nx = g->nx[layer];
              long ny = g->ny[layer];
              long factorx = g->pooling_factorx[layer];
              long factory = g->pooling_factory[layer];
              long dx = nx / factorx;
              long dy = ny / factory;
              T tmp_val,mean_val;
              T factor = 1.0 / (factorx * factory);
              for(long m=0,i=0;m<M;m++)
              {
                  for(long y=0,oy=0;y<ny;y+=factory,oy++)
                  for(long x=0,ox=0;x<nx;x+=factorx,ox++,i++)
                  {
                      mean_val = 0;
                      for(long ty=0;ty<factory;ty++)
                      for(long tx=0;tx<factorx;tx++)
                      {
                          mean_val += g->activation_values[layer][m*nx*nx+nx*y+x];
                      }
                      output_conv_dat[i] = mean_val * factor;
                  }
              }
              break;
            }
          case RELU_LAYER :
            {
              for(long i=0;i<g->n_nodes[layer+1];i++)
              {
                  output_conv_dat[i] = max(0.0,g->activation_values[layer][i]);
              }
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

