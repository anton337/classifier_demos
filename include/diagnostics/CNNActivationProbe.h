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

    T * kernel_dat;
    T * output_conv_dat;

    CNNActivationProbe(ConvolutionalNN<T> * cnn,long layer)
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
              long M = n_features[layer];
              long N = n_features[layer+1];
              long wx = (kx[layer]/2);
              long wy = (ky[layer]/2);
              long dx = nx[layer] - wx*2;
              long dy = ny[layer] - wy*2;
              kernel_dat = NULL;
              output_conv_dat = NULL;
              if(kernel_dat == NULL)
              {
                  kernel_dat = new T[N*M*kx[layer]*ky[layer]];
              }
              num_kernel = N*M*kx[layer]*ky[layer];
              if(output_conv_dat == NULL)
              {
                  output_conv_dat = new T[N*(dx)*(dy)];
              }
              num_output_conv = N*(dx)*(dy);
              break;
            }
          case MAX_POOLING_LAYER :
          case MEAN_POOLING_LAYER :
            {
              kernel_dat = NULL;
              num_kernel = 0;
              long M = n_features[layer];
              long N = n_features[layer+1];
              long dx = nx[layer] / pooling_factorx[layer+1];
              long dy = ny[layer] / pooling_factory[layer+1];
              output_conv_dat = new T[N*dx*dy];
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

    void get_neuron_inputs(Perceptron<T> * perceptron,long layer)
    {
        for(long i=0,k=0;i<perceptron->n_nodes[layer+1];i++)
        {
            for(long j=0;j<perceptron->n_nodes[layer];j++,k++)
            {
                input_dat[k] = perceptron->weights_neuron[layer][i][j];
            }
        }
    }

    void get_neuron_outputs(Perceptron<T> * perceptron,long layer)
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
    }

};

#endif

