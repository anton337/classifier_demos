#ifndef ACTIVATION_PROBE_H
#define ACTIVATION_PROBE_H

template<typename T>
struct ActivationProbe
{

    long num_inputs;
    long  input_grid_nx;
    long  input_grid_ny;

    long num_outputs;
    long output_grid_nx;
    long output_grid_ny;

    T * input_dat;
    T * output_dat;

    ActivationProbe(Perceptron<T> * perceptron,long layer)
    {
         input_dat = NULL;
        output_dat = NULL;
        if(input_dat == NULL)
        {
            input_dat = new T[perceptron->n_nodes[layer+1]*perceptron->n_nodes[layer]];
        }
        num_inputs = perceptron->n_nodes[layer];
        if(output_dat == NULL)
        {
            output_dat = new T[perceptron->n_nodes[layer+1]];
        }
        num_outputs = perceptron->n_nodes[layer+1];
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

