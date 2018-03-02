#ifndef CRBM_CONVOLUTION_PROBE_H
#define CRBM_CONVOLUTION_PROBE_H

template<typename T>
struct CRBMConvolutionProbe
{

    long num_inputs;
    long  input_grid_nx;
    long  input_grid_ny;

    long num_outputs;
    long output_grid_nx;
    long output_grid_ny;

    T * input_dat;
    T * output_dat;

    long nx;
    long ny;

    long dx;
    long dy;

    long M;
    long N;

    CRBMConvolutionProbe(ConvolutionalRBM<T> * crbm)
    {
        nx = crbm->nx;
        ny = crbm->ny;
        dx = crbm->dx;
        dy = crbm->dy;
        N = crbm->K;
        M = 1; // MARK!!!
        input_dat = new T[M*nx*ny];
        output_dat = new T[N*dx*dy];
    }

    void set_input_grid(int p_nx,int p_ny)
    {
        std::cout << "set input grid not implemented." << std::endl;
        exit(1);
    }

    void set_output_grid(int nx,int ny)
    {
        std::cout << "set output grid not implemented." << std::endl;
        exit(1);
    }

    bool check_input_grid()
    {
        return false;
    }

    bool check_output_grid()
    {
        return false;
    }

    void get_neuron_inputs(ConvolutionalRBM<T> * crbm,long layer)
    {

    }

    void get_neuron_outputs(ConvolutionalRBM<T> * crbm,long layer)
    {
        std::cout << "convolutional rbm probe get neuron outputs not defined" << std::endl;
        exit(1);
    }

};

#endif

