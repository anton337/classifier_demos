#ifndef CRBM_CONVOLUTION_PROBE_H
#define CRBM_CONVOLUTION_PROBE_H

template<typename T>
struct CRBMConvolutionProbe
{

    long num_inputs;
    long in_input_grid_nx;
    long in_input_grid_ny;
    long out_input_grid_nx;
    long out_input_grid_ny;
    long ker_input_grid_nx;
    long ker_input_grid_ny;

    long num_outputs;
    long in_output_grid_nx;
    long in_output_grid_ny;
    long out_output_grid_nx;
    long out_output_grid_ny;
    long ker_output_grid_nx;
    long ker_output_grid_ny;

    T * input_dat;
    T * output_dat;
    T * kernel_dat;

    long kx;
    long ky;

    long nx;
    long ny;

    long dx;
    long dy;

    long M;
    long N;

    CRBMConvolutionProbe(ConvolutionalRBM<T> * crbm)
    {
        kx = crbm->kx;
        ky = crbm->ky;
        nx = crbm->nx;
        ny = crbm->ny;
        dx = crbm->dx;
        dy = crbm->dy;
        N = crbm->K;
        M = crbm->M; // MARK!!!
        input_dat = new T[M*nx*ny];
        output_dat = new T[N*dx*dy];
        kernel_dat = new T[N*M*kx*ky];
        in_input_grid_nx = nx;
        in_input_grid_ny = ny;
        in_output_grid_nx = 1;
        in_output_grid_ny = M;
        out_input_grid_nx = dx;
        out_input_grid_ny = dy;
        out_output_grid_nx = 1;
        out_output_grid_ny = N;
        ker_input_grid_nx = kx;
        ker_input_grid_ny = ky;
        ker_output_grid_nx = M;
        ker_output_grid_ny = N;
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

    void get_neuron_inputs(ConvolutionalRBM<T> * crbm,long index)
    {
        for(long m=0,t=0,k=crbm->v*index;m<crbm->M;m++)
        {
          for(long x=0;x<crbm->nx;x++)
            for(long y=0;y<crbm->ny;y++,k++,t++)
            {
              //input_dat[t] = fabs(crbm->vis[k] - crbm->vis0[k]);
              input_dat[t] = fabs(crbm->vis0[k]);
            }
        }
    }

    void get_neuron_outputs(ConvolutionalRBM<T> * crbm,long index)
    {
        T min_val = 1000000;
        T max_val =-1000000;
        for(long n=0,t=0,k=crbm->h*index;n<crbm->K;n++)
        {
          for(long x=0;x<crbm->dx;x++)
            for(long y=0;y<crbm->dy;y++,k++,t++)
            {
              output_dat[t] = crbm->hid[k];
              min_val = (output_dat[t]<min_val)?output_dat[t]:min_val;
              max_val = (output_dat[t]>max_val)?output_dat[t]:max_val;
            }
        }
        //std::cout << min_val << '\t' << max_val << std::endl;
    }

    void get_neuron_kernels(ConvolutionalRBM<T> * crbm)
    {
        long kx = crbm->kx;
        long ky = crbm->ky;
        long M = crbm->M;
        long K = crbm->K;
        for(long m=0,k=0;m<M;m++)
        for(long n=0;n<K;n++)
          {
            for(long x=0;x<kx;x++)
              for(long y=0;y<ky;y++,k++)
              {
                kernel_dat[k] = crbm->W[k];
              }
          }
    }

};

#endif

