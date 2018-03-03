#ifndef VISUALIZE_CRBM_KERNEL_PROBE_H
#define VISUALIZE_CRBM_KERNEL_PROBE_H

#include "ConvolutionalRBM.h"
#include "CRBMConvolutionProbe.h"
#include "display.h"

template<typename T>
struct VisualizeCRBMKernelProbe : public Display < T >
{

    ConvolutionalRBM < double > * crbm;
    CRBMConvolutionProbe < double > * crbm_convolution_probe;
    
    long   probe_in_nx;
    long  probe_out_nx;
    long   probe_in_ny;
    long  probe_out_ny;

    VisualizeCRBMKernelProbe     ( ConvolutionalRBM     <T> * p_crbm
                                 , CRBMConvolutionProbe <T> * p_crbm_convolution_probe
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
        set_probe_data ( p_crbm
                       , p_crbm_convolution_probe
                       );
    }

    VisualizeCRBMKernelProbe     ( ConvolutionalRBM<T> * p_crbm
                                 , CRBMConvolutionProbe<T> * p_crbm_convolution_probe
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
        set_probe_data ( p_crbm
                       , p_crbm_convolution_probe
                       , in_nx
                       , out_nx
                       , in_ny
                       , out_ny
                       );
    }
    
    void set_probe_data ( ConvolutionalRBM     <T> * p_crbm
                        , CRBMConvolutionProbe <T> * p_crbm_convolution_probe
                        )
    {
        crbm = p_crbm;
        crbm_convolution_probe = p_crbm_convolution_probe;
         probe_in_nx = 0;
        probe_out_nx = 0;
         probe_in_ny = 0;
        probe_out_ny = 0;
    }
    
    void set_probe_data ( ConvolutionalRBM     <T> * p_crbm
                        , CRBMConvolutionProbe <T> * p_crbm_convolution_probe
                        , long in_nx 
                        , long out_nx 
                        , long in_ny 
                        , long out_ny 
                        )
    {
        crbm = p_crbm;
        crbm_convolution_probe = p_crbm_convolution_probe;
         probe_in_nx =  in_nx;
        probe_out_nx = out_nx;
         probe_in_ny =  in_ny;
        probe_out_ny = out_ny;
    }

    void update ()
    {
        visualize_convolution_probe( crbm
                                   , crbm_convolution_probe
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
    
    void visualize_convolution_probe( ConvolutionalRBM     <T> * p_crbm
                                    , CRBMConvolutionProbe <T> * p_crbm_convolution_probe
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
    
        if(p_crbm == NULL)
        {
            std::cout << "CRBM is NULL" << std::endl;
            return;
        }
    
        if(p_crbm_convolution_probe == NULL)
        {
            std::cout << "CRBM convolution probe is NULL" << std::endl;
            return;
        }
  
        p_crbm_convolution_probe -> get_neuron_kernels ( p_crbm );
        
        float val;
        T  in_dx = 0.9 / p_crbm_convolution_probe ->  ker_input_grid_nx;
        T  in_dy = 0.9 / p_crbm_convolution_probe ->  ker_input_grid_ny;
        T out_dx = 1.0 / p_crbm_convolution_probe -> ker_output_grid_nx;
        T out_dy = 1.0 / p_crbm_convolution_probe -> ker_output_grid_ny;

        glBegin(GL_QUADS);
        for(long ox=0,k=0,o=0;ox<p_crbm_convolution_probe->ker_output_grid_nx;ox++)
        {
            for(long oy=0;oy<p_crbm_convolution_probe->ker_output_grid_ny;oy++,o++)
            {
                float min_val = 100000;
                float max_val =-100000;
                long init_k = k;
                for(long iy=0;iy<p_crbm_convolution_probe->ker_input_grid_ny;iy++)
                {
                    for(long ix=0;ix<p_crbm_convolution_probe->ker_input_grid_nx;ix++,k++)
                    {
                        val  = p_crbm_convolution_probe ->  kernel_dat [ k ] ;
                        min_val = (min_val<val)?min_val:val;
                        max_val = (max_val>val)?max_val:val;
                    }
                }
                k = init_k;

                for(long iy=0;iy<p_crbm_convolution_probe->ker_input_grid_ny;iy++)
                {
                    for(long ix=0;ix<p_crbm_convolution_probe->ker_input_grid_nx;ix++,k++)
                    {
                        val  = p_crbm_convolution_probe ->  kernel_dat [ k ] ;
                        val = 0.5f*(val - min_val)/(max_val-min_val);
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
    
    }

};

#endif

