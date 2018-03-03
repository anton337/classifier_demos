#include <iostream>
#include "ConvolutionalRBM.h"
#include "text.h"
#include "readBMP.h"
#include "visualization.h"

void clear() 
{
  // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
  std::cout << "\x1B[2J\x1B[H";
}

ConvolutionalRBM<double> * rbm = NULL;

long nx = 20;
long ny = 20;
long kx = 5;
long ky = 5;
long dx = nx-2*(kx/2);
long dy = ny-2*(ky/2);
long M = 2;
long K = 16+1;
long n = 8;

void train()
{
    double eps = 1e-3;
    bool init = true;
    double init_error;
    for(long i=0;;i++)
    {
      rbm -> init(0);
      rbm -> cd(10,eps,0);
      if(init)
      {
        init = false;
        init_error = rbm->final_error;
      }
      if(i%100==0)
      {
        std::cout << n << '\t' << init_error << '\t' << rbm -> final_error << std::endl;
      }
    }

}

int main(int argc,char ** argv)
{
  std::cout << "Convolutional RBM Test." << std::endl;
  srand(time(0));
  if(argc>0)
  {
    Image * img = new Image();
    ImageLoad(argv[1],img);
    double * dat_full = img->get_doubles(nx,ny);
    for(long m=0,k=0;m<M;m++)
    {
        for(long x=0;x<nx;x++)
        {
          for(long y=0;y<ny;y++,k++)
          {
              std::cout << ((dat_full[0*M*nx*ny+m*ny*nx+x*ny+y]>0.5)?'*':' ');
          }
          std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    long full_n = img->get_size()/(nx*ny);
    double * dat = new double[M*n*nx*ny];
    for(long i=0,k=0;i<n;i++)
    {
      long ind_1 = rand()%(full_n-1);
      long ind_2 = ind_1+1;
        for(long m=0;m<M;m++)
        for(long x=0;x<nx;x++)
        {
            for(long y=0;y<ny;y++,k++)
            {
              if(m==0)
                dat[k] = dat_full[ind_1*nx*ny+x*ny+y];
              else
                dat[k] = dat_full[ind_2*nx*ny+x*ny+y];
            }
        }
    }
    rbm = new ConvolutionalRBM<double> ( M*nx*ny
                                       , K*dx*dy
                                       , nx
                                       , ny
                                       , dx
                                       , dy
                                       , kx
                                       , ky
                                       , M
                                       , K
                                       , n
                                       , dat
                                       );

    VisualizeCRBMVisibleProbe < double > * viz_crbm_visible = NULL;
    viz_crbm_visible = new VisualizeCRBMVisibleProbe < double > ( rbm
                                                                , new CRBMConvolutionProbe < double > ( rbm )
                                                                , -1 , -.5
                                                                , -1 , 1
                                                                );
    addDisplay ( viz_crbm_visible );

    VisualizeCRBMKernelProbe < double > * viz_crbm_kernel = NULL;
    viz_crbm_kernel = new VisualizeCRBMKernelProbe < double > ( rbm
                                                              , new CRBMConvolutionProbe < double > ( rbm )
                                                              , -.5 , .5
                                                              ,  -1 , 1
                                                              );
    addDisplay ( viz_crbm_kernel );

    VisualizeCRBMHiddenProbe < double > * viz_crbm_hidden = NULL;
    viz_crbm_hidden = new VisualizeCRBMHiddenProbe < double > ( rbm
                                                              , new CRBMConvolutionProbe < double > ( rbm )
                                                              ,  .5 , 1
                                                              ,  -1 , 1
                                                              );
    addDisplay ( viz_crbm_hidden );
  }
  else
  {
    std::cout << "Please specify input name [bmp format]." << std::endl;
    exit(1);
  }

  new boost::thread(train);

  startGraphics(argc,argv,"CRBM");
  std::cout << "Done!" << std::endl;
  return 0;
}

