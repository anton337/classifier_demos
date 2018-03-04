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

ConvolutionalRBM<double> * rbm2 = NULL;

long nx = 32;
long ny = 32;
long kx = 3;
long ky = 3;
long kx2 = 5;
long ky2 = 5;
long dx = nx-2*(kx/2);
long dy = ny-2*(ky/2);
long dx2 = dx-2*(kx2/2);
long dy2 = dy-2*(ky2/2);
long M = 1;
long K = 8+1;
long K2 = 16+1;
long n = 8;

void train()
{
  while(CONTINUE)
  {
  
    {
      double eps = 1e-4;
      bool init = true;
      double init_error;
      double prev_error;
      init = true;
      for(long i=0;i<1000&&CONTINUE;i++)
      {
        rbm2 -> init(0);
        rbm2 -> cd(1,eps,0);
        if(init)
        {
          init = false;
          init_error = rbm2->final_error;
        }
        if(i%100==0)
        {
          std::cout << n << '\t' << init_error << '\t' << rbm2 -> final_error << std::endl;
        }
      }
    }

    /*
    {
      double eps = 1e-2;
      bool init = true;
      double init_error;
      init = true;
      for(long i=0;i<1000;i++)
      {
        rbm -> init(0);
        rbm -> cd(1,eps,0);
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
    */

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
      long ind_1 = i;
        for(long m=0;m<M;m++)
        for(long x=0;x<nx;x++)
        {
            for(long y=0;y<ny;y++,k++)
            {
                dat[k] = dat_full[(ind_1+m)*nx*ny+x*ny+y];
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

    double eps = 1e-2;
    bool init = true;
    double init_error;
    init = true;
    for(long i=0;i<1000;i++)
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

    rbm2 = new ConvolutionalRBM<double> ( K*dx*dy
                                        , K2*dx2*dy2
                                        , dx
                                        , dy
                                        , dx2
                                        , dy2
                                        , kx2
                                        , ky2
                                        , K
                                        , K2
                                        , n
                                        , rbm->hid
                                        , false
                                        );
    {
        VisualizeCRBMVisibleProbe < double > * viz_crbm_visible = NULL;
        viz_crbm_visible = new VisualizeCRBMVisibleProbe < double > ( rbm
                                                                    , new CRBMConvolutionProbe < double > ( rbm )
                                                                    , -1 , -.75
                                                                    , -1 , 1
                                                                    );
        addDisplay ( viz_crbm_visible );

        VisualizeCRBMKernelProbe < double > * viz_crbm_kernel = NULL;
        viz_crbm_kernel = new VisualizeCRBMKernelProbe < double > ( rbm
                                                                  , new CRBMConvolutionProbe < double > ( rbm )
                                                                  , -.75 , -.25
                                                                  ,  -1 , 1
                                                                  );
        addDisplay ( viz_crbm_kernel );

        VisualizeCRBMHiddenProbe < double > * viz_crbm_hidden = NULL;
        viz_crbm_hidden = new VisualizeCRBMHiddenProbe < double > ( rbm
                                                                  , new CRBMConvolutionProbe < double > ( rbm )
                                                                  ,  -.25 , 0
                                                                  ,  -1 , 1
                                                                  );
        addDisplay ( viz_crbm_hidden );
    }

    {
        VisualizeCRBMVisibleProbe < double > * viz_crbm_visible = NULL;
        viz_crbm_visible = new VisualizeCRBMVisibleProbe < double > ( rbm2
                                                                    , new CRBMConvolutionProbe < double > ( rbm2 )
                                                                    , 0 , .25
                                                                    , -1 , 1
                                                                    );
        addDisplay ( viz_crbm_visible );

        VisualizeCRBMKernelProbe < double > * viz_crbm_kernel = NULL;
        viz_crbm_kernel = new VisualizeCRBMKernelProbe < double > ( rbm2
                                                                  , new CRBMConvolutionProbe < double > ( rbm2 )
                                                                  , .25 , .75
                                                                  ,  -1 , 1
                                                                  );
        addDisplay ( viz_crbm_kernel );

        VisualizeCRBMHiddenProbe < double > * viz_crbm_hidden = NULL;
        viz_crbm_hidden = new VisualizeCRBMHiddenProbe < double > ( rbm2
                                                                  , new CRBMConvolutionProbe < double > ( rbm2 )
                                                                  ,  .75 , 1
                                                                  ,  -1 , 1
                                                                  );
        addDisplay ( viz_crbm_hidden );
    }
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

