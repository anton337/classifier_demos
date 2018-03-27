#include <iostream>
#include <boost/thread.hpp>
#include "readBMP.h"
#include "Perceptron.h"
#include "ActivationProbe.h"
#include "MergePerceptrons.h"
#include "AutoEncoderConstructor.h"
#include "visualization.h"

long nx = 1600;
long ny = 1200;

long kx = 9;
long ky = 9;
long dx = nx - (kx/2)*2;
long dy = ny - (ky/2)*2;
long N = 8*1;

long fx = 2;
long fy = 2;
double * DAT = new double[N*(dx/fx)*(dy/fy)];

long nx1 = dx/fx;
long ny1 = dy/fy;
long kx1 = 9;
long ky1 = 9;
long dx1 = nx1 - (kx1/2)*2;
long dy1 = ny1 - (ky1/2)*2;
long N1 = (8)*2;

long fx1 = 2;
long fy1 = 2;
double * DAT2 = new double[N*N1*(dx/fx1)*(dy/fy1)];

long nx2 = dx1/fx1;
long ny2 = dy1/fy1;
long kx2 = 9;
long ky2 = 9;
long dx2 = nx2 - (kx2/2)*2;
long dy2 = ny2 - (ky2/2)*2;
long N2 = (8)*4;

ConvolutionalRBM < double > * rbm = NULL;

ConvolutionalRBM < double > * rbm2 = NULL;

ConvolutionalRBM < double > * rbm3 = NULL;

void train()
{
  while(true)
  {
    //double eps = 1e-1;
    //for(long iter=0;iter<1;iter++)
    //{
    //    rbm -> init(0);
    //    rbm -> cd(1,eps,0);
    //    std::cout << iter << '\t' << rbm -> final_error << std::endl;
    //}
    double eps1 = 10;
    for(long iter=0;iter<5;iter++)
    {
        rbm2 -> init(0);
        rbm2 -> cd(1,eps1,0);
        std::cout << iter << '\t' << rbm2 -> final_error << std::endl;
    }

    for(long n=0,k=0;n<N;n++)
    {
        double max_value;
        for(long l=0;l<N1;l++)
        {
            for(long y=0;y<dy1/fy1;y++)
            {
                for(long x=0;x<dx1/fx1;x++,k++)
                {
                    max_value = -1000000;
                    for(long _y=0;_y<fy1;_y++)
                    {
                        for(long _x=0;_x<fx1;_x++)
                        {
                            max_value = (max_value>rbm2->hid[n*N1*dx1*dy1+l*dx1*dy1+(fy1*y+_y)*dx1+(fx1*x+_x)])
                                      ?  max_value:rbm2->hid[n*N1*dx1*dy1+l*dx1*dy1+(fy1*y+_y)*dx1+(fx1*x+_x)];
                        }
                    }
                    DAT2[k] = max_value;
                }
            }
        }
    }

    double eps2 = 1;
    for(long iter=0;iter<5;iter++)
    {
        rbm3 -> init(0);
        rbm3 -> cd(1,eps2,0);
        std::cout << iter << '\t' << rbm3 -> final_error << std::endl;
    }
  }
}

int main(int argc,char ** argv)
{
  std::cout << "Cat Classifier" << std::endl;
  // load input
  if(argc>0)
  {
    Image * dat = new Image();
    ImageLoad(argv[1],dat);
    double * D = dat->get_doubles(nx,ny);
    VisualizeDataArray < double > * viz_in_dat = NULL;
    viz_in_dat = new VisualizeDataArray < double > ( dat->get_size()
                                                   , nx*ny
                                                   , dat->get_width()
                                                   , nx
                                                   , ny
                                                   , D
                                                   , -1 , 1 , -1 , 1
                                                   );
    //addDisplay ( viz_in_dat );

    scale_y = 1;
    rbm = 
      new ConvolutionalRBM < double > 
      (
        nx * ny
      , N * dx * dy
      , nx
      , ny
      , dx
      , dy
      , kx
      , ky
      , 1
      , N
      , 1
      , D
      , true
      , 1
      );

    double eps = 0.0;
    for(long iter=0;iter<1;iter++)
    {
        rbm -> init(0);
        rbm -> cd(1,eps,0);
        std::cout << iter << '\t' << rbm -> final_error << std::endl;
    }

    for(long n=0,k=0;n<1;n++)
    {
        double max_value;
        for(long l=0;l<N;l++)
        {
            for(long y=0;y<dy/fy;y++)
            {
                for(long x=0;x<dx/fx;x++,k++)
                {
                    max_value = -1000000;
                    for(long _y=0;_y<fy;_y++)
                    {
                        for(long _x=0;_x<fx;_x++)
                        {
                            max_value = (max_value>rbm->hid[n*N*dx*dy+l*dx*dy+(fy*y+_y)*dx+(fx*x+_x)])
                                      ?  max_value:rbm->hid[n*N*dx*dy+l*dx*dy+(fy*y+_y)*dx+(fx*x+_x)];
                        }
                    }
                    DAT[k] = max_value;
                }
            }
        }
    }

    rbm2 = 
      new ConvolutionalRBM < double > 
      (
        N  * nx1 * ny1
      , N1 * dx1 * dy1
      , nx1
      , ny1
      , dx1
      , dy1
      , kx1
      , ky1
      , N
      , N1
      , 1
      , DAT
      , false
      , 2
      );

    double eps1 = 0.0;
    for(long iter=0;iter<1;iter++)
    {
        rbm2 -> init(0);
        rbm2 -> cd(1,eps1,0);
        std::cout << iter << '\t' << rbm2 -> final_error << std::endl;
    }

    for(long n=0,k=0;n<N;n++)
    {
        double max_value;
        for(long l=0;l<N1;l++)
        {
            for(long y=0;y<dy1/fy1;y++)
            {
                for(long x=0;x<dx1/fx1;x++,k++)
                {
                    max_value = -1000000;
                    for(long _y=0;_y<fy1;_y++)
                    {
                        for(long _x=0;_x<fx1;_x++)
                        {
                            max_value = (max_value>rbm2->hid[n*N1*dx1*dy1+l*dx1*dy1+(fy1*y+_y)*dx1+(fx1*x+_x)])
                                      ?  max_value:rbm2->hid[n*N1*dx1*dy1+l*dx1*dy1+(fy1*y+_y)*dx1+(fx1*x+_x)];
                        }
                    }
                    DAT2[k] = max_value;
                }
            }
        }
    }

    rbm3 = 
      new ConvolutionalRBM < double > 
      (
        N1 * nx2 * ny2
      , N2 * dx2 * dy2
      , nx2
      , ny2
      , dx2
      , dy2
      , kx2
      , ky2
      , N1
      , N2
      , 1
      , DAT2
      , false
      , 2
      );

    double eps2 = 0.0;
    for(long iter=0;iter<1;iter++)
    {
        rbm3 -> init(0);
        rbm3 -> cd(1,eps2,0);
        std::cout << iter << '\t' << rbm3 -> final_error << std::endl;
    }
  
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
                                                                    ,  0 , .25
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

    {
        VisualizeCRBMVisibleProbe < double > * viz_crbm_visible = NULL;
        viz_crbm_visible = new VisualizeCRBMVisibleProbe < double > ( rbm3
                                                                    , new CRBMConvolutionProbe < double > ( rbm3 )
                                                                    ,  1 , 1.25
                                                                    , -1 , 1
                                                                    );
        addDisplay ( viz_crbm_visible );

        VisualizeCRBMKernelProbe < double > * viz_crbm_kernel = NULL;
        viz_crbm_kernel = new VisualizeCRBMKernelProbe < double > ( rbm3
                                                                  , new CRBMConvolutionProbe < double > ( rbm3 )
                                                                  , 1.25 , 1.75
                                                                  ,  -1 , 1
                                                                  );
        addDisplay ( viz_crbm_kernel );

        VisualizeCRBMHiddenProbe < double > * viz_crbm_hidden = NULL;
        viz_crbm_hidden = new VisualizeCRBMHiddenProbe < double > ( rbm3
                                                                  , new CRBMConvolutionProbe < double > ( rbm3 )
                                                                  ,  1.75 , 2
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

  startGraphics(argc,argv,"Cat Classifier");
  return 0;
}
