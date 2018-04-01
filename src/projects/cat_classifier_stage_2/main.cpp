#include <iostream>
#include <boost/thread.hpp>
#include "readBMP.h"
#include "Perceptron.h"
#include "ActivationProbe.h"
#include "MergePerceptrons.h"
#include "AutoEncoderConstructor.h"
#include "visualization.h"
#include "snapshot.h"

long nx = 1600;
long ny = 1200;

long kx = 9;
long ky = 9;
long dx = nx - (kx/2)*2;
long dy = ny - (ky/2)*2;
long N = 8*2;

long fx = 2;
long fy = 2;

long nx1 = dx/fx;
long ny1 = dy/fy;
long kx1 = 9;
long ky1 = 9;
long dx1 = nx1 - (kx1/2)*2;
long dy1 = ny1 - (ky1/2)*2;
long N1 = (8)*3;

long fx1 = 2;
long fy1 = 2;

long nx2 = dx1/fx1;
long ny2 = dy1/fy1;
long kx2 = 9;
long ky2 = 9;
long dx2 = nx2 - (kx2/2)*2;
long dy2 = ny2 - (ky2/2)*2;
long N2 = (8)*4;

long fx2 = 2;
long fy2 = 2;

long nx3 = dx2/fx2;
long ny3 = dy2/fy2;
long kx3 = 9;
long ky3 = 9;
long dx3 = nx3 - (kx3/2)*2;
long dy3 = ny3 - (ky3/2)*2;
long N3 = (8)*5;





long _nx = 128;
long _ny = 128;

long _kx = 9;
long _ky = 9;
long _dx = _nx - (_kx/2)*2;
long _dy = _ny - (_ky/2)*2;
long _N = 8*2;

long _fx = 2;
long _fy = 2;
double * DAT = NULL;

long _nx1 = _dx/_fx;
long _ny1 = _dy/_fy;
long _kx1 = 9;
long _ky1 = 9;
long _dx1 = _nx1 - (_kx1/2)*2;
long _dy1 = _ny1 - (_ky1/2)*2;
long _N1 = (8)*3;

long _fx1 = 2;
long _fy1 = 2;
double * DAT2 = NULL;

long _nx2 = _dx1/_fx1;
long _ny2 = _dy1/_fy1;
long _kx2 = 9;
long _ky2 = 9;
long _dx2 = _nx2 - (_kx2/2)*2;
long _dy2 = _ny2 - (_ky2/2)*2;
long _N2 = (8)*4;

long _fx2 = 2;
long _fy2 = 2;
double * DAT3 = NULL;

long _nx3 = _dx2/_fx2;
long _ny3 = _dy2/_fy2;
long _kx3 = 9;
long _ky3 = 9;
long _dx3 = _nx3 - (_kx3/2)*2;
long _dy3 = _ny3 - (_ky3/2)*2;
long _N3 = (8)*5;

ConvolutionalRBM < double > * rbm = NULL;

ConvolutionalRBM < double > * rbm2 = NULL;

ConvolutionalRBM < double > * rbm3 = NULL;

ConvolutionalRBM < double > * rbm4 = NULL;

std::string snapshots =  "";

std::string rbm1_suffix = "../cat_classifier/rbm-1.crbm";

std::string rbm2_suffix = "../cat_classifier/rbm-2.crbm";

std::string rbm3_suffix = "../cat_classifier/rbm-3.crbm";

std::string rbm4_suffix = "../cat_classifier/rbm-4.crbm";

int main(int argc,char ** argv)
{
  if(argc>1)
  {
    snapshots = argv[2];
  }
  else
  {
    std::cout << "Please specify snapshot directory." << std::endl;
    exit(1);
  }
  if(argc>0)
  {
    Image * dat = new Image();
    ImageLoad(argv[1],dat);
    double * D = dat->get_doubles(_nx,_ny,2,2);
    std::cout << dat->get_size() << std::endl;
    long nsamp = dat->get_size()/(_nx*_ny);
    double * IN = new double[_nx*_ny];
    DAT  = new double[1*_N*(_dx/_fx)*(_dy/_fy)];
    DAT2 = new double[1*_N1*(_dx1/_fx1)*(_dy1/_fy1)];
    DAT3 = new double[1*_N2*(_dx2/_fx2)*(_dy2/_fy2)];
    VisualizeDataArray < double > * viz_in_dat = NULL;
    viz_in_dat = new VisualizeDataArray < double > ( dat->get_size()
                                                   , _nx*_ny
                                                   , dat->get_width()
                                                   , _nx
                                                   , _ny
                                                   , D
                                                   , -1 , 1 , -1 , 1
                                                   );
    addDisplay ( viz_in_dat );

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
      , NULL
      , true
      , 1
      );
    load_from_file(rbm,snapshots+rbm1_suffix);
    rbm -> change_input_size ( _nx , _ny , _dx , _dy , 1 , IN );
    
    double eps = 0.0;
    for(long iter=0;iter<1;iter++)
    {
        rbm -> init(0);
        rbm -> cd(1,eps,0);
    }

    for(long n=0,k=0;n<1;n++)
    {
        double max_value;
        for(long l=0;l<_N;l++)
        {
            for(long y=0;y<_dy/_fy;y++)
            {
                for(long x=0;x<_dx/_fx;x++,k++)
                {
                    max_value = -1000000;
                    for(long _y=0;_y<_fy;_y++)
                    {
                        for(long _x=0;_x<_fx;_x++)
                        {
                            max_value = (max_value>rbm->hid[n*_dx*_dy*_N+l*_dx*_dy+(_fy*y+_y)*_dx+(_fx*x+_x)])
                                      ?  max_value:rbm->hid[n*_dx*_dy*_N+l*_dx*_dy+(_fy*y+_y)*_dx+(_fx*x+_x)];
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
      , NULL
      , false
      , 2
      );
    load_from_file(rbm2,snapshots+rbm2_suffix);
    rbm2 -> change_input_size ( _nx1 , _ny1 , _dx1 , _dy1 , 1 , DAT );

    double eps1 = 0.0;
    for(long iter=0;iter<1;iter++)
    {
        rbm2 -> init(0);
        rbm2 -> cd(1,eps1,0);
    }

    for(long n=0,k=0;n<1;n++)
    {
        double max_value;
        for(long l=0;l<_N1;l++)
        {
            for(long y=0;y<_dy1/_fy1;y++)
            {
                for(long x=0;x<_dx1/_fx1;x++,k++)
                {
                    max_value = -1000000;
                    for(long _y=0;_y<_fy1;_y++)
                    {
                        for(long _x=0;_x<_fx1;_x++)
                        {
                            max_value = (max_value>rbm2->hid[n*_dx1*_dy1*_N1+l*_dx1*_dy1+(_fy1*y+_y)*_dx1+(_fx1*x+_x)])
                                      ?  max_value:rbm2->hid[n*_dx1*_dy1*_N1+l*_dx1*_dy1+(_fy1*y+_y)*_dx1+(_fx1*x+_x)];
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
      , NULL
      , false
      , 2
      );
    load_from_file(rbm3,snapshots+rbm3_suffix);
    rbm3 -> change_input_size ( _nx2 , _ny2 , _dx2 , _dy2 , 1 , DAT2 );
    
    double eps2 = 0.0;
    for(long iter=0;iter<1;iter++)
    {
        rbm3 -> init(0);
        rbm3 -> cd(1,eps2,0);
    }

    for(long n=0,k=0;n<1;n++)
    {
        double max_value;
        for(long l=0;l<_N2;l++)
        {
            for(long y=0;y<_dy2/_fy2;y++)
            {
                for(long x=0;x<_dx2/_fx2;x++,k++)
                {
                    max_value = -1000000;
                    for(long _y=0;_y<_fy2;_y++)
                    {
                        for(long _x=0;_x<_fx2;_x++)
                        {
                            max_value = (max_value>rbm3->hid[n*_dx2*_dy2*_N2+l*_dx2*_dy2+(_fy2*y+_y)*_dx2+(_fx2*x+_x)])
                                      ?  max_value:rbm3->hid[n*_dx2*_dy2*_N2+l*_dx2*_dy2+(_fy2*y+_y)*_dx2+(_fx2*x+_x)];
                        }
                    }
                    DAT3[k] = max_value;
                }
            }
        }
    }

    rbm4 = 
      new ConvolutionalRBM < double > 
      (
        N2 * nx3 * ny3
      , N3 * dx3 * dy3
      , nx3
      , ny3
      , dx3
      , dy3
      , kx3
      , ky3
      , N2
      , N3
      , 1
      , NULL
      , false
      , 2
      );
    load_from_file(rbm4,snapshots+rbm4_suffix);
    rbm4 -> change_input_size ( _nx3 , _ny3 , _dx3 , _dy3 , 1 , DAT3 );

    double eps3 = 0.0;
    for(long iter=0;iter<1;iter++)
    {
        rbm4 -> init(0);
        rbm4 -> cd(1,eps3,0);
    }

    double * DATA = new double[nsamp*rbm4->h];

    for(long t=0;t<nsamp;t++)
    {

      std::cout << "x=" << t << "/" << nsamp << std::endl;

      for(long x=0,k=0;x<_nx;x++)
        for(long y=0;y<_ny;y++,k++)
        {
          IN[k] = D[t*_nx*_ny+k];
        }

      double eps = 0.0;
      for(long iter=0;iter<1;iter++)
      {
          rbm -> init(0);
          rbm -> cd(1,eps,0);
      }

      for(long n=0,k=0;n<1;n++)
      {
          double max_value;
          for(long l=0;l<_N;l++)
          {
              for(long y=0;y<_dy/_fy;y++)
              {
                  for(long x=0;x<_dx/_fx;x++,k++)
                  {
                      max_value = -1000000;
                      for(long _y=0;_y<_fy;_y++)
                      {
                          for(long _x=0;_x<_fx;_x++)
                          {
                              max_value = (max_value>rbm->hid[n*_dx*_dy*_N+l*_dx*_dy+(_fy*y+_y)*_dx+(_fx*x+_x)])
                                        ?  max_value:rbm->hid[n*_dx*_dy*_N+l*_dx*_dy+(_fy*y+_y)*_dx+(_fx*x+_x)];
                          }
                      }
                      DAT[k] = max_value;
                  }
              }
          }
      }

      double eps1 = 0.0;
      for(long iter=0;iter<1;iter++)
      {
          rbm2 -> init(0);
          rbm2 -> cd(1,eps1,0);
      }

      for(long n=0,k=0;n<1;n++)
      {
          double max_value;
          for(long l=0;l<_N1;l++)
          {
              for(long y=0;y<_dy1/_fy1;y++)
              {
                  for(long x=0;x<_dx1/_fx1;x++,k++)
                  {
                      max_value = -1000000;
                      for(long _y=0;_y<_fy1;_y++)
                      {
                          for(long _x=0;_x<_fx1;_x++)
                          {
                              max_value = (max_value>rbm2->hid[n*_dx1*_dy1*_N1+l*_dx1*_dy1+(_fy1*y+_y)*_dx1+(_fx1*x+_x)])
                                        ?  max_value:rbm2->hid[n*_dx1*_dy1*_N1+l*_dx1*_dy1+(_fy1*y+_y)*_dx1+(_fx1*x+_x)];
                          }
                      }
                      DAT2[k] = max_value;
                  }
              }
          }
      }
    
      double eps2 = 0.0;
      for(long iter=0;iter<1;iter++)
      {
          rbm3 -> init(0);
          rbm3 -> cd(1,eps2,0);
      }

      for(long n=0,k=0;n<1;n++)
      {
          double max_value;
          for(long l=0;l<_N2;l++)
          {
              for(long y=0;y<_dy2/_fy2;y++)
              {
                  for(long x=0;x<_dx2/_fx2;x++,k++)
                  {
                      max_value = -1000000;
                      for(long _y=0;_y<_fy2;_y++)
                      {
                          for(long _x=0;_x<_fx2;_x++)
                          {
                              max_value = (max_value>rbm3->hid[n*_dx2*_dy2*_N2+l*_dx2*_dy2+(_fy2*y+_y)*_dx2+(_fx2*x+_x)])
                                        ?  max_value:rbm3->hid[n*_dx2*_dy2*_N2+l*_dx2*_dy2+(_fy2*y+_y)*_dx2+(_fx2*x+_x)];
                          }
                      }
                      DAT3[k] = max_value;
                  }
              }
          }
      }

      double eps3 = 0.0;
      for(long iter=0;iter<1;iter++)
      {
          rbm4 -> init(0);
          rbm4 -> cd(1,eps3,0);
      }

      for(long x=0;x<rbm4->h;x++)
      {
          DATA[t*rbm4->h+x] = rbm4->hid[x];
      }

    }

    long N0  = rbm4->h;
    long N1 = .8*N0;
    long N2 = .8*N1;
    long N3 = .8*N2;
    long N4 = .8*N3;
    long N5 = .8*N4;
    long N6 = N5-4;
    long N7 = N6-3;
    long N8 = N7-3;

    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N0;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  1;
            range_max[x] = -1;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (DATA[k]<range_min[x])?DATA[k]:range_min[x];
            range_max[x] = (DATA[k]>range_max[x])?DATA[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            DATA[k] = (DATA[k]-range_min[x])/(range_max[x]-range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
            range_min[x] =  1;
            range_max[x] = -1;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (DATA[k]<range_min[x])?DATA[k]:range_min[x];
            range_max[x] = (DATA[k]>range_max[x])?DATA[k]:range_max[x];
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    RBM<double> * inner_rbm1 = new RBM<double>(N0,N1,nsamp,DATA);
    for(long iter=0;iter<100000;iter++)
    {
        inner_rbm1 -> init(0);
        inner_rbm1 -> cd(1,0.1,0);
        std::cout << iter << '\t' << inner_rbm1->final_error << std::endl;
        //std::cout << "1" << '\t' << N1 << std::endl;
    }
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N1;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  10000000000000;
            range_max[x] = -10000000000000;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (inner_rbm1->hid[k]<range_min[x])?inner_rbm1->hid[k]:range_min[x];
            range_max[x] = (inner_rbm1->hid[k]>range_max[x])?inner_rbm1->hid[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            inner_rbm1->hid[k] = (inner_rbm1->hid[k] - range_min[x])/(range_max[x] - range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    RBM<double> * inner_rbm2 = new RBM<double>(N1,N2,nsamp,inner_rbm1->hid);
    for(long iter=0;iter<10000;iter++)
    {
        inner_rbm2 -> init(0);
        inner_rbm2 -> cd(1,0.1,0);
        //std::cout << "2" << '\t' << N2 << std::endl;
    }
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N2;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  10000000000000;
            range_max[x] = -10000000000000;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (inner_rbm2->hid[k]<range_min[x])?inner_rbm2->hid[k]:range_min[x];
            range_max[x] = (inner_rbm2->hid[k]>range_max[x])?inner_rbm2->hid[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            inner_rbm2->hid[k] = (inner_rbm2->hid[k] - range_min[x])/(range_max[x] - range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    RBM<double> * inner_rbm3 = new RBM<double>(N2,N3,nsamp,inner_rbm2->hid);
    for(long iter=0;iter<10000;iter++)
    {
        inner_rbm3 -> init(0);
        inner_rbm3 -> cd(1,0.1,0);
        //std::cout << "3" << '\t' << N3 << std::endl;
    }
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N3;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  10000000000000;
            range_max[x] = -10000000000000;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (inner_rbm3->hid[k]<range_min[x])?inner_rbm3->hid[k]:range_min[x];
            range_max[x] = (inner_rbm3->hid[k]>range_max[x])?inner_rbm3->hid[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            inner_rbm3->hid[k] = (inner_rbm3->hid[k] - range_min[x])/(range_max[x] - range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    RBM<double> * inner_rbm4 = new RBM<double>(N3,N4,nsamp,inner_rbm3->hid);
    for(long iter=0;iter<10000;iter++)
    {
        inner_rbm4 -> init(0);
        inner_rbm4 -> cd(1,0.1,0);
        //std::cout << "4" << '\t' << N4 << std::endl;
    }
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N4;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  10000000000000;
            range_max[x] = -10000000000000;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (inner_rbm4->hid[k]<range_min[x])?inner_rbm4->hid[k]:range_min[x];
            range_max[x] = (inner_rbm4->hid[k]>range_max[x])?inner_rbm4->hid[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            inner_rbm4->hid[k] = (inner_rbm4->hid[k] - range_min[x])/(range_max[x] - range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    RBM<double> * inner_rbm5 = new RBM<double>(N4,N5,nsamp,inner_rbm4->hid);
    for(long iter=0;iter<10000;iter++)
    {
        inner_rbm5 -> init(0);
        inner_rbm5 -> cd(1,0.1,0);
        //std::cout << "5" << '\t' << N5 << std::endl;
    }
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N5;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  10000000000000;
            range_max[x] = -10000000000000;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (inner_rbm5->hid[k]<range_min[x])?inner_rbm5->hid[k]:range_min[x];
            range_max[x] = (inner_rbm5->hid[k]>range_max[x])?inner_rbm5->hid[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            inner_rbm5->hid[k] = (inner_rbm5->hid[k] - range_min[x])/(range_max[x] - range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    RBM<double> * inner_rbm6 = new RBM<double>(N5,N6,nsamp,inner_rbm4->hid);
    for(long iter=0;iter<10000;iter++)
    {
        inner_rbm6 -> init(0);
        inner_rbm6 -> cd(1,0.1,0);
        //std::cout << "6" << '\t' << N6 << std::endl;
    }
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N6;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  10000000000000;
            range_max[x] = -10000000000000;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (inner_rbm6->hid[k]<range_min[x])?inner_rbm6->hid[k]:range_min[x];
            range_max[x] = (inner_rbm6->hid[k]>range_max[x])?inner_rbm6->hid[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            inner_rbm6->hid[k] = (inner_rbm6->hid[k] - range_min[x])/(range_max[x] - range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    RBM<double> * inner_rbm7 = new RBM<double>(N6,N7,nsamp,inner_rbm4->hid);
    for(long iter=0;iter<10000;iter++)
    {
        inner_rbm7 -> init(0);
        inner_rbm7 -> cd(1,0.1,0);
        //std::cout << "6" << '\t' << N6 << std::endl;
    }
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N7;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  10000000000000;
            range_max[x] = -10000000000000;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (inner_rbm7->hid[k]<range_min[x])?inner_rbm7->hid[k]:range_min[x];
            range_max[x] = (inner_rbm7->hid[k]>range_max[x])?inner_rbm7->hid[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            inner_rbm7->hid[k] = (inner_rbm7->hid[k] - range_min[x])/(range_max[x] - range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    RBM<double> * inner_rbm8 = new RBM<double>(N7,N8,nsamp,inner_rbm4->hid);
    for(long iter=0;iter<10000;iter++)
    {
        inner_rbm8 -> init(0);
        inner_rbm8 -> cd(1,0.1,0);
        //std::cout << "6" << '\t' << N6 << std::endl;
    }
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
        long N = N8;
        float * range_min = new float[N];
        float * range_max = new float[N];
        for(long x=0;x<N;x++)
        {
            range_min[x] =  10000000000000;
            range_max[x] = -10000000000000;
        }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            range_min[x] = (inner_rbm8->hid[k]<range_min[x])?inner_rbm8->hid[k]:range_min[x];
            range_max[x] = (inner_rbm8->hid[k]>range_max[x])?inner_rbm8->hid[k]:range_max[x];
          }
        for(long i=0,k=0;i<nsamp;i++)
          for(long x=0;x<N;x++,k++)
          {
            inner_rbm8->hid[k] = (inner_rbm8->hid[k] - range_min[x])/(range_max[x] - range_min[x]);
          }
        for(long x=0;x<N;x++)
        {
          std::cout << range_min[x] << '\t' << range_max[x] << std::endl;
        }
    }
    {
        std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
        long N = N8;
        for(long i=0,k=0;i<nsamp;i++)
        {
          for(long x=0;x<N;x++,k++)
          {
            std::cout << inner_rbm8->hid[k] << ' ';
          }
          std::cout << std::endl;
        }
    }


  }

  startGraphics(argc,argv,"Cat Classifier - Stage 2");
  return 0;
}

