#include <iostream>
#include "ConvolutionalRBM.h"
#include "text.h"
#include "readBMP.h"
#include "visualization.h"
#include "sep_reader.h"
#include "snapshot.h"

void clear() 
{
  // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
  std::cout << "\x1B[2J\x1B[H";
}

ConvolutionalRBM<double> * rbm = NULL;

ConvolutionalRBM<double> * rbm2 = NULL;

ConvolutionalRBM<double> * rbm3 = NULL;

std::string snapshots =  "";

std::string rbm1_suffix = "/rbm-1.crbm";

std::string rbm2_suffix = "/rbm-2.crbm";

std::string rbm3_suffix = "/rbm-3.crbm";

std::string filename_dat = "/home/antonk/data/oxy.hdr";

std::string filename_afi = "/home/antonk/data/oxy-afi.hdr";

SEPReader reader_dat(filename_dat.c_str());

SEPReader reader_afi(filename_afi.c_str());

long nx = 18;
long ny = 18;
long kx = 3;
long ky = 3;
long kx2 = 3;
long ky2 = 3;
long kx3 = 3;
long ky3 = 3;
long dx = nx-2*(kx/2);
long dy = ny-2*(ky/2);
long _dx = dx/2;
long _dy = dy/2;
long dx2 = _dx-2*(kx2/2);
long dy2 = _dy-2*(ky2/2);
long __dx = dx2/2;
long __dy = dy2/2;
long dx3 = __dx-2*(kx3/2);
long dy3 = __dy-2*(ky3/2);
long M = 1;
long K = 80+1; // 80 = 16*5
long K2 = 160+1;
long K3 = 320+1;
long n = 10;

long mode = 3;

void train()
{
  while(CONTINUE)
  {

    switch(mode)
    {
        case 3:
        {
          double eps = 100e-1;
          bool init = true;
          double init_error;
          double prev_error;
          init = true;
          for(long i=0;i<10&&CONTINUE;i++)
          {
            rbm3 -> init(0);
            rbm3 -> cd(10,eps,0);
            if(init)
            {
              init = false;
              init_error = rbm3->final_error;
            }
            if(i%1==0)
            {
              std::cout << n << '\t' << init_error << '\t' << rbm3 -> final_error << std::endl;
            }
          }
          dump_to_file(rbm3,snapshots+rbm3_suffix);
        }
        break;
        
        case 2:
        {
          double eps = 1e-2;
          bool init = true;
          double init_error;
          double prev_error;
          init = true;
          for(long i=0;i<10&&CONTINUE;i++)
          {
            rbm2 -> init(0);
            rbm2 -> cd(1,eps,0);
            if(init)
            {
              init = false;
              init_error = rbm2->final_error;
            }
            if(i%1==0)
            {
              std::cout << n << '\t' << init_error << '\t' << rbm2 -> final_error << std::endl;
            }
          }
          dump_to_file(rbm2,snapshots+rbm2_suffix);
        }
        break;

        case 1:
        {
          double eps = 1e-3;
          bool init = true;
          double init_error;
          init = true;
          for(long i=0;i<30&&CONTINUE;i++)
          {
            rbm -> init(0);
            rbm -> cd(1,eps,0);
            if(init)
            {
              init = false;
              init_error = rbm->final_error;
            }
            if(i%1==0)
            {
              std::cout << n << '\t' << init_error << '\t' << rbm -> final_error << std::endl;
            }
          }
          dump_to_file(rbm,snapshots+rbm1_suffix);
        }
        break;
    }

  }
}

struct point
{
  int x,y,z;
  point(int _x,int _y,int _z)
    : x(_x) , y(_y) , z(_z)
  {

  }
};

int main(int argc,char ** argv)
{
  std::cout << "Convolutional RBM Test." << std::endl;
  srand(time(0));
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
    //long X = reader_dat.n3/3;
    //long NX = reader_dat.n3 - 2*X;
    long X = reader_dat.n3/3;
    long NX = 10;//reader_dat.n3 - 2*X;
    double * dat = new double[NX*M*n*nx*ny];
    for(long iter=0,k=0;iter<NX;iter++)
    {
      float * slice = new float[reader_dat.n1*reader_dat.n2];
      float * afi_slice = new float[reader_dat.n1*reader_dat.n2];
      reader_dat.read_sepval ( &slice[0]
                             , reader_dat.o1
                             , reader_dat.o2
                             , reader_dat.o3 + X + iter
                             , reader_dat.n1
                             , reader_dat.n2
                             , 1 // reader.n3
                             );
      reader_afi.read_sepval ( &afi_slice[0]
                             , reader_afi.o1
                             , reader_afi.o2
                             , reader_afi.o3 + X + iter
                             , reader_afi.n1
                             , reader_afi.n2
                             , 1 // reader.n3
                             );
      std::vector<point> faults;
      for(long x=0,t=0;x<1;x++)
      {
          for(long y=0;y<reader_afi.n2;y++)
          {
              for(long z=0;z<reader_afi.n1;z++,t++)
              {
                  if(z>reader_afi.n1/3&&z<2*reader_afi.n1/3)
                  {
                      if(afi_slice[t] > 0.4)
                      {
                          faults.push_back(point(x,y,z));
                      }
                  }
              }
          }
      }
      double min_val =  10000000;
      double max_val = -10000000;
      for(long i=0;i<1*reader_dat.n1*reader_dat.n2;i++)
      {
          min_val = (min_val<slice[i])?min_val:slice[i];
          max_val = (max_val>slice[i])?max_val:slice[i];
      }
      for(long i=0;i<1*reader_dat.n1*reader_dat.n2;i++)
      {
          slice[i] = (slice[i]-min_val)/(max_val-min_val);
      }
      for(long i=0;i<n;i++)
      {
          long X,Y,Z;
          while(true)
          {
              long ind = rand()%faults.size();
              Z = faults[ind].z - nx/2;
              Y = faults[ind].y - ny/2;
              X = faults[ind].x;
              if(X>=0&&X<1&&Z>=0&&Z+ny<reader_afi.n1&&Y>=0&&Y+nx<reader_afi.n2)break;
          };
          for(long m=0;m<M;m++)
          for(long y=0;y<ny;y++)
          {
              for(long x=0;x<nx;x++,k++)
              {
                  dat[k] = slice[X*reader_dat.n1*reader_dat.n2 + reader_dat.n1*(x+Y) + (y+Z)];
              }
          }
      }
      delete [] slice;
      delete [] afi_slice;
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
                                       , NX*n
                                       , dat
                                       );

    double eps = 1e-2;
    bool init = true;
    double init_error;
    init = true;
    //load_from_file(rbm,snapshots+rbm1_suffix);
    for(long i=0;i<1/*1000*/;i++)
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
    //dump_to_file(rbm,snapshots+rbm1_suffix);
    
    double * dat2 = new double[NX*K*n*_dx*_dy];
    // max pooling 
    for(long i=0,k=0;i<NX*n;i++)
    {
        double max_value;
        for(long l=0;l<K;l++)
        {
            for(long y=0;y<_dy;y++)
            {
                for(long x=0;x<_dx;x++,k++)
                {
                    max_value = -1000000;
                    for(long _y=0;_y<2;_y++)
                    {
                        for(long _x=0;_x<2;_x++)
                        {
                            max_value = (max_value>rbm->hid[i*K*dx*dy+l*dx*dy+(2*y+_y)*dx+(2*x+_x)])
                                      ?  max_value:rbm->hid[i*K*dx*dy+l*dx*dy+(2*y+_y)*dx+(2*x+_x)];
                        }
                    }
                    dat2[k] = max_value;
                }
            }
        }
    }

    rbm2 = new ConvolutionalRBM<double> ( K*_dx*_dy
                                        , K2*dx2*dy2
                                        , _dx
                                        , _dy
                                        , dx2
                                        , dy2
                                        , kx2
                                        , ky2
                                        , K
                                        , K2
                                        , NX*n
                                        , dat2
                                        , false
                                        );
    for(long i=0;i<1/*1000*/;i++)
    {
      rbm2 -> init(0);
      rbm2 -> cd(1,eps,0);
    }
    //load_from_file(rbm2,snapshots+rbm2_suffix);




    std::cout << nx << '\t' << ny << std::endl;
    std::cout << dx << '\t' << dy << std::endl;
    std::cout << _dx << '\t' << _dy << std::endl;
    std::cout << dx2 << '\t' << dy2 << std::endl;
    std::cout << __dx << '\t' << __dy << std::endl;
    std::cout << dx3 << '\t' << dy3 << std::endl;


    double * dat3 = new double[NX*K2*n*__dx*__dy];
    // max pooling 
    for(long i=0,k=0;i<NX*n;i++)
    {
        double max_value;
        for(long l=0;l<K2;l++)
        {
            for(long y=0;y<__dy;y++)
            {
                for(long x=0;x<__dx;x++,k++)
                {
                    max_value = -1000000;
                    for(long _y=0;_y<2;_y++)
                    {
                        for(long _x=0;_x<2;_x++)
                        {
                            max_value = (max_value>rbm2->hid[i*K2*dx2*dy2+l*dx2*dy2+(2*y+_y)*_dx+(2*x+_x)])
                                      ?  max_value:rbm2->hid[i*K2*dx2*dy2+l*dx2*dy2+(2*y+_y)*_dx+(2*x+_x)];
                        }
                    }
                    dat3[k] = max_value;
                }
            }
        }
    }

    rbm3 = new ConvolutionalRBM<double> ( K2*__dx*__dy
                                        , K3*dx3*dy3
                                        , __dx
                                        , __dy
                                        , dx3
                                        , dy3
                                        , kx3
                                        , ky3
                                        , K2
                                        , K3
                                        , NX*n
                                        , dat3
                                        , false
                                        );
    for(long i=0;i<1/*1000*/;i++)
    {
      rbm3 -> init(0);
      rbm3 -> cd(1,eps,0);
    }
    //load_from_file(rbm3,snapshots+rbm3_suffix);

    /*
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
    */
    
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

