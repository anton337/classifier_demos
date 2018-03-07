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

std::string snapshots =  "";

std::string rbm1_suffix = "/rbm-1.crbm";

std::string rbm2_suffix = "/rbm-2.crbm";

std::string filename_dat = "/home/antonk/data/oxy.hdr";

std::string filename_afi = "/home/antonk/data/oxy-afi.hdr";

SEPReader reader_dat(filename_dat.c_str());

SEPReader reader_afi(filename_afi.c_str());

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
long n = 3200;

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
      dump_to_file(rbm2,snapshots+rbm2_suffix);
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
    long X = reader_dat.n3/2;
    float * slice = new float[reader_dat.n1*reader_dat.n2];
    float * afi_slice = new float[reader_dat.n1*reader_dat.n2];
    reader_dat.read_sepval ( &slice[0]
                           , reader_dat.o1
                           , reader_dat.o2
                           , reader_dat.o3 + X
                           , reader_dat.n1
                           , reader_dat.n2
                           , 1 // reader.n3
                           );
    reader_afi.read_sepval ( &afi_slice[0]
                           , reader_afi.o1
                           , reader_afi.o2
                           , reader_afi.o3 + X
                           , reader_afi.n1
                           , reader_afi.n2
                           , 1 // reader.n3
                           );
    std::vector<std::pair<int,int> > faults;
    for(long y=0,k=0;y<reader_afi.n2;y++)
    {
        for(long z=0;z<reader_afi.n1;z++,k++)
        {
            if(z>reader_afi.n1/3&&z<2*reader_afi.n1/3)
            {
                if(afi_slice[k] > 0.4)
                {
                    faults.push_back(std::pair<int,int>(y,z));
                }
            }
        }
    }
    double min_val =  10000000;
    double max_val = -10000000;
    for(long i=0;i<reader_dat.n1*reader_dat.n2;i++)
    {
        min_val = (min_val<slice[i])?min_val:slice[i];
        max_val = (max_val>slice[i])?max_val:slice[i];
    }
    for(long i=0;i<reader_dat.n1*reader_dat.n2;i++)
    {
        slice[i] = (slice[i]-min_val)/(max_val-min_val);
    }
    double * dat = new double[M*n*nx*ny];
    for(long i=0,k=0;i<n;i++)
    {
        long Z,Y;
        while(true)
        {
            long ind = rand()%faults.size();
            Z = faults[ind].second - nx/2;
            Y = faults[ind].first - ny/2;
            if(Z>=0&&Z+ny<reader_afi.n1&&Y>=0&&Y+nx<reader_afi.n2)break;
        };
        for(long m=0;m<M;m++)
        for(long y=0;y<ny;y++)
        {
            for(long x=0;x<nx;x++,k++)
            {
                dat[k] = slice[reader_dat.n1*(x+Y) + (y+Z)];
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
    load_from_file(rbm,snapshots+rbm1_suffix);
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
    load_from_file(rbm2,snapshots+rbm2_suffix);
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
