#include <iostream>
#include <boost/thread.hpp>
//#define VISUALIZE_DATA_ARRAY
#define VISUALIZE_RBM_RECONSTRUCTION
#include "visualization.h"
#include "readBMP.h"
#include "RBM.h"

RBM<double> * rbm = NULL;

// learns to classify hand written digits
void test_rbm_mnist()
{
  double eps = 0.2;
  for(long i=0;i<100000;i++)
  {
    rbm -> init(0);
    rbm -> cd(1,eps,0);
  }
}

int main(int argc,char ** argv)
{
    std::cout << "Running simple ann classification demo" << std::endl;

    // load input
    if(argc>0)
    {
      Image * dat = new Image();
      ImageLoad(argv[1],dat);
      long nx = 20;
      long ny = 20;
      double * D = dat->get_doubles(nx,ny);
      //set_viz_data ( nx*ny*10*5*100
      //             , nx*ny
      //             , nx
      //             , ny
      //             , D
      //             );
      rbm = new RBM<double>(nx*ny,nx*ny,10*5*100,D);
      set_rbm_data ( rbm
                   , nx*ny*10*5*100
                   , nx*ny
                   , nx
                   , ny
                   , D
                   );
    }
    else
    {
      std::cout << "Please specify input name [bmp format]." << std::endl;
      exit(1);
    }

    // run test
    new boost::thread(test_rbm_mnist);

    // start graphics
    startGraphics(argc,argv,"Simple RBM Example");
    return 0;
}

