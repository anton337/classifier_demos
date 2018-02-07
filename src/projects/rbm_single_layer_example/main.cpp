#include <iostream>
#include <boost/thread.hpp>
#define VISUALIZE_DATA_ARRAY
#include "visualization.h"
#include "readBMP.h"
#include "RBM.h"

// learns to classify hand written digits
void test_rbm_mnist()
{
  RBM<double> * rbm = new RBM<double>(n_x*n_y,n_x*n_y,10*5*100,viz_dat);
  for(long i=0;i<100000;i++)
  {
    rbm -> init(0);
    rbm -> cd(1,0.1,0);
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
      set_viz_data(nx*ny*10*5*100,nx*ny,nx,ny,dat->get_doubles(nx,ny));
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

