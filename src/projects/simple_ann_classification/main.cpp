#include <iostream>
#include <boost/thread.hpp>
#include "visualization.h"
#include "readBMP.h"
#include "Perceptron.h"

// learns to classify hand written digits
void test_mnist()
{
  std::vector<long> nodes;
  nodes.push_back(n_x*n_y); // inputs
  nodes.push_back(30); // hidden layer
  nodes.push_back(10); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  Perceptron<double> * perceptron = new Perceptron<double>(nodes);
  long N = n_elems/(n_x*n_y);
  double * out_dat = new double[N];
  // just recognize the digit '3'
  for(int i=0;i<N;i++)
  {
    out_dat[i] = (i/(100*5)==3)?1.0:0.0;
  }
  perceptron -> alpha = 1;
  perceptron -> train(0,.1,100000,N,n_x*n_y,1,viz_dat,out_dat,true);
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
    new boost::thread(test_mnist);

    // start graphics
    startGraphics(argc,argv,"Simple ANN Classification - MNIST Digits");
    return 0;
}

