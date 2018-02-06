#include <iostream>
#include <boost/thread.hpp>
#define VISUALIZE_MODEL_OUTPUT
#include "visualization.h"
#include "readBMP.h"
#include "Perceptron.h"

Perceptron<double> * perceptron = NULL;

void test_spiral_fitting()
{
  std::vector<long> nodes;
  nodes.push_back(2); // inputs
  nodes.push_back(10); // hidden layer
  nodes.push_back(10); // hidden layer
  nodes.push_back(10); // hidden layer
  nodes.push_back(10); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  perceptron = new Perceptron<double>(nodes);

  // set up visualization
  set_mod_data(perceptron,-1,1,100,-1,1,100);

  long dim = 2;
  long num = 10000;
  double * in_dat = new double[dim*num];
  double * out_dat = new double[num];
  double R, th;
  for(long i=0;i<num;i++)
  {
    R = (double)i/num;
    th = 4*M_PI*R;
    if(i%2==0)
    {
      in_dat[dim*i+0] = R*cos(th);
      in_dat[dim*i+1] = R*sin(th);
      out_dat[i] = 1;
    }
    else
    {
      in_dat[dim*i+0] = -R*cos(th);
      in_dat[dim*i+1] = -R*sin(th);
      out_dat[i] = 0;
    }
  }
  perceptron -> alpha = 1;
  perceptron -> train(0,.1,100000,num,dim,1,in_dat,out_dat,true);
}

int main(int argc,char ** argv)
{
    std::cout << "Running simple xor test" << std::endl;

    // run test
    boost::thread * T = new boost::thread(test_spiral_fitting);

    // start graphics
    startGraphics(argc,argv,"Spiral Fitting");
    return 0;
}

