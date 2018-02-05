#include <iostream>
#include <boost/thread.hpp>
#include "visualization.h"
#include "readBMP.h"
#include "Perceptron.h"

void test_xor()
{
  std::vector<long> nodes;
  nodes.push_back(2); // inputs
  nodes.push_back(3); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  Perceptron<double> * perceptron = new Perceptron<double>(nodes);
  double in_dat[8];
  in_dat[0] = 0;
  in_dat[1] = 0;
  in_dat[2] = 0;
  in_dat[3] = 1;
  in_dat[4] = 1;
  in_dat[5] = 0;
  in_dat[6] = 1;
  in_dat[7] = 1;
  double out_dat[4];
  out_dat[0] = 0;
  out_dat[1] = 1;
  out_dat[2] = 1;
  out_dat[3] = 0;
  perceptron -> train(0,.1,100000,4,2,1,in_dat,out_dat);
}

int main(int argc,char ** argv)
{
    std::cout << "Running simple xor test" << std::endl;

    // run test
    boost::thread * T = new boost::thread(test_xor);

    T -> join();

    return 0;
}

