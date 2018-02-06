#include <iostream>
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
#include "visualize_signal.h"
#define VISUALIZE_SIGNAL
#include "visualization.h"
#include "csv_reader.h"
#include "Perceptron.h"

Perceptron<double> * perceptron = NULL;

double * dat = NULL;
double * prct = NULL;
long num_elems = -1;
long num_in = 26;

template<typename T>
void test_signal_tracking(T * prct,long num_in,long num_elems)
{
  std::vector<long> nodes;
  nodes.push_back(num_in); // inputs
  nodes.push_back(2*num_in+1); // hidden layer
  nodes.push_back(2*num_in+1); // hidden layer
  nodes.push_back(1); // output layer
  nodes.push_back(1); // outputs
  perceptron = new Perceptron<double>(nodes);

  // set data
  set_sig_data ( perceptron , dat , prct , num_elems , num_in , num_in );

  long dim = num_in;
  long num = num_elems - num_in;

  double * in_dat = new double[dim*num];
  double * out_dat = new double[num];
  double R, th, val;
  for(long i=0;i<num;i++)
  {
    for(long j=0;j<num_in;j++)
    {
      val = prct[i+j];
      val = (val - (-0.05)) / (0.05 - (-0.05));
      val = (val<0)?0:val;
      val = (val>1)?1:val;
      in_dat[dim*i+j] = val;
    }
    val = prct[i+num_in];
    val = (val - (-0.05)) / (0.05 - (-0.05));
    val = (val<0)?0:val;
    val = (val>1)?1:val;
    out_dat[i] = val;
  }
  perceptron -> alpha = .1;
  perceptron -> sigmoid_type = 1;
  perceptron -> train(0,.1,100000,num,dim,1,in_dat,out_dat,true);
}

int main(int argc,char ** argv)
{
    std::cout << "Running simple signal tracking demo" << std::endl;

    // load input
    if(argc>0)
    {
      csvReader<double> csv_reader;
      dat = csv_reader.read_data_floating_point(argv[1],5,1000);
      num_elems = csv_reader.get_size();
      prct = calculate_percent_change(dat,num_elems);
    }
    else
    {
      std::cout << "Please specify input name [csv format]." << std::endl;
      exit(1);
    }

    // run test
    new boost::thread(test_signal_tracking<double>,prct,num_in,num_elems);

    // start graphics
    startGraphics(argc,argv,"Signal Tracking Demo");
    return 0;
}

