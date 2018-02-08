#include <iostream>
#include <boost/thread.hpp>
#include "visualization.h"
#include "readBMP.h"
#include "Perceptron.h"
#include "ActivationProbe.h"
#include "snapshot.h"
#include "MergePerceptrons.h"

VisualizeDataArray < double > * viz_in_dat = NULL;

VisualizeActivationProbe < double > * viz_probe = NULL;

std::vector<Perceptron<double> * > perceptrons;

Perceptron<double> * perceptron = NULL;

std::string output_dir = "";

long n_out = 1;

// models stuff periodically
void model()
{
    while(true)
    {
        double * out = perceptron->model(viz_in_dat->n_vars,n_out,&viz_in_dat->viz_dat[viz_in_dat->viz_selection*viz_in_dat->n_vars]);
        delete [] out;
        usleep(100000);
    }
}

// learns to classify hand written digits
void test_mnist()
{
  long N = viz_in_dat->n_elems/(viz_in_dat->n_x*viz_in_dat->n_y);
  double * out_dat = new double[n_out*N];
  // just recognize the digit '3'
  int J = 0;
  while(true)
  {
    for(int i=0;i<N;i++)
    {
      for(int j=0;j<n_out;j++)
      {
        {
          out_dat[n_out*i+j] = (i/(100*5)==J)?1.0:0.0;
        }
      }
    }
    perceptrons[J] -> train(0,.01,1000,N,viz_in_dat->n_x*viz_in_dat->n_y,n_out,viz_in_dat->viz_dat,out_dat);
    std::stringstream ss;
    ss << output_dir << "/mnist-" << J << ".ann";
    dump_to_file(perceptrons[J],ss.str());
    J = (J+1)%10;
    perceptron = perceptrons[J];
    viz_probe->probe_perceptron = perceptrons[J];
  }
}

int main(int argc,char ** argv)
{
    std::cout << "Running simple ann classification demo" << std::endl;

    srand(time(0));

    // load input
    if(argc>0)
    {
      Image * dat = new Image();
      ImageLoad(argv[1],dat);
      long nx = 20;
      long ny = 20;
      std::vector<long> nodes;
      nodes.push_back(nx*ny); // inputs
      nodes.push_back(16); // hidden layer
      nodes.push_back(n_out); // output layer
      nodes.push_back(n_out); // outputs
      for(int i=0;i<10;i++)
      {
        perceptrons.push_back(new Perceptron<double>(nodes));
      }
      perceptron = perceptrons[0];
      double * D = dat->get_doubles(nx,ny);
      viz_in_dat = new VisualizeDataArray < double > ( dat->get_size()
                                                     , nx*ny
                                                     , dat->get_width()
                                                     , nx
                                                     , ny
                                                     , D
                                                     , -1 , 0 , -1 , 1
                                                     );
      viz_probe = new VisualizeActivationProbe < double > ( perceptrons[0]
                                                          , new ActivationProbe<double>(perceptrons[0],0)
                                                          , 20     , 4
                                                          , 20     , 4
                                                          //20x20  //16
                                                          , 0 , 1 , -1 , 1
                                                          );
      addDisplay ( viz_in_dat );
      addDisplay ( viz_probe  );
    }
    else
    {
      std::cout << "Please specify input name [bmp format]." << std::endl;
      exit(1);
    }

    // load input
    if(argc>1)
    {
      output_dir = std::string(argv[2]);
      std::cout << "Output directory: " << output_dir << std::endl;
    }
    else
    {
      std::cout << "Please specify output directory." << std::endl;
      exit(1);
    }

    // run test
    new boost::thread(test_mnist);

    new boost::thread(model);

    // start graphics
    startGraphics(argc,argv,"Simple ANN Classification - MNIST Digits");
    return 0;
}

