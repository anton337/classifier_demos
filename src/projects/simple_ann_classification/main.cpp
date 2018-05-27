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

Perceptron<double> * merged = NULL;

std::string output_dir = "";

//long n_out = 1;
long n_out = 10;

// models stuff periodically
void model()
{
    while(true)
    {
        double * out = perceptron->model(viz_in_dat->n_vars,n_out,&viz_in_dat->viz_dat[viz_in_dat->viz_selection*viz_in_dat->n_vars]);
        int max_i=0;
        double max_val=0;
        for(int i=0;i<10;i++)
        {
          if(out[i]>max_val)
          {
            max_i=i;
            max_val=out[i];
          }
        }
        std::cout << "ans:" << max_i << std::endl;
        delete [] out;
        usleep(1000000);
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
          out_dat[n_out*i+j] = (i/(100*5)==J)?1.0:1.0e-5;
          if(out_dat[n_out*i+j]<0.1)if(rand()%10<5)out_dat[n_out*i+j]=0;
        }
      }
    }
    perceptrons[J] -> train(0,1e-1,100000,N,viz_in_dat->n_x*viz_in_dat->n_y,n_out,viz_in_dat->viz_dat,out_dat,0,false,NULL,NULL,NULL,true,1000);
    //perceptron -> train(0,1e-1,10000,N,viz_in_dat->n_x*viz_in_dat->n_y,n_out,viz_in_dat->viz_dat,out_dat);
    std::stringstream ss;
    ss << output_dir << "/mnist-" << J << ".ann";
    //ss << output_dir << "/mnist.ann";
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
      nodes.push_back(1); // output layer
      nodes.push_back(1); // outputs
      for(int i=0;i<10;i++)
      {
        perceptrons.push_back(new Perceptron<double>(nodes,1e-1));
        std::stringstream ss;
        ss << output_dir << "/mnist-" << i << ".ann";
        //load_from_file ( perceptrons[i] , ss.str() );
      }

      MergePerceptrons<double> merge_perceptrons;
      merged = merge_perceptrons . merge ( perceptrons );

      perceptron = perceptrons[0];
      //perceptron = merged;

      double * D = dat->get_doubles(nx,ny);
      viz_in_dat = new VisualizeDataArray < double > ( dat->get_size()
                                                     , nx*ny
                                                     , dat->get_width()
                                                     , nx
                                                     , ny
                                                     , D
                                                     , -1 , 0 , -1 , 1
                                                     );
      viz_probe = new VisualizeActivationProbe < double > ( perceptrons[0] // merged
                                                          , new ActivationProbe<double> ( perceptrons[0] // merged
                                                                                        , 0
                                                                                        )
                                                          , 20     , 4 * 2
                                                          , 20     , 2 // 4 * 5
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

    // run test
    new boost::thread(test_mnist);

    //new boost::thread(model);

    // start graphics
    startGraphics(argc,argv,"Simple ANN Classification - MNIST Digits");
    return 0;
}

