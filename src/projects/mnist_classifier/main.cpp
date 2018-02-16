#include <iostream>
#include <boost/thread.hpp>
#include "visualization.h"
#include "readBMP.h"
#include "binaryReader.h"
#include "Perceptron.h"
#include "ActivationProbe.h"
#include "snapshot.h"
#include "MergePerceptrons.h"

std::string output_dir = "";

char * label_dat = NULL;

VisualizeDataArray < double > * viz_in_dat = NULL;

VisualizeActivationProbe < double > * viz_probe = NULL;

std::vector<Perceptron<double> * > perceptrons;

Perceptron<double> * perceptron = NULL;

int J = 0;

void model()
{
    while(true)
    {
      double * out = perceptron->model(viz_in_dat->n_vars,1,&viz_in_dat->viz_dat[viz_in_dat->viz_selection*viz_in_dat->n_vars]);
      int max_i=1;
      double max_val=0;
      for(int i=0;i<1;i++)
      {
        if(out[i]>max_val)
        {
          max_i=i;
          max_val=out[i];
        }
      }
      std::cout << (char)(65+J) << "\t" << (char)(64+label_dat[viz_in_dat->viz_selection]) << "\t" << "ans:" << max_i << "\t" << max_val << std::endl;
      delete [] out;
      usleep(1000000);
    }
}

// learns to classify hand written digits
void test_mnist()
{
  long N = viz_in_dat->n_elems/(viz_in_dat->n_x*viz_in_dat->n_y);
  double * out_dat = new double[N];
  while(true)
  {
    for(int i=0;i<N;i++)
    {
      {
        if(label_dat[i] == J+1)
        {
          out_dat[i] = 1;
        }
        else
        if(label_dat[i] < J+10 && label_dat[i] > J-10)
        {
          out_dat[i] = 1e-5;
        }
        else
        {
          out_dat[i] = 0;
        }
      }
    }
    perceptrons[J] -> train(0,1.0,10000,N,viz_in_dat->n_x*viz_in_dat->n_y,1,viz_in_dat->viz_dat,out_dat);
    std::stringstream ss;
    ss << output_dir << "/mnist-" << (char)(65+J) << ".ann";
    dump_to_file(perceptrons[J],ss.str());
    J = (J+1)%26;
    perceptron = perceptrons[J];
    viz_probe->probe_perceptron = perceptrons[J];
  }
}

int main(int argc,char ** argv)
{
    std::cout << "Running simple ann classification demo" << std::endl;

    srand(time(0));

    // snapshot directory
    if(argc>2)
    {
      output_dir = std::string(argv[3]);
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
      long nx = 28;
      long ny = 28;
      binaryReader<double> reader;
      double * dat = reader.readBinary(16,nx,ny,argv[1]);
      viz_in_dat = new VisualizeDataArray < double > ( reader.get_size()
                                                     , nx*ny
                                                     , nx*ny
                                                     , nx
                                                     , ny
                                                     , dat
                                                     , -1 , 0 , -1 , 1
                                                     );
      addDisplay ( viz_in_dat );
      
      std::vector<long> nodes;
      nodes.push_back(nx*ny); // inputs
      nodes.push_back(16); // hidden layer
      nodes.push_back(1); // output layer
      nodes.push_back(1); // outputs
      for(int i=0;i<26;i++)
      {
        perceptrons.push_back(new Perceptron<double>(nodes));
        std::stringstream ss;
        ss << output_dir << "/mnist-" << (char)(65+i) << ".ann";
        load_from_file ( perceptrons[i] , ss.str() );
      }
      perceptron = perceptrons[0];
      viz_probe = new VisualizeActivationProbe < double > ( perceptrons[0]
                                                          , new ActivationProbe<double> ( perceptrons[0]
                                                                                        , 0
                                                                                        )
                                                          , 28     , 4
                                                          , 28     , 4
                                                          //28x28  //16
                                                          , 0 , 1 , -1 , 1
                                                          );
      addDisplay ( viz_probe  );

    }
    else
    {
      std::cout << "Please specify input image data [binary format]." << std::endl;
      exit(1);
    }


    if(argc>1)
    {
      binaryReader<double> reader;
      label_dat = reader.readBinaryChars(8,argv[2]);
    }
    else
    {
      std::cout << "Please specify input label data [binary format]." << std::endl;
      exit(1);
    }


    // run test
    new boost::thread(test_mnist);

    new boost::thread(model);


    // start graphics
    startGraphics(argc,argv,"EMNIST Digits");
    return 0;
}

