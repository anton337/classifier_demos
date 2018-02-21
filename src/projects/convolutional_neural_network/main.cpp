#include "ConvolutionalNN.h"
#include <iostream>
#include <vector>
#include "binaryReader.h"
#include "visualization.h"

std::string output_dir = "";

char * label_dat = NULL;

VisualizeDataArray < double > * viz_in_dat = NULL;

VisualizeActivationProbe < double > * viz_probe = NULL;

std::vector<ConvolutionalNeuralNetwork <double> * > models;

ConvolutionalNeuralNetwork <double> * model = NULL;

long J = 0;

// learns to classify hand written digits
void test_cnn()
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
    models[J] -> train(0,0.1,10000,N,viz_in_dat->n_x*viz_in_dat->n_y,1,viz_in_dat->viz_dat,out_dat);
    //std::stringstream ss;
    //ss << output_dir << "/mnist-" << (char)(65+J) << ".ann";
    //dump_to_file(models[J],ss.str());
    J = (J+1)%26;
    model = models[J];
    //viz_probe->probe_perceptron = models[J];
  }
}

int main(int argc,char ** argv)
{
    std::cout << "Starting Convolutional Neural Network Test ... " << std::endl;

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
      double * dat = reader.readBinary(16,nx,ny,argv[1],32*100);
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
      /* 0 */ nodes.push_back(nx*ny); // inputs
      /* 1 */ nodes.push_back(2304); // conv layer
      /* 2 */ nodes.push_back(576); // pool layer
      /* 3 */ nodes.push_back(1200); // conv layer
      /* 4 */ nodes.push_back(300); // pool layer
      /* 5 */ nodes.push_back(50/*16*/); // hidden layer
      /* 6 */ nodes.push_back(10/*16*/); // hidden layer
      /*   */ nodes.push_back(1); // output layer
      /*   */ nodes.push_back(1); // outputs
      std::vector<LayerType> layer_type;
      layer_type.push_back(FULLY_CONNECTED_LAYER);
      layer_type.push_back(CONVOLUTIONAL_LAYER);
      layer_type.push_back(POOLING_LAYER);
      layer_type.push_back(CONVOLUTIONAL_LAYER);
      layer_type.push_back(POOLING_LAYER);
      layer_type.push_back(FULLY_CONNECTED_LAYER);
      layer_type.push_back(FULLY_CONNECTED_LAYER);
      layer_type.push_back(FULLY_CONNECTED_LAYER);
      layer_type.push_back(FULLY_CONNECTED_LAYER);
      layer_type.push_back(FULLY_CONNECTED_LAYER);
      layer_type.push_back(FULLY_CONNECTED_LAYER);
      std::vector<ActivationType> activation_type;
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      activation_type.push_back(LOGISTIC);
      std::vector<long> features;
      features.push_back(1);
      features.push_back(4);
      features.push_back(4);
      features.push_back(12);
      features.push_back(12);
      features.push_back(1);
      features.push_back(1);
      features.push_back(1);
      std::vector<long> layer_kx;
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(3);
      layer_kx.push_back(3);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      layer_kx.push_back(5);
      std::vector<long> layer_ky;
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(3);
      layer_ky.push_back(3);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      layer_ky.push_back(5);
      std::vector<long> layer_nx;
      layer_nx.push_back(nx);
      layer_nx.push_back(24);
      layer_nx.push_back(12);
      layer_nx.push_back(10);
      layer_nx.push_back(5);
      layer_nx.push_back(nx-10);
      layer_nx.push_back(nx-12);
      layer_nx.push_back(nx-14);
      layer_nx.push_back(nx-16);
      layer_nx.push_back(nx-18);
      layer_nx.push_back(nx-20);
      layer_nx.push_back(nx-22);
      layer_nx.push_back(nx);
      layer_nx.push_back(nx);
      std::vector<long> layer_ny;
      layer_ny.push_back(ny);
      layer_ny.push_back(24);
      layer_ny.push_back(12);
      layer_ny.push_back(10);
      layer_ny.push_back(5);
      layer_ny.push_back(ny-10);
      layer_ny.push_back(ny-12);
      layer_ny.push_back(ny-14);
      layer_ny.push_back(ny-16);
      layer_ny.push_back(ny-18);
      layer_ny.push_back(ny-20);
      layer_ny.push_back(ny-22);
      layer_ny.push_back(ny);
      layer_ny.push_back(ny);
      std::vector<long> layer_pooling_factorx;
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      layer_pooling_factorx.push_back(2);
      std::vector<long> layer_pooling_factory;
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      layer_pooling_factory.push_back(2);
      //for(int i=0;i<26;i++)
      {
        models.push_back 
            ( 
                new ConvolutionalNeuralNetwork < double > 
                    ( nodes 
                    , layer_type
                    , activation_type
                    , features
                    , layer_kx
                    , layer_ky
                    , layer_nx
                    , layer_ny
                    , layer_pooling_factorx
                    , layer_pooling_factory
                    ) 
            );
        //std::stringstream ss;
        //ss << output_dir << "/mnist-" << (char)(65+i) << ".ann";
        //load_from_file ( perceptrons[i] , ss.str() );
      }
      model = models[0];
      // viz_probe = new VisualizeActivationProbe < double > ( perceptrons[0]
      //                                                     , new ActivationProbe<double> ( perceptrons[0]
      //                                                                                   , 0
      //                                                                                   )
      //                                                     , 28     , 4
      //                                                     , 28     , 4
      //                                                     //28x28  //16
      //                                                     , 0 , 1 , -1 , 1
      //                                                     );
      // addDisplay ( viz_probe  );

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
    new boost::thread(test_cnn);

    startGraphics(argc,argv,"Convolutional Neural Network");
    std::cout << "Finished." << std::endl;
    return 0;
}

