#include <iostream>
#include "AutoEncoderConstructor.h"

int main()
{
  std::cout << "Auto Encoder Constructor" << std::endl;
  AutoEncoderConstructor < double > constructor;
  int nx=32;
  int ny=32;
  std::vector < long > nodes;
  nodes.push_back ( nx*ny );
  nodes.push_back (  5000 );
  nodes.push_back (  4000 );
  nodes.push_back (  3000 );
  nodes.push_back (  2000 );
  nodes.push_back (  1000 );
  nodes.push_back (   500 );
  nodes.push_back (  1000 );
  nodes.push_back (  2000 );
  nodes.push_back (  3000 );
  nodes.push_back (  4000 );
  nodes.push_back (  5000 );
  nodes.push_back ( nx*ny );
  std::vector<LayerType> layer_type;
  layer_type.push_back ( CONVOLUTIONAL_LAYER );
  layer_type.push_back (   MAX_POOLING_LAYER );
  layer_type.push_back ( CONVOLUTIONAL_LAYER );
  layer_type.push_back (   MAX_POOLING_LAYER );
  layer_type.push_back ( CONVOLUTIONAL_LAYER );
  layer_type.push_back (          RELU_LAYER );
  layer_type.push_back ( CONVOLUTIONAL_LAYER );
  layer_type.push_back (   MAX_POOLING_LAYER );
  layer_type.push_back ( CONVOLUTIONAL_LAYER );
  layer_type.push_back (   MAX_POOLING_LAYER );
  layer_type.push_back ( CONVOLUTIONAL_LAYER );
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
  std::vector<bool> locked;
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  locked.push_back(false);
  std::vector<long> features;
  features.push_back(    1);
  features.push_back( 80+1);
  features.push_back( 80+1);
  features.push_back(160+1);
  features.push_back(160+1);
  features.push_back(320+1);
  features.push_back(320+1);
  features.push_back(320+1);
  features.push_back(160+1);
  features.push_back(160+1);
  features.push_back( 80+1);
  features.push_back( 80+1);
  features.push_back(    1);
  std::vector<long> layer_kx;
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
  layer_kx.push_back(5);
  layer_kx.push_back(5);
  layer_kx.push_back(5);
  layer_kx.push_back(5);
  std::vector<long> layer_ky;
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
  layer_ky.push_back(5);
  layer_ky.push_back(5);
  layer_ky.push_back(5);
  layer_ky.push_back(5);
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
  std::vector<long> layer_nx;
  layer_nx.push_back(nx);
  layer_nx.push_back(28);
  layer_nx.push_back(14);
  layer_nx.push_back(10);
  layer_nx.push_back(5);
  layer_nx.push_back(1);
  layer_nx.push_back(1);
  layer_nx.push_back(1);
  layer_nx.push_back(5);
  layer_nx.push_back(10);
  layer_nx.push_back(14);
  layer_nx.push_back(28);
  layer_nx.push_back(nx);
  std::vector<long> layer_ny;
  layer_ny.push_back(ny);
  layer_ny.push_back(28);
  layer_ny.push_back(14);
  layer_ny.push_back(10);
  layer_ny.push_back(5);
  layer_ny.push_back(1);
  layer_ny.push_back(1);
  layer_ny.push_back(1);
  layer_ny.push_back(5);
  layer_ny.push_back(10);
  layer_ny.push_back(14);
  layer_ny.push_back(28);
  layer_ny.push_back(ny);
  ConvolutionalNeuralNetwork < double > * model = NULL;
  model = new ConvolutionalNeuralNetwork < double > 
              ( nodes 
              , locked
              , layer_type
              , activation_type
              , features
              , layer_kx
              , layer_ky
              , layer_nx
              , layer_ny
              , layer_pooling_factorx
              , layer_pooling_factory
              ); 

  /*
  {
    long n = 1000;
    long K = 2;
    long M = 2;
    long kx = 5;
    long ky = 5;
    long nx = 500;
    long ny = 500;
    long dx = nx - 2*(kx/2);
    long dy = ny - 2*(ky/2);
    double * dat = new double[n*nx*ny];
    ConvolutionalRBM<double> * rbm 
      = new ConvolutionalRBM<double> ( 
                                     , K*dx*dy
                                     , nx
                                     , ny
                                     , dx
                                     , dy
                                     , kx
                                     , ky
                                     , M
                                     , K
                                     , n
                                     , dat
                                     );
  }
  */

  std::cout << "Done." << std::endl;

  return 0;
}


