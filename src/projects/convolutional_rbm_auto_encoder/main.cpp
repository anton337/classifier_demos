#include <iostream>
#include "AutoEncoderConstructor.h"

int main()
{
  std::cout << "Auto Encoder Constructor" << std::endl;
  AutoEncoderConstructor < double > constructor;
  int nx=32;
  int ny=32;
  std::vector<LayerType> layer_type;
  layer_type.push_back (   CONVOLUTIONAL_LAYER );
  layer_type.push_back (     MAX_POOLING_LAYER );
  layer_type.push_back (   CONVOLUTIONAL_LAYER );
  layer_type.push_back (     MAX_POOLING_LAYER );
  layer_type.push_back (   CONVOLUTIONAL_LAYER );
  layer_type.push_back (        IDENTITY_LAYER );
  layer_type.push_back ( DECONVOLUTIONAL_LAYER );
  layer_type.push_back (   MAX_UNPOOLING_LAYER );
  layer_type.push_back ( DECONVOLUTIONAL_LAYER );
  layer_type.push_back (   MAX_UNPOOLING_LAYER );
  layer_type.push_back ( DECONVOLUTIONAL_LAYER );
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
  long tmp;
  std::vector<long> layer_nx;
  tmp = nx;                         layer_nx.push_back(tmp);
  tmp -= 2*(layer_kx[0]/2);         layer_nx.push_back(tmp);
  tmp /= layer_pooling_factorx[1];  layer_nx.push_back(tmp);
  tmp -= 2*(layer_kx[2]/2);         layer_nx.push_back(tmp);
  tmp /= layer_pooling_factorx[3];  layer_nx.push_back(tmp);
  tmp -= 2*(layer_kx[2]/2);         layer_nx.push_back(tmp);
                                    layer_nx.push_back(tmp);
                                    layer_nx.push_back(tmp);
  tmp += 2*(layer_kx[0]/2);         layer_nx.push_back(tmp);
  tmp *= layer_pooling_factorx[1];  layer_nx.push_back(tmp);
  tmp += 2*(layer_kx[2]/2);         layer_nx.push_back(tmp);
  tmp *= layer_pooling_factorx[3];  layer_nx.push_back(tmp);
  tmp += 2*(layer_kx[2]/2);         layer_nx.push_back(tmp);
  std::vector<long> layer_ny;
  tmp = ny;                         layer_ny.push_back(tmp);
  tmp -= 2*(layer_ky[0]/2);         layer_ny.push_back(tmp);
  tmp /= layer_pooling_factory[1];  layer_ny.push_back(tmp);
  tmp -= 2*(layer_ky[2]/2);         layer_ny.push_back(tmp);
  tmp /= layer_pooling_factory[3];  layer_ny.push_back(tmp);
  tmp -= 2*(layer_ky[2]/2);         layer_ny.push_back(tmp);
                                    layer_ny.push_back(tmp);
                                    layer_ny.push_back(tmp);
  tmp += 2*(layer_ky[0]/2);         layer_ny.push_back(tmp);
  tmp *= layer_pooling_factory[1];  layer_ny.push_back(tmp);
  tmp += 2*(layer_ky[2]/2);         layer_ny.push_back(tmp);
  tmp *= layer_pooling_factory[3];  layer_ny.push_back(tmp);
  tmp += 2*(layer_ky[2]/2);         layer_ny.push_back(tmp);
  std::vector < long > nodes;
  nodes.push_back ( features[0 ]*layer_nx[0 ]*layer_ny[0 ] );
  nodes.push_back ( features[1 ]*layer_nx[1 ]*layer_ny[1 ] );
  nodes.push_back ( features[2 ]*layer_nx[2 ]*layer_ny[2 ] );
  nodes.push_back ( features[3 ]*layer_nx[3 ]*layer_ny[3 ] );
  nodes.push_back ( features[4 ]*layer_nx[4 ]*layer_ny[4 ] );
  nodes.push_back ( features[5 ]*layer_nx[5 ]*layer_ny[5 ] );
  nodes.push_back ( features[6 ]*layer_nx[6 ]*layer_ny[6 ] );
  nodes.push_back ( features[7 ]*layer_nx[7 ]*layer_ny[7 ] );
  nodes.push_back ( features[8 ]*layer_nx[8 ]*layer_ny[8 ] );
  nodes.push_back ( features[9 ]*layer_nx[9 ]*layer_ny[9 ] );
  nodes.push_back ( features[10]*layer_nx[10]*layer_ny[10] );
  nodes.push_back ( features[11]*layer_nx[11]*layer_ny[11] );
  nodes.push_back ( features[12]*layer_nx[12]*layer_ny[12] );
  for(long i=0;i<layer_nx.size();i++)
  {
    std::cout << i << "\tsize=" << layer_nx[i] << '\t' << layer_ny[i] << '\t' << features[i] << '\t' << nodes[i] << std::endl;
  }
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


