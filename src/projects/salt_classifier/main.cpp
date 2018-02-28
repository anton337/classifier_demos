#include <iostream>
#include <boost/thread.hpp>
#include "visualization.h"
#include "readBMP.h"
#include "binaryReader.h"
#include "Perceptron.h"
#include "ActivationProbe.h"
#include "snapshot.h"
#include "MergePerceptrons.h"

VisualizeDataArray < double > * viz_in_dat = NULL;
VisualizeDataArray < double > * viz_live_dat = NULL;
VisualizeDataArray < double > * viz_salt_dat = NULL;

ConvolutionalNeuralNetwork <double> * model = NULL;

long nx = 32;
long ny = 32;

double * D = NULL;

double * out = NULL;

long nsamp = 0;

void train_cnn()
{
      model -> train(0,.1,10000,nsamp,nx*ny,1,D,out);
}

int main(int argc,char ** argv)
{
    std::cout << "Salt Classification" << std::endl;

    // load data
    if(argc>1)
    {
        Image * dat = new Image();
        ImageLoad(argv[1],dat);
        Image * mask = new Image();
        ImageLoad(argv[2],mask);
        long dx = 1;
        long dy = 1;
        D = dat->get_doubles(nx,ny,dx,dy);
        viz_in_dat = new VisualizeDataArray < double > ( dat->get_size()
                                                       , nx*ny
                                                       , nx*ny
                                                       , nx
                                                       , ny
                                                       , D
                                                       , -1 , -0.5 , -1 , 1
                                                       );
        addDisplay ( viz_in_dat );
        double * L = mask->get_doubles(nx,ny,dx,dy,0,false);
        viz_live_dat = new VisualizeDataArray < double > ( mask->get_size()
                                                         , nx*ny
                                                         , nx*ny
                                                         , nx
                                                         , ny
                                                         , L
                                                         , 0 , 0.5 , -1 , 1
                                                         );
        //addDisplay ( viz_live_dat );
        double * S = mask->get_doubles(nx,ny,dx,dy,1,false);
        viz_salt_dat = new VisualizeDataArray < double > ( mask->get_size()
                                                         , nx*ny
                                                         , nx*ny
                                                         , nx
                                                         , ny
                                                         , S
                                                         , -0.5 , 0.0 , -1 , 1
                                                         );
        addDisplay ( viz_salt_dat );
        out = new double[dat->get_size()/(nx*ny)];
        nsamp = dat->get_size()/(nx*ny);
        long n_pos = 0;
        long n_neg = 0;
        for(long i=0,size=dat->get_size()/(nx*ny);i<size;i++)
        {
          {
            double max_L = 0;
            for(long k=0;k<nx*ny;k++)
            {
                max_L = (max_L>L[i*nx*ny+k])?max_L:L[i*nx*ny+k];
            }
            double max_S = 0;
            double min_S = 1e10;
            for(long k=0;k<nx*ny;k++)
            {
                max_S = (max_S>S[i*nx*ny+k])?max_S:S[i*nx*ny+k];
                min_S = (min_S<S[i*nx*ny+k])?min_S:S[i*nx*ny+k];
            }
            if(max_L>0.5)
            {
                out[i]=0;
                continue;
            }
            if(max_S>0.5&&min_S>0.5)
            {
                if(n_neg<10)
                  out[i]=1e-5;
                else 
                  out[i]=0;
                n_neg++;
                continue;
            }
            if(max_S<0.5&&min_S<0.5)
            {
                if(n_pos<10)
                  out[i]=1;
                else 
                  out[i]=0;
                n_pos++;
                continue;
            }
            out[i]=0;
          }
        }
        std::cout << n_neg << '\t' << n_pos << '\t' << (double)n_neg/(n_pos+n_neg) << std::endl;

        std::vector<long> nodes;
        /* 0 */ nodes.push_back(nx*ny);
        /* 1 */ nodes.push_back(14112);
        /* 2 */ nodes.push_back(3528); 
        /* 3 */ nodes.push_back(4096);
        /* 4 */ nodes.push_back(1024);
        /* 5 */ nodes.push_back(1024);
        /* 7 */ nodes.push_back(256);   
        /* 8 */ nodes.push_back(64); 
        /*   */ nodes.push_back(1);    
        /*   */ nodes.push_back(1);    
        std::vector<LayerType> layer_type;          
        layer_type.push_back(CONVOLUTIONAL_LAYER);  //0
        layer_type.push_back(MAX_POOLING_LAYER);    //1
        layer_type.push_back(CONVOLUTIONAL_LAYER);  //2
        layer_type.push_back(MAX_POOLING_LAYER);    //3
        layer_type.push_back(RELU_LAYER);           //4
        layer_type.push_back(FULLY_CONNECTED_LAYER);//5
        layer_type.push_back(FULLY_CONNECTED_LAYER);//6
        layer_type.push_back(FULLY_CONNECTED_LAYER);//7
        layer_type.push_back(FULLY_CONNECTED_LAYER);//8
        layer_type.push_back(FULLY_CONNECTED_LAYER);//9
        layer_type.push_back(FULLY_CONNECTED_LAYER);//10
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
        features.push_back(18);
        features.push_back(18);
        features.push_back(64);
        features.push_back(64);
        features.push_back(64);
        features.push_back(1);
        features.push_back(1);
        features.push_back(1);
        features.push_back(1);
        std::vector<long> layer_kx;
        layer_kx.push_back(5);
        layer_kx.push_back(5);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        layer_kx.push_back(7);
        std::vector<long> layer_ky;
        layer_ky.push_back(5);
        layer_ky.push_back(5);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        layer_ky.push_back(7);
        std::vector<long> layer_nx;
        layer_nx.push_back(nx);
        layer_nx.push_back(28);
        layer_nx.push_back(14);
        layer_nx.push_back(8);
        layer_nx.push_back(4);
        layer_nx.push_back(4);
        layer_nx.push_back(3);
        layer_nx.push_back(nx-14);
        layer_nx.push_back(nx-16);
        layer_nx.push_back(nx-18);
        layer_nx.push_back(nx-20);
        layer_nx.push_back(nx-22);
        layer_nx.push_back(nx);
        layer_nx.push_back(nx);
        std::vector<long> layer_ny;
        layer_ny.push_back(ny);
        layer_ny.push_back(28);
        layer_ny.push_back(14);
        layer_ny.push_back(8);
        layer_ny.push_back(4);
        layer_ny.push_back(4);
        layer_ny.push_back(3);
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
        model = new ConvolutionalNeuralNetwork < double > 
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
                    ); 

        VisualizeCNNActivationProbe < double > * viz_cnn_activation0 = NULL;
        viz_cnn_activation0 = new VisualizeCNNActivationProbe < double > ( model
                                                                         , new CNNActivationProbe < double > ( model , 0 )
                                                                         , 0.0 , 0.25
                                                                         ,-1 , 1
                                                                         );
        addDisplay ( viz_cnn_activation0 );

        VisualizeCNNActivationProbe < double > * viz_cnn_activation2 = NULL;
        viz_cnn_activation2 = new VisualizeCNNActivationProbe < double > ( model
                                                                         , new CNNActivationProbe < double > ( model , 2 )
                                                                         , 0.25 , 0.5
                                                                         ,-1 , 1
                                                                         );
        addDisplay ( viz_cnn_activation2 );

        VisualizeCNNActivationProbe < double > * viz_cnn_activation5 = NULL;
        viz_cnn_activation5 = new VisualizeCNNActivationProbe < double > ( model
                                                                         , new CNNActivationProbe < double > ( model , 5 )
                                                                         , 32 // in_nx
                                                                         , 16 // out_nx
                                                                         , 32 // in_ny
                                                                         , 16 // out_ny
                                                                         , 0.5 , 0.75
                                                                         ,-1 , 1
                                                                         );
        addDisplay ( viz_cnn_activation5 );

        VisualizeCNNActivationProbe < double > * viz_cnn_activation6 = NULL;
        viz_cnn_activation6 = new VisualizeCNNActivationProbe < double > ( model
                                                                         , new CNNActivationProbe < double > ( model , 6 )
                                                                         , 16 // in_nx
                                                                         , 8  // out_nx
                                                                         , 16 // in_ny
                                                                         , 8  // out_ny
                                                                         , 0.75 , 1
                                                                         ,-1 , 1
                                                                         );
        addDisplay ( viz_cnn_activation6 );
    }

    new boost::thread(train_cnn);

    // start graphics
    startGraphics(argc,argv,"Salt Segmentation Example");
    return 0;
}

