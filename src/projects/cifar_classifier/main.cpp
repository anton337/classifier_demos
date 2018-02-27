#include "ConvolutionalNN.h"
#include <iostream>
#include "binaryReader.h"
#include "visualization.h"

ConvolutionalNeuralNetwork <double> * model = NULL;

VisualizeDataArrayColor < double > * viz_in_dat = NULL;

VisualizeDataArray < double > * viz_in_dat_1d = NULL;

long nx = 32;
long ny = 32;
long nsamp = 10000;
long tmp_samp = 5000;

double * in = new double[3*nx*ny*nsamp];
double * in_1d = new double[nx*ny*nsamp];
double * out = new double[nsamp];
double * dat = NULL;

void train_cnn()
{
      model -> train(0,.1,10000,nsamp,nx*ny,1,in_1d,out);
}

int main(int argc,char ** argv)
{
    std::cout << "Starting Cifar Clasifier Convolutional Neural Network Test ... " << std::endl;

    srand(time(0));

    // load input
    if(argc>0)
    {
        binaryReader<double> reader;
        dat = reader.readBinary(0,3*nx*ny+1,argv[1],nsamp);
        nsamp = reader.get_size()/(3*nx*ny+1);
        for(long i=0,t=0,d=0;i<nsamp;i++)
        {
            for(long c=0;c<3;c++)
            for(long x=0;x<nx;x++)
            for(long y=0;y<ny;y++,t++)
            {
                in[t] = dat[c*nx*ny+y+ny*(nx-1-x)+i*(3*nx*ny+1)+1];
            }
            long d_init = d;
            for(long x=0;x<nx;x++)
            for(long y=0;y<ny;y++,d++)
            {
                in_1d[d]  = dat[0*nx*ny+y+ny*(nx-1-x)+i*(3*nx*ny+1)+1];
                in_1d[d] += dat[1*nx*ny+y+ny*(nx-1-x)+i*(3*nx*ny+1)+1];
                in_1d[d] += dat[2*nx*ny+y+ny*(nx-1-x)+i*(3*nx*ny+1)+1];
                in_1d[d] /= 3.0;
            }
            out[i] = 256*dat[i*(3*nx*ny+1)];
            if((int)round(out[i]) == 5)
            {
                out[i] = 1;
            }
            else 
            if((int)round(out[i]) == 6)
            {
                out[i] = 1e-5;
            }
            else
            {
                out[i] = 0;
            }
            //d = d_init;
            //for(long x=0;x<nx;x++)
            //for(long y=0;y<ny;y++,d++)
            //{
            //  switch((int)(256*dat[i*(3*nx*ny+1)]) % 4)
            //  {
            //    case 0:
            //      in_1d[d] = (x==nx/2)?1:0;
            //      break;
            //    case 1:
            //      in_1d[d] = (y==ny/2)?1:0;
            //      break;
            //    case 2:
            //      in_1d[d] = (x+y==2*ny/2)?1:0;
            //      break;
            //    case 3:
            //      in_1d[d] = (x-y==0)?1:0;
            //      break;
            //  }
            //}
            //if(i<tmp_samp)
            //{
            //    std::cout << out[i] << std::endl;
            //}
        }
        viz_in_dat = new VisualizeDataArrayColor < double > ( nsamp*3*nx*ny
                                                            , 3*nx*ny
                                                            , 3*nx*ny
                                                            , nx
                                                            , ny
                                                            , in
                                                            , -1 , -0.5 , -1 , 1
                                                            );
        addDisplay ( viz_in_dat );
        viz_in_dat_1d = new VisualizeDataArray < double > ( nsamp*nx*ny
                                                          , nx*ny
                                                          , nx*ny
                                                          , nx
                                                          , ny
                                                          , in_1d
                                                          , -0.5 , 0 , -1 , 1
                                                          );
        addDisplay ( viz_in_dat_1d );
        std::vector<long> nodes;
        /* 0 */ nodes.push_back(nx*ny);
        /* 1 */ nodes.push_back(6272);
        /* 2 */ nodes.push_back(1568); 
        /* 3 */ nodes.push_back(1600);
        /* 4 */ nodes.push_back(400);
        /* 5 */ nodes.push_back(400);
        /* 7 */ nodes.push_back(100);   
        /* 8 */ nodes.push_back(25);   
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
        features.push_back(8);
        features.push_back(8);
        features.push_back(16);
        features.push_back(16);
        features.push_back(16);
        features.push_back(1);
        features.push_back(1);
        features.push_back(1);
        features.push_back(1);
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
        std::vector<long> layer_nx;
        layer_nx.push_back(nx);
        layer_nx.push_back(28);
        layer_nx.push_back(14);
        layer_nx.push_back(10);
        layer_nx.push_back(5);
        layer_nx.push_back(5);
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
        layer_ny.push_back(10);
        layer_ny.push_back(5);
        layer_ny.push_back(5);
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
                                                                         , 20 // in_nx
                                                                         , 10 // out_nx
                                                                         , 20 // in_ny
                                                                         , 10 // out_ny
                                                                         , 0.5 , 0.75
                                                                         ,-1 , 1
                                                                         );
        addDisplay ( viz_cnn_activation5 );

        VisualizeCNNActivationProbe < double > * viz_cnn_activation6 = NULL;
        viz_cnn_activation6 = new VisualizeCNNActivationProbe < double > ( model
                                                                         , new CNNActivationProbe < double > ( model , 6 )
                                                                         , 10 // in_nx
                                                                         , 5  // out_nx
                                                                         , 10 // in_ny
                                                                         , 5  // out_ny
                                                                         , 0.75 , 1
                                                                         ,-1 , 1
                                                                         );
        addDisplay ( viz_cnn_activation6 );
    }

    new boost::thread(train_cnn);

    startGraphics(argc,argv,"Convolutional Neural Network");
    std::cout << "Finished." << std::endl;
    return 0;
}

