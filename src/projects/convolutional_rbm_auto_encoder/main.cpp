#include <iostream>
#include <stdlib.h>
#include "readBMP.h"
#include "AutoEncoderConstructor.h"
#include "visualization.h"

long nsamp = 0;

std::vector<long> layer_nx;
std::vector<long> layer_ny;
std::vector<long> features;
std::vector<long> layer_pooling_factory;
std::vector<long> layer_pooling_factorx;

std::vector < double * > rbm_data;

std::vector < ConvolutionalRBM < double > * > rbms;

std::vector<double> epsilon;

std::vector<long> niters;

std::vector<long> nsteps;

long STAGE = 0;

void train()
{
  while(true)
  {
    CONTINUE=true;
    if(STAGE==0)CONTINUE=false;
    for(long i=0;i<niters[STAGE]&&CONTINUE;i++)
    {
      usleep(10000);
      rbms[STAGE]->init(0);
      rbms[STAGE]->cd(nsteps[STAGE],epsilon[STAGE],0);
      std::cout << STAGE << '\t' << i << '\t' << rbms[STAGE]->final_error << std::endl;
    }
    rbms[STAGE]->cd(1,0,0);
    if(STAGE+1<niters.size())
    {
        long nx = layer_nx[2*STAGE+1];
        long ny = layer_ny[2*STAGE+1];
        long K  = features[2*STAGE+1];
        long dx = layer_nx[2*STAGE+2];
        long dy = layer_ny[2*STAGE+2];
        long fx = layer_pooling_factorx[2*STAGE+1];
        long fy = layer_pooling_factory[2*STAGE+1];
        std::cout << STAGE << '\t' << K << '\t' << nx << '\t' << ny << '\t' << dx << '\t' << dy << '\t' << fx << '\t' << fy << std::endl;
        for(long n=0,k=0,size=(long)nsamp;n<size;n++)
        {
            double max_value;
            for(long l=0;l<K;l++)
            {
                for(long y=0;y<dy;y++)
                {
                    for(long x=0;x<dx;x++,k++)
                    {
                        max_value = -1000000;
                        for(long _y=0;_y<fy;_y++)
                        {
                            for(long _x=0;_x<fx;_x++)
                            {
                                max_value = (max_value>rbms[STAGE]->hid[n*K*nx*ny+l*nx*ny+(fy*y+_y)*nx+(fx*x+_x)])
                                          ?  max_value:rbms[STAGE]->hid[n*K*nx*ny+l*nx*ny+(fy*y+_y)*nx+(fx*x+_x)];
                            }
                        }
                        max_value = (max_value> 1)? 1:max_value;
                        max_value = (max_value< 0)? 0:max_value;
                        rbm_data[STAGE+1][k] = max_value;
                    }
                }
            }
        }
    }
    STAGE = (STAGE+1)%niters.size();
  }
}

int main(int argc,char ** argv)
{
  srand(time(0));
  if(argc>0)
  {
    int nx=32;
    int ny=32;
    Image * img = new Image();
    ImageLoad(argv[1],img);
    double * dat_full = img->get_doubles(nx,ny);
    nsamp = img->get_size()/(nx*ny);
    double * dat = new double[nsamp*nx*ny]; 
    double * out = new double[nsamp];
    for(long i=0,k=0;i<nsamp;i++)
    {
        long ind_1 = i;
        for(long x=0;x<nx;x++)
        {
            for(long y=0;y<ny;y++,k++)
            {
                dat[k] = dat_full[(ind_1)*nx*ny+x*ny+y];
            }
        }
        out[i] = i%2==0;
    }

    std::cout << "Auto Encoder Constructor" << std::endl;
    std::map<int,std::string> str;
    str[1] = "full";
    str[2] = "ident";
    str[3] = "relu";
    str[4] = "pool";
    str[5] = "pool";
    str[6] = "conv";
    str[7] = "deconv";
    str[8] = "unpool";
    str[9] = "unpool";
    AutoEncoderConstructor < double > constructor;
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
    features.push_back(    1);

    features.push_back(20+1);
    features.push_back(20+1);

    features.push_back(20+1);
    features.push_back(20+1);

    features.push_back(20+1);

    features.push_back(20+1);
    features.push_back(20+1);

    features.push_back(20+1);
    features.push_back(20+1);

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
    tmp = nx;                         layer_nx.push_back(tmp);
    tmp -= 2*(layer_kx[0]/2);         layer_nx.push_back(tmp);
    tmp /= layer_pooling_factorx[1];  layer_nx.push_back(tmp);
    tmp -= 2*(layer_kx[2]/2);         layer_nx.push_back(tmp);
    tmp /= layer_pooling_factorx[3];  layer_nx.push_back(tmp);
    tmp -= 2*(layer_kx[2]/2);         layer_nx.push_back(tmp);
    tmp += 2*(layer_kx[0]/2);         layer_nx.push_back(tmp);
    tmp *= layer_pooling_factorx[1];  layer_nx.push_back(tmp);
    tmp += 2*(layer_kx[2]/2);         layer_nx.push_back(tmp);
    tmp *= layer_pooling_factorx[3];  layer_nx.push_back(tmp);
    tmp += 2*(layer_kx[2]/2);         layer_nx.push_back(tmp);
    tmp = ny;                         layer_ny.push_back(tmp);
    tmp -= 2*(layer_ky[0]/2);         layer_ny.push_back(tmp);
    tmp /= layer_pooling_factory[1];  layer_ny.push_back(tmp);
    tmp -= 2*(layer_ky[2]/2);         layer_ny.push_back(tmp);
    tmp /= layer_pooling_factory[3];  layer_ny.push_back(tmp);
    tmp -= 2*(layer_ky[2]/2);         layer_ny.push_back(tmp);
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
    for(long i=0;i<layer_nx.size();i++)
    {
      std::cout << i << "\t" << str[layer_type[i]] << "\tsize=" << layer_nx[i] << '\t' << layer_ny[i] << '\t' << features[i] << '\t' << nodes[i] << std::endl;
    }
    /*
    0     CONVOLUTIONAL_LAYER  size=32 32  1   1024
    1       MAX_POOLING_LAYER  size=28 28  81  63504
    2     CONVOLUTIONAL_LAYER  size=14 14  81  15876
    3       MAX_POOLING_LAYER  size=10 10  161 16100
    4     CONVOLUTIONAL_LAYER  size=5  5   161 4025
    5          IDENTITY_LAYER  size=1  1   321 321
    6   DECONVOLUTIONAL_LAYER  size=5  5   161 4025
    7     MAX_UNPOOLING_LAYER  size=10 10  161 16100
    8   DECONVOLUTIONAL_LAYER  size=14 14  81  15876
    9     MAX_UNPOOLING_LAYER  size=28 28  81  63504
    10  DECONVOLUTIONAL_LAYER  size=32 32  1   1024
    */
    std::vector<bool> locked;
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
    locked.push_back(true);
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


    long N = 3;

    for(long i=0;i<N;i++)
    {
        rbm_data . push_back ( new double [ (long)nsamp 
                                         * features[2*i] 
                                         * layer_nx[2*i+0] 
                                         * layer_ny[2*i+0] 
                                         ] 
                            );
    }

    double min_val =  100000000;
    double max_val = -100000000;
    for(long k=0,size=(long)nsamp*features[0]*layer_nx[0]*layer_ny[0];k<size;k++)
    {
        rbm_data[0][k] = dat[k];
        min_val = (dat[k]<min_val)?dat[k]:min_val;
        max_val = (dat[k]>max_val)?dat[k]:max_val;
    }

    for(long k=0,size=(long)nsamp*features[0]*layer_nx[0]*layer_ny[0];k<size;k++)
    {
        rbm_data[0][k] = (dat[k]-min_val)/(max_val-min_val);
    }

    epsilon.push_back(1e-2);
    epsilon.push_back(1e-4);
    epsilon.push_back(1e-6);

    niters.push_back(1000000);
    niters.push_back(1000000);
    niters.push_back(1000000);

    nsteps.push_back(1);
    nsteps.push_back(1);
    nsteps.push_back(1);

    for(long i=0;i<N;i++)
    {
        ConvolutionalRBM < double > * rbm = 
          new ConvolutionalRBM < double > 
          (
            features[2*i+0] * layer_nx[2*i+0] * layer_ny[2*i+0]
          , features[2*i+1] * layer_nx[2*i+1] * layer_ny[2*i+1]
          , layer_nx[2*i+0]
          , layer_ny[2*i+0]
          , layer_nx[2*i+1]
          , layer_ny[2*i+1]
          , layer_kx[2*i+0]
          , layer_ky[2*i+0]
          , features[2*i+0]
          , features[2*i+1]
          , nsamp
          , rbm_data[i]
          , i==0
          , 5
          );

        for(long iter=0;iter<1;iter++)
        {
            rbm -> init(0);
            rbm -> cd(1,0,0);
            std::cout << iter << '\t' << rbm -> final_error << std::endl;
        }

        rbms . push_back ( rbm );

        if(i+1<N)
        {
            long nx = layer_nx[2*i+1];
            long ny = layer_ny[2*i+1];
            long K  = features[2*i+1];
            long dx = layer_nx[2*i+2];
            long dy = layer_ny[2*i+2];
            long fx = layer_pooling_factorx[2*i+1];
            long fy = layer_pooling_factory[2*i+1];
            std::cout << i << '\t' << K << '\t' << nx << '\t' << ny << '\t' << dx << '\t' << dy << '\t' << fx << '\t' << fy << std::endl;
            for(long n=0,k=0,size=(long)nsamp;n<size;n++)
            {
                double max_value;
                for(long l=0;l<K;l++)
                {
                    for(long y=0;y<dy;y++)
                    {
                        for(long x=0;x<dx;x++,k++)
                        {
                            max_value = -1000000;
                            for(long _y=0;_y<fy;_y++)
                            {
                                for(long _x=0;_x<fx;_x++)
                                {
                                    max_value = (max_value>rbm->hid[n*K*nx*ny+l*nx*ny+(fy*y+_y)*nx+(fx*x+_x)])
                                              ?  max_value:rbm->hid[n*K*nx*ny+l*nx*ny+(fy*y+_y)*nx+(fx*x+_x)];
                                }
                            }
                            rbm_data[i+1][k] = max_value;
                        }
                    }
                }
            }
        }
    }

   
    for(long i=0;i<N;i++)
    {
        VisualizeCRBMVisibleProbe < double > * viz_crbm_visible = NULL;
        viz_crbm_visible = new VisualizeCRBMVisibleProbe < double > ( rbms[i]
                                                                    , new CRBMConvolutionProbe < double > ( rbms[i] )
                                                                    , -1+0.4*i , -.9+0.4*i
                                                                    , -1 , 1
                                                                    );
        addDisplay ( viz_crbm_visible );

        VisualizeCRBMKernelProbe < double > * viz_crbm_kernel = NULL;
        viz_crbm_kernel = new VisualizeCRBMKernelProbe < double > ( rbms[i]
                                                                  , new CRBMConvolutionProbe < double > ( rbms[i] )
                                                                  , -.9+0.4*i , -.7+0.4*i
                                                                  ,  -1 , 1
                                                                  );
        addDisplay ( viz_crbm_kernel );

        VisualizeCRBMHiddenProbe < double > * viz_crbm_hidden = NULL;
        viz_crbm_hidden = new VisualizeCRBMHiddenProbe < double > ( rbms[i]
                                                                  , new CRBMConvolutionProbe < double > ( rbms[i] )
                                                                  ,  -.7+0.4*i , -.6+0.4*i
                                                                  ,  -1 , 1
                                                                  );
        addDisplay ( viz_crbm_hidden );
    }


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

    //model -> train(0,.1,1000000000000000,nsamp,nx*ny,1,dat,out);

  }

  new boost::thread(train);

  startGraphics(argc,argv,"CRBM pretrain CNN");

  std::cout << "Done." << std::endl;

  return 0;
}


