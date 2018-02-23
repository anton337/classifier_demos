#include "ConvolutionalNN.h"
#include "sep_reader.h"
#include "visualization.h"

ConvolutionalNeuralNetwork <double> * model = NULL;

VisualizeDataArray < double > * viz_in_dat = NULL;

VisualizeDataArray < double > * viz_in_afi = NULL;

long wx = 32;
long wy = 32;
long nsamp = 100;
double * in = new double[nsamp*wx*wy];
double * in_afi = new double[nsamp*wx*wy];
double * out= new double[nsamp];

void train()
{
        model -> train(0,.2,10000,nsamp,wx*wy,1,in,out);
}

template<typename T>
T * sample_sep(T * dat,T * dat_afi,long z,long wx,long wy,long ox,long oy,long nx,long ny,long nz)
{
    // assumption, data is arranged by: dat[x*ny*nz+y*nz+z]
    T * ret = new T[wx*wy + 1]; // last sample contains classification
    T * afi = new T[wx*wy + 1]; // last sample contains classification
    if(ox<0||oy<0||ox+wx>=nx||oy+wy>=ny||z<0||z>=nz)
    {
        std::cout << "please make sure to sample within the volume." << std::endl;
        exit(1);
    }
    T min_val = 10000000000;
    T max_val =-10000000000;
    T min_afi = 10000000000;
    T max_afi =-10000000000;
    for(long x=ox,k=0;x<ox+wx;x++)
    {
        for(long y=oy;y<oy+wy;y++,k++)
        {
            ret[k] = dat    [x*ny*nz+y*nz+z];
            afi[k] = dat_afi[x*ny*nz+y*nz+z];
            if(ret[k]>max_val)max_val=ret[k];
            if(ret[k]<min_val)min_val=ret[k];
            if(afi[k]>max_afi)max_afi=afi[k];
            if(afi[k]<min_afi)min_afi=afi[k];
        }
    }
    for(long i=0;i<wx*wy;i++)
    {
      ret[i] = (ret[i]-min_val+0.0001)/(max_val-min_val+0.0001);
      afi[i] = (afi[i]-min_afi+0.0001)/(max_afi-min_afi+0.0001);
    }
    T num = 0;
    T den = 0;
    T tmp;
    long c = wx*(wy/2) + (wx/2);
    long w = 2;
    for(long x=-w;x<=w;x++)
    for(long y=-w;y<=w;y++)
    {
        tmp = ret[c+y+wy*x];
        tmp = ret[c+y];
        num += tmp;
        den += tmp*tmp;
    }
    num *= num;
    den *= 2*w+1;
    den *= 2*w+1;
    //T semb = num/den;
    {
      ret[wx*wy] = afi[c];
    }
    //ret[wx*wy] = semb;
    ret[c] = 0;
    delete [] afi;
    return ret;
}

int main(int argc,char ** argv)
{
    std::cout << "Starting SEP Classifier Convolutional Neural Network Test ... " << std::endl;

    srand(time(0));

    // load input
    if(argc>1)
    {
        SEPReader reader(argv[1]);
        int ox = reader.o3;
        int oy = reader.o2;
        int oz = reader.o1;
        int nx = reader.n3;
        int ny = reader.n2;
        int nz = reader.n1;
        float * dat = new float[nx*ny*nz];
        reader.read_sepval ( &dat[0]
                           , reader.o1
                           , reader.o2
                           , reader.o3
                           , reader.n1
                           , reader.n2
                           , reader.n3
                           );
        SEPReader reader_afi(argv[2]);
        float * dat_afi = new float[nx*ny*nz];
        reader_afi.read_sepval ( &dat_afi[0]
                           , reader.o1
                           , reader.o2
                           , reader.o3
                           , reader.n1
                           , reader.n2
                           , reader.n3
                           );
        long it = 0;
        long k = 0;
        long pos = 0;
        long neg = 0;
        double thresh = 0.05;
        double thresh2 = 0.95;
        while(true)
        {
            //float * tmp = sample_sep(dat,rand()%nz,wx,wy,rand()%(nx-wx),rand()%(ny-wy),nx,ny,nz);
            long z = 10+it%(nz-20);
            long x = 10+(it/nz)%(nx-wx-20);
            long y = 10+(it/(nz*nx))%(ny-wy-20);
            float * tmp     = sample_sep<float>(dat_afi,dat_afi,z,wx,wy,x,y,nx,ny,nz);
            float * tmp_afi = sample_sep<float>(dat    ,dat    ,z,wx,wy,x,y,nx,ny,nz);
            //std::cout << i << "\t" << tmp[wx*wy] << std::endl;
            if((tmp[wx*wy] < thresh && k%2==0) || (tmp[wx*wy] > thresh2 && k%2==1))
            {
                for(long i=0;i<wx*wy;i++)
                {
                    in[k*wx*wy+i] = tmp[i];
                    in_afi[k*wx*wy+i] = tmp_afi[i];
                }
                if(tmp[wx*wy] < thresh)
                {
                    neg++;
                    out[k] = 1e-5;
                    it = rand();
                }
                if(tmp[wx*wy] > thresh2)
                {
                    pos++;
                    out[k] = 1;
                }
                k++;
                std::cout << "k:" << k << std::endl;
            }
            it++;
            delete [] tmp;
            delete [] tmp_afi;
            if(k>=nsamp)break;
        }
        delete [] dat;
        delete [] dat_afi;
        std::cout << "pos:" << pos << std::endl;
        std::cout << "neg:" << neg << std::endl;


        viz_in_dat = new VisualizeDataArray < double > ( nsamp*wx*wy
                                                       , wx*wy
                                                       , wx*wy
                                                       , wx
                                                       , wy
                                                       , in
                                                       , -1 , 0 , -1 , 1
                                                       );
        addDisplay ( viz_in_dat );

        viz_in_afi = new VisualizeDataArray < double > ( nsamp*wx*wy
                                                       , wx*wy
                                                       , wx*wy
                                                       , wx
                                                       , wy
                                                       , in_afi
                                                       , 0 , 1 , -1 , 1
                                                       );
        addDisplay ( viz_in_afi );

        std::vector<long> nodes;
        /* 0 */ nodes.push_back(wx*wy);
        /* 1 */ nodes.push_back(6272);
        /* 2 */ nodes.push_back(1568); 
        /* 3 */ nodes.push_back(2400);
        /* 4 */ nodes.push_back(600);
        /* 5 */ nodes.push_back(600);
        /* 7 */ nodes.push_back(100);   
        /* 8 */ nodes.push_back(16);   
        /*   */ nodes.push_back(1);    
        /*   */ nodes.push_back(1);    
        std::vector<LayerType> layer_type;
        layer_type.push_back(CONVOLUTIONAL_LAYER);
        layer_type.push_back(MAX_POOLING_LAYER);
        layer_type.push_back(CONVOLUTIONAL_LAYER);
        layer_type.push_back(MAX_POOLING_LAYER);
        layer_type.push_back(RELU_LAYER);
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
        features.push_back(8);
        features.push_back(8);
        features.push_back(24);
        features.push_back(24);
        features.push_back(24);
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
        layer_nx.push_back(wx);
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
        layer_ny.push_back(wy);
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
    }
    else
    {
        std::cout << "Please specify input sep header." << std::endl;
        exit(1);
    }

    new boost::thread(train);

    startGraphics(argc,argv,"SEP Classifier Convolutional Neural Network");
    std::cout << "Finished." << std::endl;

    return 0;
}

