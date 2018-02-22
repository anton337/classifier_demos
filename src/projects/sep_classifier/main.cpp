#include "ConvolutionalNN.h"
#include "sep_reader.h"

ConvolutionalNeuralNetwork <double> * model = NULL;

template<typename T>
T * sample_sep(T * dat,long z,long wx,long wy,long ox,long oy,long nx,long ny,long nz)
{
    // assumption, data is arranged by: dat[x*ny*nz+y*nz+z]
    T * ret = new T[wx*wy + 1]; // last sample contains classification
    if(ox<0||oy<0||ox+wx>=nx||oy+wy>=ny||z<0||z>=nz)
    {
        std::cout << "please make sure to sample within the volume." << std::endl;
        exit(1);
    }
    for(long x=ox,k=0;x<ox+wx;x++)
    {
        for(long y=oy;y<oy+wy;y++,k++)
        {
            ret[k] = dat[x*ny*nz+y*nz+z];
        }
    }
    T num = 0;
    T den = 0;
    T tmp;
    long c = wx*wy/2;
    long w = 3;
    for(long x=-w;x<=w;x++)
    for(long y=-w;y<=w;y++)
    {
        tmp = ret[c+y+wx*x];
        num += tmp;
        den += tmp*tmp;
    }
    num *= num;
    den *= 2*w+1;
    den *= 2*w+1;
    T semb = num/den;
    semb *= semb;
    semb *= semb;
    semb *= semb;
    semb = 1 - semb;
    ret[wx*wy] = semb;
    return ret;
}

int main(int argc,char ** argv)
{
    std::cout << "Starting SEP Classifier Convolutional Neural Network Test ... " << std::endl;

    srand(time(0));

    // load input
    if(argc>0)
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
        long wx = 32;
        long wy = 32;
        long nsamp = 100;
        double * in = new double[nsamp*wx*wy];
        double * out= new double[nsamp];
        long k = 0;
        long pos = 0;
        long neg = 0;
        double thresh = 0.98;
        while(true)
        {
            float * tmp = sample_sep(dat,rand()%nz,wx,wy,rand()%(nx-wx),rand()%(ny-wy),nx,ny,nz);
            if(tmp[wx*wy] < thresh || tmp[wx*wy] > thresh)
            {
                for(long i=0;i<wx*wy;i++)
                {
                    in[k*wx*wy+i] = tmp[i];
                }
                if(tmp[wx*wy] < thresh)
                {
                    neg++;
                    out[k] = 1e-5;
                }
                if(tmp[wx*wy] > thresh)
                {
                    pos++;
                    out[k] = 1;
                }
                k++;
            }
            delete [] tmp;
            if(k>=nsamp)break;
        }
        std::cout << "pos:" << pos << std::endl;
        std::cout << "neg:" << neg << std::endl;

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
        model -> train(0,.01,10000,nsamp,wx*wy,1,in,out);
    }
    else
    {
        std::cout << "Please specify input sep header." << std::endl;
        exit(1);
    }


    return 0;
}

