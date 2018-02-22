#ifndef CONVOLUTIONAL_NEURAL_NETWORK_H
#define CONVOLUTIONAL_NEURAL_NETWORK_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <boost/thread.hpp>

#include "Perceptron.h"

enum LayerType
{
    FULLY_CONNECTED_LAYER       = 1 // => y_l := W_l * y_l-1                <= dEdy_l-1 := W_l * dEdy_l                 dEdW_l-1 := dEdy_l * y_l
  , RELU_LAYER                  = 2 // => y_l := max(0,y_l-1)               <= dEdy_l-1 := dEdy_l
  , MAX_POOLING_LAYER           = 3 // => y_l := max(y_k)                   <= (l==k)?dEdy_l-1=dEdy_l:dEdy_l-1=0        
  , MEAN_POOLING_LAYER          = 4 // => y_l := avg(y_k)                   <= dEdy_l-1 := avg(dEdy_l)
  , CONVOLUTIONAL_LAYER         = 5 // => y_l := W_l * y_l-1                <= dEdy_l-1 := W_l * dEdy_l                 dEdW_l-1 := dEdy_l *_y_l
};

enum ActivationType
{
    IDENTITY        = 1 //                                      f(x) = x                                    f'(x) = 1
  , BINARY_STEP     = 2 //                                      f(x) = (x>=0)?1:0                           f'(x) = (x!=0)?0:inf
  , LOGISTIC        = 3 // (soft step)                          f(x) = 1/(1+exp(-x))                        f'(x) = f(x) (1 - f(x))
  , HYPERBOLIC_TAN  = 4 //                                      f(x) = tanh(x) = (2/(1+exp(-2x))) - 1       f'(x) = 1 - f(x)^2
  , ARC_TAN         = 5 //                                      f(x) = arctan(x)                            f'(x) = 1/(x^2+1)
  , RELU            = 6 // (rectified linear unit)              f(x) = (x>=0)?x:0                           f'(x) = (x>=0)?1:0
  , PRELU           = 7 // (parametric rectified linear unit)   f(x) = (x>=0)?x:a*x                         f'(x) = (x>=0)?1:a
  , ELU             = 7 // (exponential linear unit)            f(x) = (x>=0)?x:a*(exp(x)-1)                f'(x) = (x>=0)?1:f(x)+a
  , SOFT_PLUS       = 8 //                                      f(x) = ln(1+exp(x))                         f'(x) = 1/(1+exp(-x))
};

template<typename T>
struct cnn_training_info
{

    quasi_newton_info<T> * quasi_newton;

    std::vector<long> n_nodes;
    std::vector<LayerType> n_layer_type;
    std::vector<ActivationType> n_activation_type;
    std::vector<long> n_features;
    std::vector<long> kx;
    std::vector<long> ky;
    std::vector<long> nx;
    std::vector<long> ny;
    std::vector<long> pooling_factorx;
    std::vector<long> pooling_factory;
    T **  activation_values;
    T **  deltas;
    long n_variables;
    long n_labels;
    long n_layers;
    long n_elements;

    T *** weights_neuron;
    T **  weights_bias;
    T *** partial_weights_neuron;
    T **  partial_weights_bias;

    T *** mu_weights_neuron;
    T **  mu_weights_bias;
    T *** mu_partial_weights_neuron;
    T **  mu_partial_weights_bias;

    T partial_error;
    T smallest_index;

    T epsilon;

    int type;

    cnn_training_info()
    {

    }

    void init(T _alpha)
    {
        type = 0;
        smallest_index = 0;
        partial_error = 0;
        activation_values  = new T*[n_nodes.size()];
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            switch(n_layer_type[layer])
            {
                case RELU_LAYER :
                    {
                        activation_values [layer] = new T[n_nodes[layer]];
                        break;
                    }
                case FULLY_CONNECTED_LAYER :
                    {
                        activation_values [layer] = new T[n_nodes[layer]];
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        long wx = (kx[layer]/2);
                        long wy = (ky[layer]/2);
                        long dx = nx[layer] - wx*2;
                        long dy = ny[layer] - wy*2;
                        activation_values [layer] = new T[M*nx[layer]*ny[layer]];
                        break;
                    }
                case MAX_POOLING_LAYER :
                case MEAN_POOLING_LAYER :
                    {
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        if(M != N)
                        {
                            std::cout << "pooling layer has to have the same number of input/output features" << std::endl;
                            exit(1);
                        }
                        long dx = nx[layer] / pooling_factorx[layer+1];
                        if(nx[layer] != pooling_factorx[layer+1]*nx[layer+1])
                        {
                            std::cout << "pooling layer nx: " << layer+1 << " should be: " << dx << std::endl;
                            exit(1);
                        }
                        long dy = ny[layer] / pooling_factory[layer+1];
                        if(ny[layer] != pooling_factory[layer+1]*ny[layer+1])
                        {
                            std::cout << "pooling layer ny: " << layer+1 << " should be: " << dy << std::endl;
                            exit(1);
                        }
                        activation_values [layer] = new T[M*nx[layer]*ny[layer]];
                        break;
                    }
                default :
                    {
                        std::cout << "1. Undefined Layer Type. " << std::endl;
                        exit(1);
                        break;
                    }
            }
        }
        deltas = new T*[n_nodes.size()];
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            switch(n_layer_type[layer])
            {
                case RELU_LAYER :
                    {
                        deltas[layer] = new T[n_nodes[layer]];
                        break;
                    }
                case FULLY_CONNECTED_LAYER :
                    {
                        deltas[layer] = new T[n_nodes[layer]];
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        long wx = (kx[layer]/2);
                        long wy = (ky[layer]/2);
                        long dx = nx[layer] - wx*2;
                        long dy = ny[layer] - wy*2;
                        deltas[layer] = new T[M*nx[layer]*nx[layer]];
                        break;
                    }
                case MAX_POOLING_LAYER :
                case MEAN_POOLING_LAYER :
                    {
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        long wx = (kx[layer]/2);
                        long wy = (ky[layer]/2);
                        long dx = nx[layer] - wx*2;
                        long dy = ny[layer] - wy*2;
                        deltas[layer] = new T[M*nx[layer]*nx[layer]];
                        break;
                    }
                default :
                    {
                        std::cout << "2. Undefined Layer Type. " << std::endl;
                        exit(1);
                        break;
                    }
            }
        }
        partial_weights_neuron = new T**[n_layers];
        partial_weights_bias = new T*[n_layers];
        mu_partial_weights_neuron = new T**[n_layers];
        mu_partial_weights_bias = new T*[n_layers];
        mu_weights_neuron = new T**[n_layers];
        mu_weights_bias = new T*[n_layers];
        for(long layer = 0;layer < n_layers;layer++)
        {
            switch(n_layer_type[layer])
            {
                case FULLY_CONNECTED_LAYER :
                    {
                        partial_weights_neuron[layer] = new T*[n_nodes[layer+1]];
                        mu_partial_weights_neuron[layer] = new T*[n_nodes[layer+1]];
                        mu_weights_neuron[layer] = new T*[n_nodes[layer+1]];
                        for(long i=0;i<n_nodes[layer+1];i++)
                        {
                            partial_weights_neuron[layer][i] = new T[n_nodes[layer]];
                            mu_partial_weights_neuron[layer][i] = new T[n_nodes[layer]];
                            mu_weights_neuron[layer][i] = new T[n_nodes[layer]];
                            for(long j=0;j<n_nodes[layer];j++)
                            {
                                partial_weights_neuron[layer][i][j] = 0;
                                mu_partial_weights_neuron[layer][i][j] = 0;
                                mu_weights_neuron[layer][i][j] = 0;
                            }
                        }
                        partial_weights_bias[layer] = new T[n_nodes[layer+1]];
                        mu_partial_weights_bias[layer] = new T[n_nodes[layer+1]];
                        mu_weights_bias[layer] = new T[n_nodes[layer+1]];
                        for(long i=0;i<n_nodes[layer+1];i++)
                        {
                            partial_weights_bias[layer][i] = 0;
                            mu_partial_weights_bias[layer][i] = 0;
                            mu_weights_bias[layer][i] = 0;
                        }
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        //if(n_nodes[layer+1] != ky[layer]*N)
                        //{
                        //    std::cout << "convolutional weight matrix mismatch, " << std::endl;
                        //    std::cout << "n_nodes " << layer+1 << " is " << n_nodes[layer+1] << " should be " << ky[layer]*N << std::endl;
                        //    exit(1);
                        //}
                        //if(n_nodes[layer] != kx[layer]*M)
                        //{
                        //    std::cout << "convolutional weight matrix mismatch, " << std::endl;
                        //    std::cout << "n_nodes " << layer << " is " << n_nodes[layer] << " should be " << kx[layer]*M << std::endl;
                        //    exit(1);
                        //}
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        long wx = (kx[layer]/2);
                        long wy = (ky[layer]/2);
                        long dx = nx[layer] - wx*2;
                        long dy = ny[layer] - wy*2;
                        std::cout << "conv dim:" << ky[layer]*N << "x" << kx[layer]*M << std::endl;
                        partial_weights_neuron[layer] = new T*[ky[layer]*N];
                        mu_partial_weights_neuron[layer] = new T*[ky[layer]*N];
                        mu_weights_neuron[layer] = new T*[ky[layer]*N];
                        for(long i=0;i<ky[layer]*N;i++)
                        {
                            partial_weights_neuron[layer][i] = new T[kx[layer]*M];
                            mu_partial_weights_neuron[layer][i] = new T[kx[layer]*M];
                            mu_weights_neuron[layer][i] = new T[kx[layer]*M];
                            for(long j=0;j<kx[layer]*M;j++)
                            {
                                partial_weights_neuron[layer][i][j] = 0;
                                mu_partial_weights_neuron[layer][i][j] = 0;
                                mu_weights_neuron[layer][i][j] = 0;
                            }
                        }
                        partial_weights_bias[layer] = NULL;
                        mu_partial_weights_bias[layer] = NULL;
                        mu_weights_bias[layer] = NULL;
                        break;
                    }
                case RELU_LAYER :
                case MAX_POOLING_LAYER :
                case MEAN_POOLING_LAYER :
                    {
                        partial_weights_neuron[layer] = NULL;
                        mu_partial_weights_neuron[layer] = NULL;
                        mu_weights_neuron[layer] = NULL;
                        partial_weights_bias[layer] = NULL;
                        mu_partial_weights_bias[layer] = NULL;
                        mu_weights_bias[layer] = NULL;
                        break;
                    }
                default :
                    {
                        std::cout << "3. Undefined Layer Type. " << std::endl;
                        exit(1);
                        break;
                    }
            }
        }
    }

    void reset()
    {
        type = 0;
        smallest_index = 0;
        partial_error = 0;
        for(long layer = 0;layer < n_layers;layer++)
        {
            switch(n_layer_type[layer])
            {
                case FULLY_CONNECTED_LAYER :
                    {
                        for(long i=0;i<n_nodes[layer+1];i++)
                        {
                            for(long j=0;j<n_nodes[layer];j++)
                            {
                                partial_weights_neuron[layer][i][j] = 0;
                            }
                        }
                        for(long i=0;i<n_nodes[layer+1];i++)
                        {
                            partial_weights_bias[layer][i] = 0;
                        }
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        //if(n_nodes[layer+1] != ky[layer]*N)
                        //{
                        //    std::cout << "convolutional weight matrix mismatch, " << std::endl;
                        //    std::cout << "n_nodes " << layer+1 << " is " << n_nodes[layer+1] << " should be " << ky[layer]*N << std::endl;
                        //    exit(1);
                        //}
                        //if(n_nodes[layer] != kx[layer]*M)
                        //{
                        //    std::cout << "convolutional weight matrix mismatch, " << std::endl;
                        //    std::cout << "n_nodes " << layer << " is " << n_nodes[layer] << " should be " << kx[layer]*M << std::endl;
                        //    exit(1);
                        //}
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        long wx = (kx[layer]/2);
                        long wy = (ky[layer]/2);
                        long dx = nx[layer] - wx*2;
                        long dy = ny[layer] - wy*2;
                        for(long i=0;i<ky[layer]*N;i++)
                        {
                            for(long j=0;j<kx[layer]*M;j++)
                            {
                                partial_weights_neuron[layer][i][j] = 0;
                            }
                        }
                        break;
                    }
                case RELU_LAYER :
                case MAX_POOLING_LAYER :
                case MEAN_POOLING_LAYER :
                    {
                        break;
                    }
                default :
                    {
                        std::cout << "4. Undefined Layer Type. " << std::endl;
                        exit(1);
                        break;
                    }
            }
        }
    }

    void destroy()
    {
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            delete [] activation_values [layer];
        }
        delete [] activation_values;
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            delete [] deltas [layer];
        }
        delete [] deltas;
        for(long layer = 0;layer < n_layers;layer++)
        {
            switch(n_layer_type[layer])
            {
                case FULLY_CONNECTED_LAYER :
                    {
                        for(long i=0;i<n_nodes[layer+1];i++)
                        {
                            delete [] partial_weights_neuron[layer][i];
                            delete [] mu_partial_weights_neuron[layer][i];
                            delete [] mu_weights_neuron[layer][i];
                        }
                        delete [] partial_weights_neuron[layer];
                        delete [] mu_partial_weights_neuron[layer];
                        delete [] mu_weights_neuron[layer];
                        delete [] partial_weights_bias[layer];
                        delete [] mu_partial_weights_bias[layer];
                        delete [] mu_weights_bias[layer];
                    }
                    break;
                case CONVOLUTIONAL_LAYER :
                    {
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        long wx = (kx[layer]/2);
                        long wy = (ky[layer]/2);
                        long dx = nx[layer] - wx*2;
                        long dy = ny[layer] - wy*2;
                        for(long i=0;i<ky[layer]*N;i++)
                        {
                            delete [] partial_weights_neuron[layer][i];
                            delete [] mu_partial_weights_neuron[layer][i];
                            delete [] mu_weights_neuron[layer][i];
                        }
                        delete [] partial_weights_neuron[layer];
                        delete [] mu_partial_weights_neuron[layer];
                        delete [] mu_weights_neuron[layer];
                        break;
                    }
                default :
                    break;
            }
        }
        delete [] partial_weights_neuron;
        delete [] mu_partial_weights_neuron;
        delete [] mu_weights_neuron;
        delete [] partial_weights_bias;
        delete [] mu_partial_weights_bias;
        delete [] mu_weights_bias;
    }

    void update_gradient ()
    {
        if(quasi_newton != NULL)
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                for(long i=0;i<n_nodes[layer+1];i++)
                {
                    for(long j=0;j<n_nodes[layer];j++,k++)
                    {
                        quasi_newton->grad_tmp[k] += partial_weights_neuron[layer][i][j];
                    }
                }
                for(long i=0;i<n_nodes[layer+1];i++,k++)
                {
                    quasi_newton->grad_tmp[k] += partial_weights_bias[layer][i];
                }
            }
        }
    }

    void globalUpdate()
    {
        //if(quasi_newton != NULL && quasi_newton->quasi_newton_update)
        //{
        //    for(long layer = 0,k = 0;layer < n_layers;layer++)
        //    {
        //        if(weights_neuron[layer] != NULL)
        //        {
        //            for(long i=0;i<n_nodes[layer+1];i++)
        //            {
        //                for(long j=0;j<n_nodes[layer];j++,k++)
        //                {
        //                    weights_neuron[layer][i][j] += quasi_newton->dX[k];
        //                }
        //            }
        //        }
        //        if(weights_bias[layer] != NULL)
        //        {
        //            for(long i=0;i<n_nodes[layer+1];i++,k++)
        //            {
        //                weights_bias[layer][i] += quasi_newton->dX[k];
        //            }
        //        }
        //    }
        //}
        //else if(quasi_newton != NULL)
        //{
        //    for(long layer = 0,k = 0;layer < n_layers;layer++)
        //    {
        //        if(weights_neuron[layer] != NULL)
        //        {
        //            for(long i=0;i<n_nodes[layer+1];i++)
        //            {
        //                for(long j=0;j<n_nodes[layer];j++,k++)
        //                {
        //                    weights_neuron[layer][i][j] += epsilon * quasi_newton->grad_tmp[k];
        //                }
        //            }
        //        }
        //        if(weights_bias[layer] != NULL)
        //        {
        //            for(long i=0;i<n_nodes[layer+1];i++,k++)
        //            {
        //                weights_bias[layer][i] += epsilon * quasi_newton->grad_tmp[k];
        //            }
        //        }
        //    }
        //}
        //else
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                switch(n_layer_type[layer])
                {
                    case FULLY_CONNECTED_LAYER :
                        {
                            {
                                for(long i=0;i<n_nodes[layer+1];i++)
                                {
                                    for(long j=0;j<n_nodes[layer];j++,k++)
                                    {
                                        weights_neuron[layer][i][j] += epsilon * partial_weights_neuron[layer][i][j];
                                    }
                                }
                            }
                            {
                                for(long i=0;i<n_nodes[layer+1];i++,k++)
                                {
                                    weights_bias[layer][i] += epsilon * partial_weights_bias[layer][i];
                                }
                            }
                            break;
                        }
                    case CONVOLUTIONAL_LAYER :
                        {
                            //if(n_nodes[layer+1] != ky[layer]*N)
                            //{
                            //    std::cout << "convolutional weight matrix mismatch, " << std::endl;
                            //    std::cout << "n_nodes " << layer+1 << " is " << n_nodes[layer+1] << " should be " << ky[layer]*N << std::endl;
                            //    exit(1);
                            //}
                            //if(n_nodes[layer] != kx[layer]*M)
                            //{
                            //    std::cout << "convolutional weight matrix mismatch, " << std::endl;
                            //    std::cout << "n_nodes " << layer << " is " << n_nodes[layer] << " should be " << kx[layer]*M << std::endl;
                            //    exit(1);
                            //}
                            long M = n_features[layer];
                            long N = n_features[layer+1];
                            long wx = (kx[layer]/2);
                            long wy = (ky[layer]/2);
                            long dx = nx[layer] - wx*2;
                            long dy = ny[layer] - wy*2;
                            {
                                for(long i=0;i<ky[layer]*N;i++)
                                {
                                    for(long j=0;j<kx[layer]*M;j++,k++)
                                    {
                                        weights_neuron[layer][i][j] += epsilon * partial_weights_neuron[layer][i][j];
                                    }
                                }
                            }
                            //{
                            //    for(long i=0;i<n_nodes[layer+1];i++,k++)
                            //    {
                            //        weights_bias[layer][i] += epsilon * partial_weights_bias[layer][i];
                            //    }
                            //}
                            break;
                        }
                    default :
                        {
                            break;
                        }
                }
            }
        }
    }

};

template<typename T>
void cnn_training_worker(long n_threads,long iter,cnn_training_info<T> * g,std::vector<long> const & vrtx,T * variables,T * labels)
{
    {
        for(long n=0;n<vrtx.size();n++)
        {
            //std::cout << "n:" << n << std::endl;
            T avg_factor = 1.0 / (1.0 + n);
            // initialize input activations
            for(long i=0;i<g->n_nodes[0];i++)
            {
                g->activation_values[0][i] = variables[vrtx[n]*g->n_variables+i];
                //std::cout << ((g->activation_values[0][i]>0.5)?'*':' ');
                //if(i%28==0)std::cout << std::endl;
            }
            // forward propagation
            for(long layer = 0; layer < g->n_layers; layer++)
            {
                switch ( g->n_layer_type[layer] )
                {
                    case RELU_LAYER :
                        {
                            for(long i=0;i<g->n_nodes[layer+1];i++)
                            {
                                g->activation_values[layer+1][i] = max(0.0,g->activation_values[layer][i]);
                            }
                            break;
                        }
                    case FULLY_CONNECTED_LAYER :
                        {
                            for(long i=0;i<g->n_nodes[layer+1];i++)
                            {
                                    T sum = g->weights_bias[layer][i];
                                    for(long j=0;j<g->n_nodes[layer];j++)
                                    {
                                        // W * y
                                        sum += g->weights_neuron[layer][i][j] * g->activation_values[layer][j];
                                    }
                                    g->activation_values[layer+1][i] = sigmoid(sum,g->type);
                            }
                            break;
                        }
                    case CONVOLUTIONAL_LAYER :
                        {

                            //************************************************************************************//
                            //
                            //                  _____
                            //      Y_{k+1}     |   ||||||||||       N features
                            //                  |   ||||||||||
                            //                  |___||||||||||
                            //
                            //
                            //
                            //
                            //                              M                           M
                            //                  _________________________           _________
                            //                  |    |                  |           |       |
                            //                  |    | ky               |           |   b   |  N
                            //     \        /   |____|                  |           |       |
                            //      \  /\  /    |                       |           |_______|
                            //       \/  \/     | kx                    |
                            //                  |                       |  N
                            //                  |                       |
                            //                  |                       |
                            //                  |                       |
                            //                  |                       |
                            //                  |_______________________|
                            //
                            //
                            //
                            //
                            //
                            //                 ________________
                            //      Y_{k}      |        | | | |        M features
                            //                 |        | | | |
                            //                 |        | | | |
                            //                 |        | | | |
                            //                 |________|_|_|_|
                            //
                            //
                            //
                            //
                            //
                            //      Y_{k+1} = s ( W * Y_{k} + b )
                            //
                            //
                            //
                            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            //
                            //      W is arranged by:
                            //
                            //      W[N * ky][M * kx]      // contains M * N convolution kernels
                            //
                            //      // row major
                            //
                            //      W[N * ky + y][M * kx + x]
                            //
                            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            //
                            //      dx = nx - (kx/2)*2
                            //      dy = ny - (ky/2)*2
                            //
                            //      b[N * (dx * dy)]      // contains N bias terms
                            //
                            //      // row major
                            //
                            //      b[(dx * dy) * n + (dx) * y + x]
                            //
                            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            //
                            //      Y_{k} [M * (nx * ny)]       // contains M features from current layer
                            //
                            //      // row major
                            //
                            //      Y_{k} [(nx * ny) * m + (nx) * y + x]
                            //
                            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            //
                            //      dx = nx - (kx/2)*2
                            //      dy = ny - (ky/2)*2
                            //
                            //      Y_{k+1} [N * (dx * dy)]     // contains N features from next layer
                            //
                            //      // row major
                            //
                            //      Y_{k+1} [(dx * dy) * n + (dx) * y + x]
                            //
                            //
                            //
                            //************************************************************************************//

                            // j : n_nodes[layer  ] = curr size = M * nx * ny
                            // i : n_nodes[layer+1] = next size = N * dx * dy
                            long M = g->n_features[layer];
                            long N = g->n_features[layer+1];
                            long kx = g->kx[layer];
                            long ky = g->ky[layer];
                            long nx = g->nx[layer];
                            long ny = g->ny[layer];
                            long wx = (kx/2);
                            long wy = (ky/2);
                            long dx = nx - wx*2;
                            long dy = ny - wy*2;
                            T factor = 1.0 / (kx*ky);

                            {
                                for(long n=0,i=0;n<N;n++)
                                {
                                    for(long oy=0;oy<dy;oy++)
                                    for(long ox=0;ox<dx;ox++,i++)
                                    {
                                        T sum = 0.0;//g->weights_bias[layer][i];
                                        long ix = ox+wx;
                                        long iy = oy+wy;
                                        for(long m=0;m<M;m++)
                                        {
                                            for(long fy=-wy,ty=0;fy<=wy;fy++,ty++)
                                            for(long fx=-wx,tx=0;fx<=wx;fx++,tx++)
                                            {
                                                // W * y
                                                sum += g->weights_neuron[layer][ky*n+ty][kx*m+tx]
                                                     * g->activation_values[layer][(nx*ny)*m + nx*(iy+fy) + (ix+fx)]
                                                     * factor;
                                            }
                                        }
                                        g->activation_values[layer+1][i] = sigmoid(sum,2); // arctan
                                        //std::cout << g->activation_values[layer+1][i] << "  ";
                                        //std::cout << sum << "  ";
                                        //if(i%dx==0)std::cout << std::endl;
                                    }
                                }
                            }
                            break;
                        }
                    case MAX_POOLING_LAYER :
                        {
                            // j : n_nodes[layer  ] = curr size = M * nx * ny
                            // i : n_nodes[layer+1] = next size = N * dx * dy
                            long M = g->n_features[layer];
                            long nx = g->nx[layer];
                            long ny = g->ny[layer];
                            long factorx = g->pooling_factorx[layer];
                            long factory = g->pooling_factory[layer];
                            long dx = nx / factorx;
                            long dy = ny / factory;
                            T tmp_val,max_val;

                            for(long m=0,i=0;m<M;m++)
                            {
                                for(long y=0,oy=0;y<ny;y+=factory,oy++)
                                for(long x=0,ox=0;x<nx;x+=factorx,ox++)
                                {
                                    max_val = -100000000;
                                    for(long ty=0;ty<factory;ty++)
                                    for(long tx=0;tx<factorx;tx++)
                                    {
                                        tmp_val = g->activation_values[layer][m*nx*nx+nx*y+x];
                                        if(tmp_val>max_val)
                                        {
                                          max_val = tmp_val;
                                        }
                                    }
                                    g->activation_values[layer+1][m*dx*dy+dx*oy+ox] = max_val;
                                }
                            }
                            break;
                        }
                    case MEAN_POOLING_LAYER :
                        {
                            // j : n_nodes[layer  ] = curr size = M * nx * ny
                            // i : n_nodes[layer+1] = next size = N * dx * dy
                            long M = g->n_features[layer];
                            long nx = g->nx[layer];
                            long ny = g->ny[layer];
                            long factorx = g->pooling_factorx[layer];
                            long factory = g->pooling_factory[layer];
                            long dx = nx / factorx;
                            long dy = ny / factory;
                            T tmp_val,mean_val;
                            T factor = 1.0 / (factorx * factory);

                            for(long m=0,i=0;m<M;m++)
                            {
                                for(long y=0,oy=0;y<ny;y+=factory,oy++)
                                for(long x=0,ox=0;x<nx;x+=factorx,ox++)
                                {
                                    mean_val = 0;
                                    for(long ty=0;ty<factory;ty++)
                                    for(long tx=0;tx<factorx;tx++)
                                    {
                                        mean_val += g->activation_values[layer][m*nx*nx+nx*y+x];
                                    }
                                    g->activation_values[layer+1][m*dx*dy+dx*oy+ox] = mean_val * factor;
                                }
                            }
                            break;
                        }
                    default :
                        {
                            std::cout << "1. Layer type not defined." << std::endl;
                            exit(1);
                        }
                }
            }
            //char ch;
            //std::cin >> ch;
            long last_layer = g->n_nodes.size()-2;
            // initialize observed labels
            T partial_error = 0;
            long max_i = 0;
            T max_val = 0;
            T max_act_val = 0;
            T min_act_val = 1;
            T max_label = 0;
            T min_label = 1;
            for(long i=0;i<g->n_nodes[last_layer];i++)
            {
                g->deltas[last_layer+1][i] = labels[vrtx[n]*g->n_labels+i] - g->activation_values[last_layer][i];
                if(fabs(g->activation_values[last_layer][i]) > max_act_val)
                {
                    max_act_val = fabs(g->activation_values[last_layer][i]);
                }
                if(fabs(g->activation_values[last_layer][i]) < min_act_val)
                {
                    min_act_val = fabs(g->activation_values[last_layer][i]);
                }
                if(fabs(labels[vrtx[n]*g->n_labels+i]) > max_label)
                {
                    max_label = fabs(labels[vrtx[n]*g->n_labels+i]);
                }
                if(fabs(labels[vrtx[n]*g->n_labels+i]) < min_label)
                {
                    min_label = fabs(labels[vrtx[n]*g->n_labels+i]);
                }
            }
            //std::cout << "vrtx[" << n << "]=" << vrtx[n] << std::endl;
            //std::cout << "max activation value:" << max_act_val << std::endl;
            //std::cout << "min activation value:" << min_act_val << std::endl;
            //std::cout << "max label:" << max_label << std::endl;
            //std::cout << "min label:" << min_label << std::endl;
            for(long i=0;i<g->n_nodes[last_layer];i++)
            {
                partial_error += fabs(g->deltas[last_layer+1][i]);
            }
            g->partial_error += partial_error;

            // back propagation
            for(long layer = g->n_layers-1; layer >= 0; layer--)
            {
                // back propagate deltas
                if(layer+1==last_layer)
                {
                    for(long i=0;i<g->n_nodes[layer+1];i++)
                    {
                        g->deltas[layer+1][i] = 0;
                        for(long j=0;j<g->n_nodes[layer+2];j++)
                        {
                                g->deltas[layer+1][i] += 
                                    // dEdy
                                    (
                                      dsigmoid(g->activation_values[layer+1][i],g->type)
                                    * g->deltas[layer+2][j]
                                    )
                                    ;
                        }
                    }
                }
                else
                {
                    switch ( g->n_layer_type[layer+1] )
                    {
                        case RELU_LAYER :
                            {
                                T max_deltas = 0;
                                for(long i=0;i<g->n_nodes[layer+1];i++)
                                {
                                    if(g->activation_values[layer+1][i] < 0.0)
                                    {
                                        g->deltas[layer+1][i] = 0;
                                    }
                                    else
                                    {
                                        g->deltas[layer+1][i] = g->deltas[layer+2][i];
                                    }
                                    if(fabs(g->deltas[layer+1][i])>max_deltas)max_deltas = fabs(g->deltas[layer+1][i]);
                                }
                                //std::cout << layer << " relu max deltas:" << max_deltas << std::endl;
                                break;
                            }
                        case FULLY_CONNECTED_LAYER :
                            {
                                for(long i=0;i<g->n_nodes[layer+1];i++)
                                {
                                    g->deltas[layer+1][i] = 0;
                                    for(long j=0;j<g->n_nodes[layer+2];j++)
                                    {
                                            g->deltas[layer+1][i] += 
                                                // dEdy
                                                (
                                                  dsigmoid(g->activation_values[layer+1][i],g->type)
                                                * g->deltas[layer+2][j]
                                                )
                                                // W
                                                * g->weights_neuron[layer+1][j][i]
                                                ;
                                    }
                                }
                                break;
                            }
                        case CONVOLUTIONAL_LAYER :
                            {
                                // i : n_nodes[layer+1] = curr size = M * nx * ny
                                // j : n_nodes[layer+2] = next size = N * dx * dy
                                long M = g->n_features[layer+1];
                                long N = g->n_features[layer+2];
                                long kx = g->kx[layer+1];
                                long ky = g->ky[layer+1];
                                long nx = g->nx[layer+1];
                                long ny = g->ny[layer+1];
                                long wx = (kx/2);
                                long wy = (ky/2);
                                long dx = nx - wx*2;
                                long dy = ny - wy*2;
                                T max_deltas = 0;
                                T max_prev_deltas = 0;
                                for(long m=0,i=0;m<M;m++)
                                {
                                    {
                                        for(long iy=wy;iy+wy<ny;iy++)
                                        for(long ix=wx;ix+wx<nx;ix++,i++)
                                        {
                                            g->deltas[layer+1][i] = 0;
                                            for(long n=0;n<N;n++)
                                            {
                                                long vy = iy-wy;
                                                long vx = ix-wx;
                                                for(long fy=-wy,ty=0;fy<=wy;fy++,ty++)
                                                for(long fx=-wx,tx=0;fx<=wx;fx++,tx++)
                                                {
                                                    g->deltas[layer+1][i] +=
                                                        // dEdy
                                                        (
                                                          g->deltas[layer+2][(dx*dy)*n + dx*vy + vx]
                                                        )
                                                        // W
                                                        * g->weights_neuron[layer+1][ky*n+ty][kx*m+tx] 
                                                        ;
                                                    if(fabs(g->deltas[layer+2][(dx*dy)*n + dx*vy + vx])>max_prev_deltas)max_prev_deltas = fabs(g->deltas[layer+2][(dx*dy)*n + dx*vy + vx]);
                                                }
                                            }
                                            if(fabs(g->deltas[layer+1][i])>max_deltas)max_deltas = fabs(g->deltas[layer+1][i]);
                                            g->deltas[layer+1][i] *= dsigmoid(g->activation_values[layer+1][(nx*ny)*m + nx*iy + ix],2);
                                        }
                                    }
                                }
                                //std::cout << layer << " conv max deltas:" << max_deltas << " " << max_prev_deltas << std::endl;
                                break;
                            }
                        case MAX_POOLING_LAYER :
                            {
                                // j : n_nodes[layer  ] = curr size = M * nx * ny
                                // i : n_nodes[layer+1] = next size = N * dx * dy
                                long M = g->n_features[layer+1];
                                long nx = g->nx[layer+1];
                                long ny = g->ny[layer+1];
                                long factorx = g->pooling_factorx[layer+1];
                                long factory = g->pooling_factory[layer+1];
                                long dx = nx / factorx;
                                long dy = ny / factory;
                                T tmp_val,max_val;

                                for(long m=0,i=0;m<M;m++)
                                {
                                    for(long y=0,oy=0;y<ny;y+=factory,oy++)
                                    for(long x=0,ox=0;x<nx;x+=factorx,ox++)
                                    {
                                        for(long ty=0;ty<factory;ty++)
                                        for(long tx=0;tx<factorx;tx++)
                                        {
                                            if(fabs(g->activation_values[layer+1][m*nx*ny+nx*y+x] - g->activation_values[layer+2][m*dx*dy+dx*oy+ox]) < 1e-8)
                                            {
                                                g->deltas[layer+1][m*nx*ny+nx*y+x] = g->deltas[layer+2][m*dx*dy+dx*oy+ox];
                                            }
                                            else
                                            {
                                                g->deltas[layer+1][m*nx*ny+nx*y+x] = 0;
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        case MEAN_POOLING_LAYER :
                            {
                                std::cout << "mean pooling backward propagation" << std::endl;
                                // j : n_nodes[layer  ] = curr size = M * nx * ny
                                // i : n_nodes[layer+1] = next size = N * dx * dy
                                long M = g->n_features[layer+1];
                                long nx = g->nx[layer+1];
                                long ny = g->ny[layer+1];
                                long factorx = g->pooling_factorx[layer+1];
                                long factory = g->pooling_factory[layer+1];
                                long dx = nx / factorx;
                                long dy = ny / factory;
                                T tmp_val,max_val;

                                for(long m=0,i=0;m<M;m++)
                                {
                                    for(long y=0,oy=0;y<ny;y+=factory,oy++)
                                    for(long x=0,ox=0;x<nx;x+=factorx,ox++)
                                    {
                                        for(long ty=0;ty<factory;ty++)
                                        for(long tx=0;tx<factorx;tx++)
                                        {
                                            {
                                                g->deltas[layer+1][m*nx*ny+nx*y+x] = g->deltas[layer+2][m*dx*dy+dx*oy+ox];
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        default :
                            {
                                std::cout << "2. Layer type not defined." << std::endl;
                                exit(1);
                            }
                    }
                }
                    
                
                // biases
                switch ( g->n_layer_type[layer] )
                {
                    case FULLY_CONNECTED_LAYER :
                        {
                            for(long i=0;i<g->n_nodes[layer+1];i++)
                            {
                                    g->partial_weights_bias[layer][i] += 
                                        ( 
                                            (
                                                g->deltas[layer+1][i] 
                                            )
                                          
                                        - g->partial_weights_bias[layer][i] 
                                        ) * avg_factor;
                            }
                            break;
                        }
                    case CONVOLUTIONAL_LAYER :
                        {
                            //for(long i=0;i<g->n_nodes[layer+1];i++)
                            //{
                            //        g->partial_weights_bias[layer][i] += 
                            //            ( 
                            //                (
                            //                    g->deltas[layer+1][i] 
                            //                )
                            //              
                            //            - g->partial_weights_bias[layer][i] 
                            //            ) * avg_factor;
                            //}
                            break;
                        }
                    case RELU_LAYER :
                    case MAX_POOLING_LAYER :
                    case MEAN_POOLING_LAYER :
                        {
                            break;
                        }
                    default :
                        {
                            std::cout << "3. Layer type not defined." << std::endl;
                            exit(1);
                        }
                }
                
                // neuron weights
                switch ( g->n_layer_type[layer] )
                {
                    case FULLY_CONNECTED_LAYER :
                        {
                            for(long i=0;i<g->n_nodes[layer+1];i++)
                            {
                                for(long j=0;j<g->n_nodes[layer];j++)
                                {
                                        g->partial_weights_neuron[layer][i][j] += 
                                            ( 
                                                (
                                                    // dEdy
                                                    g->deltas[layer+1][i] 
                                                    // y
                                                  * g->activation_values[layer][j]
                                                )
                                              
                                            - g->partial_weights_neuron[layer][i][j]
                                            ) * avg_factor;
                                }
                            }
                            break;
                        }
                    case CONVOLUTIONAL_LAYER :
                        {
                            long M = g->n_features[layer];
                            long N = g->n_features[layer+1];
                            long kx = g->kx[layer];
                            long ky = g->ky[layer];
                            long nx = g->nx[layer];
                            long ny = g->ny[layer];
                            long wx = (kx/2)*2;
                            long wy = (ky/2)*2;
                            long dx = nx - (kx/2)*2;
                            long dy = ny - (ky/2)*2;
                            {
                                for(long n=0;n<N;n++)
                                {
                                    for(long ty=0;ty<ky;ty++)
                                    {
                                        for(long m=0;m<M;m++)
                                        {
                                            for(long tx=0;tx<kx;tx++)
                                            {
                                                long x = kx*m+tx;
                                                long y = ky*n+ty;
                                                g->mu_partial_weights_neuron[layer][ky*n+ty][kx*m+tx] = 0;
                                            }
                                        }
                                    }
                                }
                            }
                            T max_disp = 0;
                            T max_wght = 0;
                            T tmp_disp;
                            T fact = 1.0;
                            for(long ty=0,fy=-wy;ty<ky;ty++,fy++)
                            {
                                for(long tx=0,fx=-wx;tx<kx;tx++,fx++)
                                {
                                    for(long m=0,j=0;m<M;m++)
                                    {
                                        for(long n=0,i=0;n<N;n++)
                                        {
                                            for(long vy=0;vy<dy;vy++)
                                            for(long vx=0;vx<dx;vx++,i++)
                                            {
                                                long iy = vy+wy;
                                                long ix = vx+wx;
                                                g->mu_partial_weights_neuron[layer][ky*n+ty][kx*m+tx] += 
                                                        (
                                                            // dEdy
                                                            g->deltas[layer+1][(dx*dy)*n + dx*vy + vx] 
                                                            // y
                                                          * g->activation_values[layer][(nx*ny)*m + nx*(iy+fy) + (ix+fx)]
                                                        ) * fact
                                                        ;
                                            }
                                        }
                                    }
                                }
                            }
                            {
                                for(long n=0;n<N;n++)
                                {
                                    for(long ty=0;ty<ky;ty++)
                                    {
                                        for(long m=0;m<M;m++)
                                        {
                                            for(long tx=0;tx<kx;tx++)
                                            {
                                                g->partial_weights_neuron[layer][ky*n+ty][kx*m+tx] +=
                                                  (
                                                      g->mu_partial_weights_neuron[layer][ky*n+ty][kx*m+tx]
                                                  -   g->   partial_weights_neuron[layer][ky*n+ty][kx*m+tx]
                                                  ) * avg_factor
                                                  ;
                                                if(fabs(g->mu_partial_weights_neuron[layer][ky*n+ty][kx*m+tx])>max_disp)
                                                {
                                                  max_disp = fabs(g->mu_partial_weights_neuron[layer][ky*n+ty][kx*m+tx]);
                                                }
                                                if(fabs(g->weights_neuron[layer][ky*n+ty][kx*m+tx])>max_wght)
                                                {
                                                  max_wght = fabs(g->weights_neuron[layer][ky*n+ty][kx*m+tx]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            //std::cout << "max disp:" << max_disp << std::endl;
                            //std::cout << "max wght:" << max_wght << std::endl;
                            break;
                        }
                    case RELU_LAYER :
                    case MAX_POOLING_LAYER :
                    case MEAN_POOLING_LAYER :
                        {
                            break;
                        }
                    default :
                        {
                            std::cout << "4. Layer type not defined." << std::endl;
                            exit(1);
                        }
                }
            }
        }
        /*
        for(long layer = 0; layer < g->n_layers; layer++)
        {
            switch(g->n_layer_type[layer])
            {
                case FULLY_CONNECTED_LAYER :
                    {
                        // biases
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            g->partial_weights_bias[layer][i] /= n_threads;
                        }
                        // neuron weights
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            for(long j=0;j<g->n_nodes[layer];j++)
                            {
                                g->partial_weights_neuron[layer][i][j] /= n_threads;
                            }
                        }
                        // biases
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            g->mu_weights_bias[layer][i] = g->weights_bias[layer][i];
                        }
                        // neuron weights
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            for(long j=0;j<g->n_nodes[layer];j++)
                            {
                                g->mu_weights_neuron[layer][i][j] = g->weights_neuron[layer][i][j];
                            }
                        }
                        // biases
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            g->mu_partial_weights_bias[layer][i] = g->partial_weights_bias[layer][i];
                        }
                        // neuron weights
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            for(long j=0;j<g->n_nodes[layer];j++)
                            {
                                g->mu_partial_weights_neuron[layer][i][j] = g->partial_weights_neuron[layer][i][j];
                            }
                        }
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        long M = g->n_features[layer-1];
                        long N = g->n_features[layer];
                        long wx = (g->kx[layer-1]/2);
                        long wy = (g->ky[layer-1]/2);
                        long dx = g->nx[layer-1] - wx*2;
                        long dy = g->ny[layer-1] - wy*2;
                        // biases
                        for(long i=0;i<N*dx*dy;i++)
                        {
                            g->partial_weights_bias[layer][i] /= n_threads;
                        }
                        // neuron weights
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            for(long j=0;j<g->n_nodes[layer];j++)
                            {
                                g->partial_weights_neuron[layer][i][j] /= n_threads;
                            }
                        }
                        // biases
                        for(long i=0;i<N*dx*dy;i++)
                        {
                            g->mu_weights_bias[layer][i] = g->weights_bias[layer][i];
                        }
                        // neuron weights
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            for(long j=0;j<g->n_nodes[layer];j++)
                            {
                                g->mu_weights_neuron[layer][i][j] = g->weights_neuron[layer][i][j];
                            }
                        }
                        // biases
                        for(long i=0;i<N*dx*dy;i++)
                        {
                            g->mu_partial_weights_bias[layer][i] = g->partial_weights_bias[layer][i];
                        }
                        // neuron weights
                        for(long i=0;i<g->n_nodes[layer+1];i++)
                        {
                            for(long j=0;j<g->n_nodes[layer];j++)
                            {
                                g->mu_partial_weights_neuron[layer][i][j] = g->partial_weights_neuron[layer][i][j];
                            }
                        }
                        break;
                    }
                default :
                    {
                        std::cout << "5. Undefined Layer Type. " << std::endl;
                        exit(1);
                        break;
                    }
            }
        }
        */
    }
}

template<typename T>
struct ConvolutionalNeuralNetwork
{
    quasi_newton_info<T> * quasi_newton;

    T ierror;
    T perror;
    T final_error;

    T *** weights_neuron;
    T **  weights_bias;
    T **  activation_values;
    T **  activation_values1;
    T **  activation_values2;
    T **  activation_values3;
    T **  deltas;

    long n_inputs;
    long n_outputs;
    long n_layers;
    std::vector<long> n_nodes;
    std::vector<LayerType> n_layer_type;
    std::vector<ActivationType> n_activation_type;
    std::vector<long> n_features;
    std::vector<long> kx;
    std::vector<long> ky;
    std::vector<long> nx;
    std::vector<long> ny;
    std::vector<long> pooling_factorx;
    std::vector<long> pooling_factory;

    bool continue_training;
    bool stop_training;

    std::vector<T> errs;
    std::vector<T> test_errs;

    T get_variable(int ind)
    {
        int I = 0;
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
              if(I==ind)return weights_bias[layer][i];
              I++;
            }
          }
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(int i=0;i<n_nodes[layer+1];i++)
            {
                for(int j=0;j<n_nodes[layer];j++)
                {
                    if(I==ind)return weights_neuron[layer][i][j];
                    I++;
                }
            }
          }
        return 0;
    }

    int get_num_variables()
    {
        int I = 0;
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
              I++;
            }
          }
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(int i=0;i<n_nodes[layer+1];i++)
            {
                for(int j=0;j<n_nodes[layer];j++)
                {
                    I++;
                }
            }
          }
        return I;
    }


    T epsilon;
    T alpha;
    int sigmoid_type;

    ConvolutionalNeuralNetwork ( std::vector <      long      > p_nodes 
                               , std::vector <      LayerType > p_layer_type
                               , std::vector < ActivationType > p_activation_type
                               , std::vector <      long      > p_features
                               , std::vector <      long      > p_kx
                               , std::vector <      long      > p_ky
                               , std::vector <      long      > p_nx
                               , std::vector <      long      > p_ny
                               , std::vector <      long      > p_pooling_factorx
                               , std::vector <      long      > p_pooling_factory
                               )
    {

        quasi_newton = NULL;

        continue_training = false;
        stop_training = false;

        sigmoid_type = 0;
        alpha = 0.1;

        ierror = 1e10;
        perror = 1e10;

                  n_nodes =           p_nodes;
             n_layer_type =      p_layer_type;
        n_activation_type = p_activation_type;
               n_features =        p_features;
                       kx =              p_kx;
                       ky =              p_ky;
                       nx =              p_nx;
                       ny =              p_ny;
          pooling_factorx = p_pooling_factorx;
          pooling_factory = p_pooling_factory;
        n_inputs = n_nodes[0];
        n_outputs = n_nodes[n_nodes.size()-1];
        n_layers = n_nodes.size()-2; // first and last numbers and output and input dimensions, so we have n-2 layers

        weights_neuron = new T**[n_layers];
        weights_bias = new T*[n_layers];
        activation_values  = new T*[n_nodes.size()];
        activation_values1 = new T*[n_nodes.size()];
        activation_values2 = new T*[n_nodes.size()];
        activation_values3 = new T*[n_nodes.size()];
        deltas = new T*[n_nodes.size()];
        
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            switch(n_layer_type[layer])
            {
                case RELU_LAYER :
                    {
                        if(n_nodes[layer+1] != n_nodes[layer])
                        {
                            std::cout << "layer " << layer+1 << " relu layer should have same number of neurons as output " << std::endl;
                            std::cout << "n_nodes " << layer+1 << " should be " << n_nodes[layer] << std::endl;
                            exit(1);
                        }
                        if(n_features[layer+1] != n_features[layer])
                        {
                            std::cout << "layer " << layer+1 << " relu layer should have same number of features as output " << std::endl;
                            std::cout << "n_features " << layer+1 << " should be " << n_features[layer] << std::endl;
                            exit(1);
                        }
                        if(nx[layer+1] != nx[layer])
                        {
                            std::cout << "layer " << layer+1 << " relu layer should have same nx as output " << std::endl;
                            std::cout << "nx " << layer+1 << " should be " << nx[layer] << std::endl;
                            exit(1);
                        }
                        if(ny[layer+1] != ny[layer])
                        {
                            std::cout << "layer " << layer+1 << " relu layer should have same ny as output " << std::endl;
                            std::cout << "ny " << layer+1 << " should be " << ny[layer] << std::endl;
                            exit(1);
                        }
                        activation_values [layer] = new T[n_nodes[layer]];
                        activation_values1[layer] = new T[n_nodes[layer]];
                        activation_values2[layer] = new T[n_nodes[layer]];
                        activation_values3[layer] = new T[n_nodes[layer]];
                        break;
                    }
                case FULLY_CONNECTED_LAYER :
                    {
                        activation_values [layer] = new T[n_nodes[layer]];
                        activation_values1[layer] = new T[n_nodes[layer]];
                        activation_values2[layer] = new T[n_nodes[layer]];
                        activation_values3[layer] = new T[n_nodes[layer]];
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        long wx = (kx[layer]/2);
                        long wy = (ky[layer]/2);
                        long dx = nx[layer] - wx*2;
                        long dy = ny[layer] - wy*2;
                        if(n_nodes[layer+1] != N*dx*dy)
                        {
                            std::cout << "layer " << layer+1 << " n_nodes convolutional is: " << n_nodes[layer+1] << " should be: " << N*dx*dy << std::endl;
                            exit(1);
                        }
                        if(nx[layer+1] != dx)
                        {
                            std::cout << "nx[" << layer+1 << "] should be: " << dx << std::endl;
                            exit(1);
                        }
                        if(ny[layer+1] != dy)
                        {
                            std::cout << "ny[" << layer+1 << "] should be: " << dy << std::endl;
                            exit(1);
                        }
                        activation_values [layer] = new T[M*nx[layer]*ny[layer]];
                        activation_values1[layer] = new T[M*nx[layer]*ny[layer]];
                        activation_values2[layer] = new T[M*nx[layer]*ny[layer]];
                        activation_values3[layer] = new T[M*nx[layer]*ny[layer]];
                        break;
                    }
                case MAX_POOLING_LAYER :
                case MEAN_POOLING_LAYER :
                    {
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        if(M != N)
                        {
                            std::cout << "pooling layer has to have the same number of input/output features" << std::endl;
                            exit(1);
                        }
                        long dx = nx[layer] / pooling_factorx[layer+1];
                        if(nx[layer] != pooling_factorx[layer+1]*nx[layer+1])
                        {
                            std::cout << "pooling layer nx: " << layer+1 << " should be: " << dx << std::endl;
                            exit(1);
                        }
                        long dy = ny[layer] / pooling_factory[layer+1];
                        if(ny[layer] != pooling_factory[layer+1]*ny[layer+1])
                        {
                            std::cout << "pooling layer ny: " << layer+1 << " should be: " << dy << std::endl;
                            exit(1);
                        }
                        if(n_nodes[layer] % (pooling_factorx[layer+1]*pooling_factory[layer+1]) != 0)
                        {
                            std::cout << "pooling layer n_nodes: " << layer << " should be divisible by: " << (pooling_factorx[layer+1]*pooling_factory[layer+1]) << std::endl;
                            exit(1);
                        }
                        if(n_nodes[layer] != pooling_factorx[layer+1]*pooling_factory[layer+1]*n_nodes[layer+1])
                        {
                            std::cout << "pooling layer n_nodes: " << layer+1 << " should be: " << n_nodes[layer]/(pooling_factorx[layer+1]*pooling_factory[layer+1]) << std::endl;
                            exit(1);
                        }
                        activation_values [layer] = new T[M*nx[layer]*ny[layer]];
                        break;
                    }
                default :
                    {
                        std::cout << "6. Undefined Layer Type. " << std::endl;
                        exit(1);
                        break;
                    }
            }
        }

        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            switch(n_layer_type[layer])
            {
                case RELU_LAYER :
                    {
                        deltas[layer] = new T[n_nodes[layer]];
                        break;
                    }
                case FULLY_CONNECTED_LAYER :
                    {
                        deltas[layer] = new T[n_nodes[layer]];
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        long M = n_features[layer];
                        deltas[layer] = new T[M*nx[layer]*nx[layer]];
                        break;
                    }
                case MAX_POOLING_LAYER :
                case MEAN_POOLING_LAYER :
                    {
                        long M = n_features[layer];
                        deltas[layer] = new T[M*nx[layer]*nx[layer]];
                        break;
                    }
                default :
                    {
                        std::cout << "7. Undefined Layer Type. " << std::endl;
                        exit(1);
                        break;
                    }
            }
        }

        for(long layer = 0;layer < n_layers;layer++)
        {
            switch(n_layer_type[layer])
            {
                case FULLY_CONNECTED_LAYER :
                    {
                        weights_neuron[layer] = new T*[n_nodes[layer+1]];
                        for(long i=0;i<n_nodes[layer+1];i++)
                        {
                            weights_neuron[layer][i] = new T[n_nodes[layer]];
                            for(long j=0;j<n_nodes[layer];j++)
                            {
                                weights_neuron[layer][i][j] = 1.0e-1 * (-1.0 + 2.0 * ((rand()%10000)/10000.0));
                            }
                        }
                        weights_bias[layer] = new T[n_nodes[layer+1]];
                        for(long i=0;i<n_nodes[layer+1];i++)
                        {
                            weights_bias[layer][i] = 1.0e-1 * (-1.0 + 2.0 * ((rand()%10000)/10000.0));
                        }
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        long M = n_features[layer];
                        long N = n_features[layer+1];
                        long wx = (kx[layer]/2);
                        long wy = (ky[layer]/2);
                        long dx = nx[layer] - wx*2;
                        long dy = ny[layer] - wy*2;
                        //if(n_nodes[layer+1] != ky[layer]*N)
                        //{
                        //    std::cout << "convolutional weight matrix mismatch, " << std::endl;
                        //    std::cout << "n_nodes " << layer+1 << " is " << n_nodes[layer+1] << " should be " << ky[layer]*N << std::endl;
                        //    exit(1);
                        //}
                        //if(n_nodes[layer] != kx[layer]*M)
                        //{
                        //    std::cout << "convolutional weight matrix mismatch, " << std::endl;
                        //    std::cout << "n_nodes " << layer << " is " << n_nodes[layer] << " should be " << kx[layer]*M << std::endl;
                        //    exit(1);
                        //}
                        weights_neuron[layer] = new T*[ky[layer]*N];
                        for(long i=0;i<ky[layer]*N;i++)
                        {
                            weights_neuron[layer][i] = new T[kx[layer]*M];
                            for(long j=0;j<kx[layer]*M;j++)
                            {
                                weights_neuron[layer][i][j] = 0;
                            }
                        }
                        {
                            for(long m=0,i=0;m<M;m++)
                            {
                                {
                                    for(long n=0;n<N;n++)
                                    {
                                        for(long fy=-wy,ty=0;fy<=wy;fy++,ty++)
                                        for(long fx=-wx,tx=0;fx<=wx;fx++,tx++)
                                        {
                                            weights_neuron[layer][ky[layer]*n+ty][kx[layer]*m+tx]
                                                = (fx==0&&fy==0)?1:0.5*(-1+2*((rand()%10000)/10000.0));
                                        }
                                    }
                                }
                            }
                        }
                        weights_bias[layer] = NULL;
                        break;
                    }
                case RELU_LAYER :
                case MAX_POOLING_LAYER :
                case MEAN_POOLING_LAYER :
                    {
                        weights_neuron[layer] = NULL;
                        weights_bias[layer] = NULL;
                        break;
                    }
                default :
                    {
                        std::cout << "8. Undefined Layer Type. " << std::endl;
                        exit(1);
                        break;
                    }
            }
        }

    }

    T * model(long n_elements,long n_labels,T * variables)
    {
        T * labels = new T[n_labels];
        // initialize input activations
        for(long i=0;i<n_nodes[0];i++)
        {
            activation_values1[0][i] = variables[i];
        }
        // forward propagation
        for(long layer = 0; layer < n_layers; layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                T sum = weights_bias[layer][i];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    sum += activation_values1[layer][j] * weights_neuron[layer][i][j];
                }
                activation_values1[layer+1][i] = sigmoid(sum,0);// <- zero is important here!!!!
            }
        }
        long last_layer = n_nodes.size()-2;
        for(long i=0;i<n_labels;i++)
        {
            labels[i] = activation_values1[last_layer][i];
        }
        return labels;
    }

    T * model2(long n_elements,long n_labels,T * variables)
    {
        T * labels = new T[n_labels];
        // initialize input activations
        for(long i=0;i<n_nodes[0];i++)
        {
            activation_values3[0][i] = variables[i];
        }
        // forward propagation
        for(long layer = 0; layer < n_layers; layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                T sum = weights_bias[layer][i];
                for(long j=0;j<n_nodes[layer];j++)
                {
                    sum += activation_values3[layer][j] * weights_neuron[layer][i][j];
                }
                activation_values3[layer+1][i] = sigmoid(sum,0);// <- zero is important here!!!!
            }
        }
        long last_layer = n_nodes.size()-2;
        for(long i=0;i<n_labels;i++)
        {
            labels[i] = activation_values3[last_layer][i];
        }
        return labels;
    }

    T verify( long n_test_elements
            , long n_variables
            , T * test_variables
            , long n_labels
            , T * test_labels
            )
    {
        T err = 0;

        T * labels = new T[n_labels];

        for(int e=0;e<n_test_elements;e++)
        {

          // initialize input activations
          for(long i=0;i<n_variables;i++)
          {
              activation_values2[0][i] = test_variables[e*n_variables+i];
          }
          // forward propagation
          for(long layer = 0; layer < n_layers; layer++)
          {
              for(long i=0;i<n_nodes[layer+1];i++)
              {
                  T sum = weights_bias[layer][i];
                  for(long j=0;j<n_nodes[layer];j++)
                  {
                      sum += activation_values2[layer][j] * weights_neuron[layer][i][j];
                  }
                  activation_values2[layer+1][i] = sigmoid(sum,0);// <- zero is important here!!!!
              }
          }
          long last_layer = n_nodes.size()-2;
          for(long i=0;i<n_labels;i++)
          {
            {
              err += fabs(test_labels[e*n_labels+i] - activation_values2[last_layer][i]);
            }
          }

        }

        delete [] labels;

        return err/n_test_elements;

    }

    int get_sigmoid()
    {
        return sigmoid_type;
    }

    void train ( int p_sigmoid_type
               , T p_epsilon
               , long n_iterations
               , long n_elements
               , long n_variables
               , long n_labels
               , T * variables
               , T * labels
               , bool enable_quasi = false
               , long n_test_elements = 0
               , T * test_variables = NULL
               , T * test_labels = NULL
               , quasi_newton_info<T> * q_newton = NULL
               )
    {
        sigmoid_type = p_sigmoid_type;
        epsilon = p_epsilon;
        if(n_variables != n_nodes[0])
        {
            std::cout << "Error: num variables doesn't match." << std::endl;
            exit(0);
        }
        quasi_newton = NULL;
        if(enable_quasi)
        {
            if(q_newton == NULL)
            {
                quasi_newton = new quasi_newton_info<T>();
                quasi_newton->alpha                 = alpha;
                quasi_newton->n_nodes               = n_nodes;
                quasi_newton->n_layers              = n_layers;
                quasi_newton->weights_neuron        = weights_neuron;
                quasi_newton->weights_bias          = weights_bias;
                quasi_newton->quasi_newton_update   = true;
                quasi_newton->init_QuasiNewton();
            }
            else
            {
                quasi_newton = q_newton;
            }
        }
        ierror = 1e10;
        bool init = true;
        perror = 1e10;
        T min_final_error = 1e10;
        long live_count = 0;

        long n_threads = boost::thread::hardware_concurrency();
        std::vector<std::vector<long> > vrtx(n_threads);
        for(long i=0;i<n_elements;i++)
        {
          if(labels[i] > 1e-10)
          {
            vrtx[i%vrtx.size()].push_back(i);
            live_count++;
          }
        }
        std::vector<cnn_training_info<T>*> g;
        for(long i=0;i<n_threads;i++)
        {
          g.push_back(new cnn_training_info<T>());
        }
        for(long thread=0;thread<g.size();thread++)
        {
          g[thread]->quasi_newton       = quasi_newton;
          g[thread]->n_nodes            = n_nodes;
          g[thread]->n_layer_type       = n_layer_type;
          g[thread]->n_activation_type  = n_activation_type;
          g[thread]->n_features         = n_features;
          g[thread]->kx                 = kx;
          g[thread]->ky                 = ky;
          g[thread]->nx                 = nx;
          g[thread]->ny                 = ny;
          g[thread]->pooling_factorx    = pooling_factorx;
          g[thread]->pooling_factory    = pooling_factory;
          g[thread]->n_elements         = n_elements;
          g[thread]->n_variables        = n_variables;
          g[thread]->n_labels           = n_labels;
          g[thread]->n_layers           = n_layers;
          g[thread]->weights_neuron     = weights_neuron;
          g[thread]->weights_bias       = weights_bias;
          g[thread]->epsilon            = epsilon;
          g[thread]->type               = get_sigmoid();
          g[thread]->init(alpha);
        }

        for(long iter = 0; iter < n_iterations || continue_training; iter++)
        {
            //std::cout << "iter=" << iter << std::endl;
            T error = 0;
            T index = 0;

            //////////////////////////////////////////////////////////////////////////////////
            //                                                                              //
            //          Multi-threaded block                                                //
            //                                                                              //
            //////////////////////////////////////////////////////////////////////////////////
            std::vector<boost::thread*> threads;
            if(quasi_newton!=NULL)
            {
              quasi_newton->init_gradient();
            }
            for(long thread=0;thread<vrtx.size();thread++)
            {
              g[thread]->reset();
              //if(iter%100==0)
              {
                threads.push_back ( new boost::thread ( cnn_training_worker<T>
                                                      , vrtx.size()
                                                      , iter,g[thread]
                                                      , vrtx[thread]
                                                      , variables
                                                      , labels
                                                      )
                                  );
              }
              //else
              //{
              //  threads.push_back ( new boost::thread ( cnn_training_worker_svrg<T>
              //                                        , vrtx.size()
              //                                        , iter,g[thread]
              //                                        , vrtx[thread]
              //                                        , variables
              //                                        , labels
              //                                        )
              //                    );
              //}
            }
            //usleep(10000);
            for(long thread=0;thread<vrtx.size();thread++)
            {
              threads[thread]->join();
              g[thread]->update_gradient();
              delete threads[thread];
            }
            if(quasi_newton!=NULL)
            {
              quasi_newton->update_QuasiNewton();
              quasi_newton->SR1_update();
            }
            for(long thread=0;thread<vrtx.size();thread++)
            {
              g[thread]->globalUpdate();
              error += g[thread]->partial_error;
              index += g[thread]->smallest_index;
            }
            threads.clear();
            std::cout << iter << "\tquasi newton=" << ((quasi_newton!=NULL)?(quasi_newton->quasi_newton_update?"true":"false"):"NULL") << "\ttype=" << sigmoid_type << "\tepsilon=" << epsilon << "\talpha=" << alpha << '\t' << "error=" << error << "\tdiff=" << (error-perror) << "\t\%error=" << 100*error/live_count << "\ttest\%error=" << 100*final_error << "\tindex=" << index/n_elements << std::endl;
            perror = error;
            errs.push_back(error/n_elements);
            test_errs.push_back(final_error);
            if(init)
            {
                ierror = error;
                init = false;
            }

            if(stop_training)
            {
                stop_training = false;
                break;
            }

        }
        for(long thread=0;thread<vrtx.size();thread++)
        {
          g[thread]->destroy();
          delete g[thread];
        }
        vrtx.clear();
        g.clear();

    }

};


#endif

