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
    FULLY_CONNECTED_LAYER = 1 // => y_l := W_l * y_l-1                <= dEdy_l-1 := W_l * dEdy_l                 dEdW_l-1 := dEdy_l * y_l
  , RELU_LAYER            = 2 // => y_l := max(0,y_l-1)               <= dEdy_l-1 := dEdy_l
  , POOLING_LAYER         = 3 // => y_l := max(y_k)                   <= (l==k)?dEdy_l-1=dEdy_l:dEdy_l-1=0        
  , CONVOLUTIONAL_LAYER   = 4 // => y_l := W_l * y_l-1                <= dEdy_l-1 := W_l * dEdy_l                 dEdW_l-1 := dEdy_l *_y_l
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
            activation_values [layer] = new T[n_nodes[layer]];
        }
        deltas = new T*[n_nodes.size()];
        for(long layer = 0;layer < n_nodes.size();layer++)
        {
            deltas[layer] = new T[n_nodes[layer]];
        }
        partial_weights_neuron = new T**[n_layers];
        partial_weights_bias = new T*[n_layers];
        mu_partial_weights_neuron = new T**[n_layers];
        mu_partial_weights_bias = new T*[n_layers];
        mu_weights_neuron = new T**[n_layers];
        mu_weights_bias = new T*[n_layers];
        for(long layer = 0;layer < n_layers;layer++)
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
        }
    }

    void reset()
    {
        type = 0;
        smallest_index = 0;
        partial_error = 0;
        for(long layer = 0;layer < n_layers;layer++)
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
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                delete [] partial_weights_neuron[layer][i];
                delete [] mu_partial_weights_neuron[layer][i];
                delete [] mu_weights_neuron[layer][i];
            }
            delete [] partial_weights_neuron[layer];
            delete [] mu_partial_weights_neuron[layer];
            delete [] mu_weights_neuron[layer];
        }
        delete [] partial_weights_neuron;
        delete [] mu_partial_weights_neuron;
        delete [] mu_weights_neuron;
        for(long layer = 0;layer < n_layers;layer++)
        {
            delete [] partial_weights_bias[layer];
            delete [] mu_partial_weights_bias[layer];
            delete [] mu_weights_bias[layer];
        }
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
        if(quasi_newton != NULL && quasi_newton->quasi_newton_update)
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                for(long i=0;i<n_nodes[layer+1];i++)
                {
                    for(long j=0;j<n_nodes[layer];j++,k++)
                    {
                        weights_neuron[layer][i][j] += quasi_newton->dX[k];
                    }
                }
                for(long i=0;i<n_nodes[layer+1];i++,k++)
                {
                    weights_bias[layer][i] += quasi_newton->dX[k];
                }
            }
        }
        else if(quasi_newton != NULL)
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                for(long i=0;i<n_nodes[layer+1];i++)
                {
                    for(long j=0;j<n_nodes[layer];j++,k++)
                    {
                        weights_neuron[layer][i][j] += epsilon * quasi_newton->grad_tmp[k];
                    }
                }
                for(long i=0;i<n_nodes[layer+1];i++,k++)
                {
                    weights_bias[layer][i] += epsilon * quasi_newton->grad_tmp[k];
                }
            }
        }
        else
        {
            for(long layer = 0,k = 0;layer < n_layers;layer++)
            {
                for(long i=0;i<n_nodes[layer+1];i++)
                {
                    for(long j=0;j<n_nodes[layer];j++,k++)
                    {
                        weights_neuron[layer][i][j] += epsilon * partial_weights_neuron[layer][i][j];
                    }
                }
                for(long i=0;i<n_nodes[layer+1];i++,k++)
                {
                    weights_bias[layer][i] += epsilon * partial_weights_bias[layer][i];
                }
            }
        }
    }

};

/*
template<typename T>
void ForwardReLU()
{
    for(long i=0;i<size;i++)
    {
        y[l][i] = max(0,y[l-1][i]);
    }
}

template<typename T>
void ReverseReLU()
{
    for(long i=0;i<size;i++)
    {
        if(y[l][i] < 0)
        {
            y[l][i] = max(0,y[l-1][i]);
        }
        else
        {
            dEdy[l-1][i] = dEdy[l][i];
        }
    }
}

template<typename T>
void ForwardFullyConnected()
{
    for(long i=0;i<size_1;i++)
    {
        y[l][i] = 0;
        for(long j=0;j<size_2;j++)
        {
            y[l][i] += W[l][i][j] * y[l-1][j];
        }
    }
}

template<typename T>
void ReverseFullyConnectedDeltas()
{
    for(long j=0;j<size_2;j++)
    {
        dEdy[l-1][j] = 0;
        for(long i=0;i<size_1;i++)
        {
            dEdy[l-1][j] += W[l][j][i] * dEdy[l][i];
        }
    }
}

template<typename T>
void ReverseFullyConnectedUpdate()
{
    for(long i=0;i<size_1;i++)
    {
        dEdW[l-1][i] = 0;
        for(long j=0;j<size_2;j++)
        {
            dEdW[l-1][i] += dEdy[l][i] * y[l][j];
        }
    }
}

template<typename T>
void ForwardPooling()
{
    for(long n=0;n<N;n++)
    for(long X=0;X<nx;X+=k)
    for(long Y=0;Y<ny;Y+=k)
    {
        y[l][nx*ny*n+X+nx*Y] = -100000000;
    }
    long NX = nx/k;
    long NY = ny/k;
    for(long n=0;n<N;n++)
    for(long x=0,X=0;x<nx;x+=k,X++)
    for(long dx=0;dx<k;dx++)
    for(long y=0,Y=0;y<ny;y+=k,Y++)
    for(long dy=0;dy<k;dy++)
    {
        y[l][NX*NY*n+X+nx*Y] = max(y[l][NX*NY*n+X+nx*Y],y[l-1][nx*ny*n+x+dx+nx*(y+dy)]);
    }
}

template<typename T>
void ReversePooling()
{
    long NX = nx/k;
    long NY = ny/k;
    for(long n=0;n<N;n++)
    for(long x=0,X=0;x<nx;x+=k,X++)
    for(long dx=0;dx<k;dx++)
    for(long y=0,Y=0;y<ny;y+=k,Y++)
    for(long dy=0;dy<k;dy++)
    {
        if(y[l][NX*NY*n+X+nx*Y] == y[l-1][nx*ny*n+x+dx+nx*(y+dy)])
        {
            dEdy[l-1][nx*ny*n+x+dx+nx*(y+dy)] = dEdy[l][NX*NY*n+X+nx*Y];
        }
        else
        {
            dEdy[l-1][nx*ny*n+x+dx+nx*(y+dy)] = 0;
        }
    }
}

template<typename T>
void ForwardConvolutional()
{
    long KX = 2*k+1;
    long KY = 2*k+1;
    long NX = nx-KX;
    long NY = ny-KY;
    for(long n=0;n<N;n++)
    for(long m=0;m<M;m++)
    {
        for(long y=k,i=0;y+k<ny;y++)
        for(long x=k;x+k<nx;x++,i++)
        {
            y[l][NX*NY*n+i] = 0;
            for(long dy=-k,j=0,ky=0;dy<=k;y++,ky++)
            for(long dy=-k,kx=0;dx<=k;x++,j++,kx++)
            {
                y[l][NX*NY*n+i] += W[l][KX*n+kx][KY*m+ky] * y[l-1][nx*ny*m+x+dx+nx*(y+dy)];
            }
        }
    }
}

template<typename T>
void ReverseConvolutionalDeltas()
{
    long NX = nx-2*k-1;
    long NY = ny-2*k-1;
    for(long n=0;n<N;n++)
    for(long m=0;m<M;m++)
    {
        for(long i=0;i<nx*ny;i++)
        {
            dEdy[l-1][nx*ny*m+i] = 0;
        }
        for(long dy=-k,j=0,ky=0;dy<=k;y++,ky++)
        for(long dx=-k,kx=0;dx<=k;x++,j++,kx++)
        {
            for(long y=k,i=0;y+k<ny;y++)
            for(long x=k;x+k<nx;x++,i++)
            {
                dEdy[l-1][nx*ny*m+x+dx+nx*(y+dy)] += W[l][KX*n+kx][KY*m+ky] * dEdy[l][NX*NY*n+i];
            }
        }
    }
}

template<typename T>
void ReverseConvolutionalUpdate()
{
    long NX = nx-2*k-1;
    long NY = ny-2*k-1;
    for(long n=0;n<N;n++)
    for(long m=0;m<M;m++)
    {
        for(long i=0;i<size_1;i++)
        {
            dEdW[l][m][n][i] = 0;
            for(long j=0;j<size_2;j++)
            {
                dEdW[l-1][KX*n+kx][KY*m+ky] += dEdy[l][NX*NY*n+i] * y[l][nx*ny*m+dx+nx*(y+dy)];
            }
        }
    }
}
*/

template<typename T>
void cnn_training_worker(long n_threads,long iter,cnn_training_info<T> * g,std::vector<long> const & vrtx,T * variables,T * labels)
{
    {
        for(long n=0;n<vrtx.size();n++)
        {
            T avg_factor = 1.0 / (1.0 + n);
            // initialize input activations
            for(long i=0;i<g->n_nodes[0];i++)
            {
                g->activation_values[0][i] = variables[vrtx[n]*g->n_variables+i];
            }
            // forward propagation
            for(long layer = 0; layer < g->n_layers; layer++)
            {
                switch ( g->n_layer_type[layer] )
                {
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
                            long M = g->n_features[layer  ];
                            long N = g->n_features[layer+1];
                            long kx = g->kx[layer];
                            long ky = g->ky[layer];
                            long nx = g->nx[layer];
                            long ny = g->ny[layer];
                            long wx = (kx/2)*2;
                            long wy = (ky/2)*2;
                            long dx = nx - (kx/2)*2;
                            long dy = ny - (ky/2)*2;

                            for(long m=0,i=0;m<M;m++)
                            {
                                for(long oy=0;oy<dy;oy++)
                                for(long ox=0;ox<dx;ox++,i++)
                                {
                                    T sum = g->weights_bias[layer][i];
                                    for(long n=0;n<N;n++)
                                    {
                                        for(long iy=wy;iy+wy<ny;iy++)
                                        for(long ix=wx;ix+wx<nx;ix++)
                                        for(long fy=-wy,ty=0;fy<=wy;fy++,ty++)
                                        for(long fx=-wx,tx=0;fx<=wx;fx++,tx++)
                                        {
                                            // W * y
                                            sum += g->weights_neuron[layer][ky*n+ty][kx*m+tx] 
                                                 * g->activation_values[layer][(nx*ny)*m + nx*(iy+fy) + (ix+fx)];
                                        }
                                    }
                                    g->activation_values[layer+1][i] = sigmoid(sum,g->type);
                                }
                            }
                            break;
                        }
                    default :
                        {
                            std::cout << "Layer type not defined." << std::endl;
                            exit(1);
                        }
                }
            }
            long last_layer = g->n_nodes.size()-2;
            // initialize observed labels
            T partial_error = 0;
            long max_i = 0;
            T max_val = 0;
            for(long i=0;i<g->n_nodes[last_layer];i++)
            {
                g->deltas[last_layer+1][i] = labels[vrtx[n]*g->n_labels+i] - g->activation_values[last_layer][i];
                if(fabs(g->deltas[last_layer+1][i])>max_val)
                {
                    max_i = i;
                    max_val = fabs(g->deltas[last_layer+1][i]);
                }
            }
            for(long i=0;i<g->n_nodes[last_layer];i++)
            {
                if(i!=max_i)
                {
                    //g->deltas[last_layer+1][i] = 0;
                }
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
                    switch ( g->n_layer_type[layer] )
                    {
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
                                long kx = g->kx[layer];
                                long ky = g->ky[layer];
                                long nx = g->nx[layer];
                                long ny = g->ny[layer];
                                long wx = (kx/2)*2;
                                long wy = (ky/2)*2;
                                long dx = nx - (kx/2)*2;
                                long dy = ny - (ky/2)*2;

                                for(long m=0,i=0;m<M;m++)
                                {
                                    for(long iy=0;iy<ny;iy++)
                                    for(long ix=0;ix<nx;ix++,i++)
                                    {
                                        g->deltas[layer+1][i] = 0;
                                        for(long n=0;n<N;n++)
                                        {
                                            for(long oy=wy,vy=0;oy+wy<ny;oy++,vy++)
                                            for(long ox=wx,vx=0;ox+wx<nx;ox++,vx++)
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
                                            }
                                        }
                                        g->deltas[layer+1][i] *= dsigmoid(g->activation_values[layer+1][(nx*ny)*m + nx*iy + ix],g->type);
                                    }
                                }

                                break;
                            }
                        default :
                            {
                                std::cout << "Layer type not defined." << std::endl;
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
                    default :
                        {
                            std::cout << "Layer type not defined." << std::endl;
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
                            long M = g->n_features[layer+1];
                            long N = g->n_features[layer+2];
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
                                                g->partial_weights_neuron[layer][ky*n+ty][kx*m+tx] = 0;
                                            }
                                        }
                                    }

                                }
                            }
                            for(long ty=0,fy=-wy;ty<ky;ty++,fy++)
                            {
                                for(long tx=0,fx=-wx;tx<kx;tx++,fx++)
                                {
                                    for(long m=0,j=0;m<M;m++)
                                    {
                                        for(long iy=wy;iy+wy<ny;iy++)
                                        for(long ix=wx;ix+wx<nx;ix++,j++)
                                        {
                                            for(long n=0,i=0;n<N;n++)
                                            {
                                                for(long vy=0;vy<dy;vy++)
                                                for(long vx=0;vx<dx;vx++,i++)
                                                {
                                                        g->partial_weights_neuron[layer][ky*n+ty][kx*m+tx] += 
                                                            ( 
                                                                (
                                                                    // dEdy
                                                                    g->deltas[layer+1][(dx*dy)*n + dx*vy + vx] 
                                                                    // y
                                                                  * g->activation_values[layer][(nx*ny)*m + nx*(iy+fy) + (ix+fx)]
                                                                )
                                                              
                                                            - g->partial_weights_neuron[layer][i][j]
                                                            ) * avg_factor;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    default :
                        {
                            std::cout << "Layer type not defined." << std::endl;
                            exit(1);
                        }
                }
                    
                
            }
        }
        for(long layer = 0; layer < g->n_layers; layer++)
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

        }
    }

}

template<typename T>
void cnn_training_worker_svrg(long n_threads,long iter,cnn_training_info<T> * g,std::vector<long> const & vrtx,T * variables,T * labels)
{
    {
        long n = rand()%vrtx.size();
        for(long layer = 0; layer < g->n_layers; layer++)
        {
            // biases
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                g->partial_weights_bias[layer][i] = g->mu_partial_weights_bias[layer][i];
            }
            // neuron weights
            for(long i=0;i<g->n_nodes[layer+1];i++)
            {
                for(long j=0;j<g->n_nodes[layer];j++)
                {
                    g->partial_weights_neuron[layer][i][j] = g->mu_partial_weights_neuron[layer][i][j];
                }
            }
        }
        {
            // initialize input activations
            for(long i=0;i<g->n_nodes[0];i++)
            {
                g->activation_values[0][i] = variables[vrtx[n]*g->n_variables+i];
            }
            // forward propagation
            for(long layer = 0; layer < g->n_layers; layer++)
            {
                for(long i=0;i<g->n_nodes[layer+1];i++)
                {
                    T sum = g->mu_weights_bias[layer][i];
                    for(long j=0;j<g->n_nodes[layer];j++)
                    {
                        sum += g->activation_values[layer][j] * g->mu_weights_neuron[layer][i][j];
                    }
                    g->activation_values[layer+1][i] = sigmoid(sum,g->type);
                }
            }
            long last_layer = g->n_nodes.size()-2;
            // initialize observed labels
            T partial_error = 0;
            long max_i = 0;
            T max_val = 0;
            for(long i=0;i<g->n_nodes[last_layer];i++)
            {
                g->deltas[last_layer+1][i] = labels[vrtx[n]*g->n_labels+i] - g->activation_values[last_layer][i];
                if(fabs(g->deltas[last_layer+1][i])>max_val)
                {
                    max_i = i;
                    max_val = fabs(g->deltas[last_layer+1][i]);
                }
            }
            for(long i=0;i<g->n_nodes[last_layer];i++)
            {
                if(i!=max_i)
                {
                    //g->deltas[last_layer+1][i] = 0;
                }
                partial_error += fabs(g->deltas[last_layer+1][i]);
            }
            // back propagation
            for(long layer = g->n_layers-1; layer >= 0; layer--)
            {
                // back propagate deltas
                for(long i=0;i<g->n_nodes[layer+1];i++)
                {
                    g->deltas[layer+1][i] = 0;
                    for(long j=0;j<g->n_nodes[layer+2];j++)
                    {
                        if(layer+1==last_layer)
                        {
                            g->deltas[layer+1][i] += dsigmoid(g->activation_values[layer+1][i],g->type)*g->deltas[layer+2][j];
                        }
                        else
                        {
                            g->deltas[layer+1][i] += dsigmoid(g->activation_values[layer+1][i],g->type)*g->deltas[layer+2][j]*g->mu_weights_neuron[layer+1][j][i];
                        }
                    }
                }
                // biases
                for(long i=0;i<g->n_nodes[layer+1];i++)
                {
                    g->partial_weights_bias[layer][i] -= g->deltas[layer+1][i];
                }
                // neuron weights
                for(long i=0;i<g->n_nodes[layer+1];i++)
                {
                    for(long j=0;j<g->n_nodes[layer];j++)
                    {
                        g->partial_weights_neuron[layer][i][j] -= g->activation_values[layer][j] * g->deltas[layer+1][i];
                    }
                }
            }
        }
        {
            // initialize input activations
            for(long i=0;i<g->n_nodes[0];i++)
            {
                g->activation_values[0][i] = variables[vrtx[n]*g->n_variables+i];
            }
            // forward propagation
            for(long layer = 0; layer < g->n_layers; layer++)
            {
                for(long i=0;i<g->n_nodes[layer+1];i++)
                {
                    T sum = g->weights_bias[layer][i];
                    for(long j=0;j<g->n_nodes[layer];j++)
                    {
                        sum += g->activation_values[layer][j] * g->weights_neuron[layer][i][j];
                    }
                    g->activation_values[layer+1][i] = sigmoid(sum,g->type);
                }
            }
            long last_layer = g->n_nodes.size()-2;
            // initialize observed labels
            T partial_error = 0;
            long max_i = 0;
            T max_val = 0;
            for(long i=0;i<g->n_nodes[last_layer];i++)
            {
                g->deltas[last_layer+1][i] = labels[vrtx[n]*g->n_labels+i] - g->activation_values[last_layer][i];
                if(fabs(g->deltas[last_layer+1][i])>max_val)
                {
                    max_i = i;
                    max_val = fabs(g->deltas[last_layer+1][i]);
                }
            }
            for(long i=0;i<g->n_nodes[last_layer];i++)
            {
                if(i!=max_i)
                {
                    //g->deltas[last_layer+1][i] = 0;
                }
                partial_error += fabs(g->deltas[last_layer+1][i]);
            }
            // back propagation
            for(long layer = g->n_layers-1; layer >= 0; layer--)
            {
                // back propagate deltas
                for(long i=0;i<g->n_nodes[layer+1];i++)
                {
                    g->deltas[layer+1][i] = 0;
                    for(long j=0;j<g->n_nodes[layer+2];j++)
                    {
                        if(layer+1==last_layer)
                        {
                            g->deltas[layer+1][i] += dsigmoid(g->activation_values[layer+1][i],g->type)*g->deltas[layer+2][j];
                        }
                        else
                        {
                            g->deltas[layer+1][i] += dsigmoid(g->activation_values[layer+1][i],g->type)*g->deltas[layer+2][j]*g->weights_neuron[layer+1][j][i];
                        }
                    }
                }
                // biases
                for(long i=0;i<g->n_nodes[layer+1];i++)
                {
                    g->partial_weights_bias[layer][i] += g->deltas[layer+1][i];
                }
                // neuron weights
                for(long i=0;i<g->n_nodes[layer+1];i++)
                {
                    for(long j=0;j<g->n_nodes[layer];j++)
                    {
                        g->partial_weights_neuron[layer][i][j] += g->activation_values[layer][j] * g->deltas[layer+1][i];
                    }
                }
            }
        }
        for(long layer = 0; layer < g->n_layers; layer++)
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
        }

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
            activation_values [layer] = new T[n_nodes[layer]];
            activation_values1[layer] = new T[n_nodes[layer]];
            activation_values2[layer] = new T[n_nodes[layer]];
            activation_values3[layer] = new T[n_nodes[layer]];
            deltas[layer] = new T[n_nodes[layer]];
        }

        for(long layer = 0;layer < n_layers;layer++)
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

        std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
        for(long i=0;i<n_elements;i++)
        {
          if(labels[i] > 1e-10)
          {
            vrtx[i%vrtx.size()].push_back(i);
          }
        }
        std::vector<cnn_training_info<T>*> g;
        for(long i=0;i<boost::thread::hardware_concurrency();i++)
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
            usleep(10000);
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

            if(n_test_elements>0&&test_variables!=NULL&&test_labels!=NULL)
            {
                final_error = verify(n_test_elements,n_variables,test_variables,n_labels,test_labels);
            }
            static int cnt1 = 0;
            if(cnt1%100==0 && error > 1e-20)
            std::cout << iter << "\tquasi newton=" << ((quasi_newton!=NULL)?(quasi_newton->quasi_newton_update?"true":"false"):"NULL") << "\ttype=" << sigmoid_type << "\tepsilon=" << epsilon << "\talpha=" << alpha << '\t' << "error=" << error << "\tdiff=" << (error-perror) << "\t\%error=" << 100*error/n_elements << "\ttest\%error=" << 100*final_error << "\tindex=" << index/n_elements << std::endl;
            cnt1++;
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
        vrtx.clear();
        for(long thread=0;thread<vrtx.size();thread++)
        {
          g[thread]->destroy();
          delete g[thread];
        }
        g.clear();

    }

};


#endif

