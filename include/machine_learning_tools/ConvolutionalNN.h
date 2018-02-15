#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

template < typename T >
T sigmoid1(T x)
{
    return 1.0f / (1.0f + exp(-x));
}

template < typename T >
T dsigmoid1(T x)
{
    return (1.0f - x)*x;
}

template < typename T >
T sigmoid3(T x)
{
    return atan(x);
}

template < typename T >
T dsigmoid3(T x)
{
    return 1.00/(1+x*x);
}

template < typename T >
T sigmoid2(T x)
{
    return log(1+exp(1.00*x));
}

template < typename T >
T dsigmoid2(T x)
{
    return 1.00/(1+exp(-1.00*x));
}

template < typename T >
T sigmoid(T x,int type)
{
    switch(type)
    {
        case 0:
            return sigmoid1(x);
        case 1:
            return sigmoid2(x);
        case 2:
            return sigmoid3(x);
    }
}

template < typename T >
T dsigmoid(T x,int type)
{
    switch(type)
    {
        case 0:
            return dsigmoid1(x);
        case 1:
            return dsigmoid2(x);
        case 2:
            return dsigmoid3(x);
    }
}

template < typename T >
T max(T a,T b)
{
    return (a>b)?a:b;
}

int maxi(int a,int b)
{
    return (a>b)?a:b;
}

template < typename T >
void apply_worker(std::vector<long> const & indices,long size,T * y,T * W,T * x)
{
  for(long k=0;k<indices.size();k++)
  {
    long i = indices[k];
    y[i] = 0;
    for(long j=0;j<size;j++)
    {
      y[i] += W[i*size+j]*x[j];
    }
  }
}

template < typename T >
void outer_product_worker(std::vector<long> const & indices,long size,T * H,T * A,T * B,T fact)
{
  for(long k=0;k<indices.size();k++)
  {
    long i = indices[k];
    for(long j=0;j<size;j++)
    {
      H[i*size+j] += A[i] * B[j] * fact;
    }
  }
}

template<typename T>
struct quasi_newton_info
{
    quasi_newton_info()
    {
        quasi_newton_update = false;
    }

    long get_size()
    {
        long size = 0;
        for(long layer = 0;layer < n_layers;layer++)
        {
            size += n_nodes[layer+1]*n_nodes[layer] + n_nodes[layer+1];
        }
        return size;
    }

    void init_gradient ()
    {
        long size = get_size();
        for(long layer = 0,k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    grad_tmp[k] = 0;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                grad_tmp[k] = 0;
            }
        }
    }

    void copy (T * src,T * dst,long size)
    {
        for(long k=0;k<size;k++)
        {
            dst[k] = src[k];
        }
    }

    void copy_avg (T * src,T * dst,T alph,long size)
    {
        for(long k=0;k<size;k++)
        {
            dst[k] += (src[k]-dst[k])*alph;
        }
    }

    bool quasi_newton_update;
    long n_layers;
    T *** weights_neuron;
    T **  weights_bias;
    std::vector<long> n_nodes;
    T * grad_tmp;
    T * grad_1;
    T * grad_2;
    T * Y;
    T * dX;
    T * B;
    T * H;
    T alpha;

    void init_QuasiNewton()
    {
        long size = get_size();
        grad_tmp = new T[size];
        init_gradient();
        grad_1 = new T[size];
        grad_2 = new T[size];
        copy(grad_tmp,grad_1,size);
        copy(grad_tmp,grad_2,size);
        B = new T[size*size];
        T * B_tmp = init_B();
        copy(B_tmp,B,size*size);
        delete [] B_tmp;
        H = new T[size*size];
        T * H_tmp = init_H();
        copy(H_tmp,H,size*size);
        delete [] H_tmp;
        dX = new T[size*size];
        T * dX_tmp = get_dx();
        copy(dX_tmp,dX,size);
        delete [] dX_tmp;
        Y = new T[size*size];
    }

    T * init_B()
    {
        long size = get_size();
        T * B = new T[size*size];
        for(long t=0;t<size*size;t++)
        {
            B[t] = 0;
        }
        for(long layer = 0, k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    B[k*size+k] = weights_neuron[layer][i][j];
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                B[k*size+k] = weights_bias[layer][i];
            }
        }
        return B;
    }

    T * init_H()
    {
        long size = get_size();
        T * H = new T[size*size];
        for(long t=0;t<size*size;t++)
        {
            H[t] = 0;
        }
        for(long layer = 0, k = 0;layer < n_layers;layer++)
        {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
                for(long j=0;j<n_nodes[layer];j++,k++)
                {
                    H[k*size+k] = -1;
                }
            }
            for(long i=0;i<n_nodes[layer+1];i++,k++)
            {
                H[k*size+k] = -1;
            }
        }
        return H;
    }

    void update_QuasiNewton()
    {
        long size = get_size();
        copy_avg(grad_2,grad_1,0.1,size);
        copy(grad_tmp,grad_2,size);
        T * Y_tmp = get_y();
        copy(Y_tmp,Y,size);
        delete [] Y_tmp;
        T * dX_tmp = get_dx();
        copy(dX_tmp,dX,size);
        delete [] dX_tmp;
    }

    T * get_y ()
    {
        long size = get_size();
        T * y = new T[size];
        //T y_m = 0;
        for(long k=0;k<size;k++)
        {
            y[k] = grad_2[k] - grad_1[k];
            //y_m = max(y_m,fabs(y[k]));
        }
        return y;
    }

    T * get_dx ()
    {
        long size = get_size();
        T * dx = apply(H,grad_1);
        for(long k=0;k<size;k++)
        {
            dx[k] *= -alpha;
        }
        return dx;
    }

    T * get_outer_product(T * a,T * b)
    {
        long size = get_size();
        long prod_size = size*size;
        T * prod = new T[prod_size];
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            prod[k] = a[i]*b[j];
          }
        }
        return prod;
    }

    T get_inner_product(T * a,T * b)
    {
        T ret = 0;
        long size = get_size();
        for(long i=0;i<size;i++)
        {
            ret += a[i]*b[i];
        }
        T eps = 1e-2;
        if(ret<0)
        {
            ret -= eps;
        }
        else
        {
            ret += eps;
        }
        return ret;
    }

    T * apply(T * W, T * x)
    {
        long size = get_size();
        T * y = new T[size];
        std::vector<boost::thread * > threads;
        long num_cpu = boost::thread::hardware_concurrency();
        std::vector<std::vector<long> > indices(num_cpu);
        for(long i=0;i<size;i++)
        {
          indices[i%num_cpu].push_back(i);
        }
        for(long i=0;i<num_cpu;i++)
        {
          threads.push_back(new boost::thread(apply_worker<T>,indices[i],size,&y[0],&W[0],&x[0]));
        }
        for(long i=0;i<threads.size();i++)
        {
          threads[i]->join();
          delete threads[i];
        }
        return y;
    }

    T * apply_t(T * x, T * W)
    {
        long size = get_size();
        T * y = new T[size];
        for(long i=0,k=0;i<size;i++)
        {
          y[i] = 0;
          for(long j=0;j<size;j++,k++)
          {
            y[i] += W[size*j+i]*x[j];
          }
        }
        return y;
    }

    T limit(T x,T eps)
    {
        if(x>0)
        {
            if(x>eps)return eps;
        }
        else
        {
            if(x<-eps)return -eps;
        }
        return x;
    }

    // SR1
    void SR1_update()
    {
        long size = get_size();
        T * dx_Hy = apply(H,Y);
        for(long i=0;i<size;i++)
        {
          dx_Hy[i] = dX[i] - dx_Hy[i];
        }
        T inner = 1.0 / (get_inner_product(dx_Hy,Y));
        std::vector<boost::thread * > threads;
        long num_cpu = boost::thread::hardware_concurrency();
        std::vector<std::vector<long> > indices(num_cpu);
        for(long i=0;i<size;i++)
        {
          indices[i%num_cpu].push_back(i);
        }
        for(long i=0;i<num_cpu;i++)
        {
          threads.push_back(new boost::thread(outer_product_worker<T>,indices[i],size,&H[0],&dx_Hy[0],&dx_Hy[0],inner));
        }
        for(long i=0;i<threads.size();i++)
        {
          threads[i]->join();
          delete threads[i];
        }
        delete [] dx_Hy;
    }

    // Broyden
    void Broyden_update()
    {
        long size = get_size();
        T * dx_Hy = apply(H,Y);
        for(long i=0;i<size;i++)
        {
          dx_Hy[i] = dX[i] - dx_Hy[i];
        }
        T * xH = apply_t(dX,H);
        T * outer = get_outer_product(dx_Hy,xH);
        T inner = 1.0 / (get_inner_product(xH,Y));
        for(long i=0;i<size*size;i++)
        {
          H[i] += outer[i] * inner;
        }
        delete [] dx_Hy;
        delete [] xH;
        delete [] outer;
    }

    // DFP
    void DFP_update()
    {
        long size = get_size();
        T * Hy = apply(H,Y);
        T * outer_2 = get_outer_product(Hy,Hy);
        T inner_2 = -1.0 / (get_inner_product(Hy,Y));
        T * outer_1 = get_outer_product(dX,dX);
        T inner_1 = 1.0 / (get_inner_product(dX,Y));
        for(long i=0;i<size*size;i++)
        {
          H[i] += outer_1[i] * inner_1 + outer_2[i] * inner_2;
        }
        delete [] outer_2;
        delete [] outer_1;
        delete [] Hy;
    }

    T * apply_M(T * A, T * B)
    {
        long size = get_size();
        T * C = new T[size*size];
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            C[k] = 0;
            for(long t=0;t<size;t++)
            {
              C[k] += A[i*size+t]*B[t*size+j];
            }
          }
        }
        return C;
    }

    // BFGS
    void BFGS_update()
    {
        long size = get_size();
        T inner = 1.0 / (get_inner_product(Y,dX));
        T * outer_xx = get_outer_product(dX,dX);
        T * outer_xy = get_outer_product(dX,Y);
        T * outer_yx = get_outer_product(Y,dX);
        for(long i=0,k=0;i<size;i++)
        {
          for(long j=0;j<size;j++,k++)
          {
            if(i==j)
            {
              outer_xy[k] = 1-outer_xy[k]*inner;
              outer_yx[k] = 1-outer_yx[k]*inner;
            }
            else
            {
              outer_xy[k] = -outer_xy[k]*inner;
              outer_yx[k] = -outer_yx[k]*inner;
            }
            outer_xx[k] = outer_xx[k]*inner;
          }
        }
        T * F = apply_M(outer_xy,H);
        T * G = apply_M(F,outer_yx);
        for(long i=0;i<size*size;i++)
        {
          H[i] = G[i] + outer_xx[i];
        }
        delete [] F;
        delete [] G;
        delete [] outer_xx;
        delete [] outer_xy;
        delete [] outer_yx;
    }

};

template<typename T>
struct training_info
{

    quasi_newton_info<T> * quasi_newton;

    std::vector<long> n_nodes;
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

    training_info()
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

template<typename T>
T min(T a,T b)
{
    return (a<b)?a:b;
}

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

enum LayerType
{
    FULLY_CONNECTED = 1 // => y_l := W_l * y_l-1                <= dEdy_l-1 := W_l * dEdy_l                 dEdW_l-1 := dEdy_l * y_l
  , RELU            = 2 // => y_l := max(0,y_l-1)               <= dEdy_l-1 := dEdy_l
  , POOLING         = 3 // => y_l := max(y_k)                   <= (l==k)?dEdy_l-1=dEdy_l:dEdy_l-1=0        
  , CONVOLUTIONAL   = 4 // => y_l := W_l * y_l-1                <= dEdy_l-1 := W_l * dEdy_l                 dEdW_l-1 := dEdy_l *_y_l
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
}

template<typename T>
void training_worker(long n_threads,long iter,training_info<T> * g,std::vector<long> const & vrtx,T * variables,T * labels)
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
            switch ( LayerType )
            {
                case FULLY_CONNECTED :
                    {
                        for(long layer = 0; layer < g->n_layers; layer++)
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
                        }
                        break;
                    }
                default :
                    {
                        std::cout << "Layer type not defined." << std::endl;
                        exit(1);
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
                    switch ( LayerType )
                    {
                        case FULLY_CONNECTED :
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
                        default :
                            {
                                std::cout << "Layer type not defined." << std::endl;
                                exit(1);
                            }
                    }
                }
                    
                
                // biases
                switch ( LayerType )
                {
                    case FULLY_CONNECTED :
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
                switch ( LayerType )
                {
                    case FULLY_CONNECTED :
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
void training_worker_svrg(long n_threads,long iter,training_info<T> * g,std::vector<long> const & vrtx,T * variables,T * labels)
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
struct Perceptron
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

    Perceptron(std::vector<long> p_nodes)
    {

        quasi_newton = NULL;

        continue_training = false;
        stop_training = false;

        sigmoid_type = 0;
        alpha = 0.1;

        ierror = 1e10;
        perror = 1e10;

        n_nodes = p_nodes;
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
                quasi_newton->alpha = alpha;
                quasi_newton->n_nodes = n_nodes;
                quasi_newton->n_layers = n_layers;
                quasi_newton->weights_neuron = weights_neuron;
                quasi_newton->weights_bias = weights_bias;
                quasi_newton->init_QuasiNewton();
                quasi_newton->quasi_newton_update = true;
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
        std::vector<training_info<T>*> g;
        for(long i=0;i<boost::thread::hardware_concurrency();i++)
        {
          g.push_back(new training_info<T>());
        }
        for(long thread=0;thread<g.size();thread++)
        {
          g[thread]->quasi_newton = quasi_newton;
          g[thread]->n_nodes = n_nodes;
          g[thread]->n_elements = n_elements;
          g[thread]->n_variables = n_variables;
          g[thread]->n_labels = n_labels;
          g[thread]->n_layers = n_layers;
          g[thread]->weights_neuron = weights_neuron;
          g[thread]->weights_bias = weights_bias;
          g[thread]->epsilon = epsilon;
          g[thread]->type = get_sigmoid();
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
              if(iter%100==0)
              {
                threads.push_back ( new boost::thread ( training_worker<T>
                                                      , vrtx.size()
                                                      , iter,g[thread]
                                                      , vrtx[thread]
                                                      , variables
                                                      , labels
                                                      )
                                  );
              }
              else
              {
                threads.push_back ( new boost::thread ( training_worker_svrg<T>
                                                      , vrtx.size()
                                                      , iter,g[thread]
                                                      , vrtx[thread]
                                                      , variables
                                                      , labels
                                                      )
                                  );
              }
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

