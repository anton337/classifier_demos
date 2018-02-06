#ifndef VISUALIZE_SIGNAL_H
#define VISUALIZE_SIGNAL_H

#include "Perceptron.h"

Perceptron<double> * sig_perceptron = NULL;
double * sig_dat = NULL;
double * sig_prct = NULL;
long sig_elems = -1;
long sig_num_in = 0;
long sig_start_elem = 0;

template<typename T>
void set_sig_data ( Perceptron<T> * perceptron , T * dat , T * prct , long num , long start , long num_in )
{
    sig_perceptron = perceptron;
    sig_dat = dat;
    sig_prct = prct;
    sig_elems = num;
    sig_start_elem = start;
    sig_num_in = num_in;
}

template<typename T>
T * calculate_percent_change(T * dat, long num_elements)
{
    T * prct = new T[num_elements];
    prct[0] = 0;
    for(long i=1;i<num_elements;i++)
    {
        prct[i] = (dat[i] - dat[i-1]) / dat[i-1];
    }
    return prct;
}

template<typename T>
void reconstruct(T * dat, T * prct, T * reconst, long start_elem, long num_elements)
{
    for(long i=0;i<num_elements;i++)
    {
        if(i<=start_elem)
        {
            reconst[i] = dat[i];
        }
        else
        {
            reconst[i] = (1+prct[i])*reconst[i-1];
        }
    }
}

template<typename T>
void reconstruct_model(Perceptron<T> * perceptron, T * dat, T * prct, T * reconst, long num_in, long start_elem, long num_elements)
{
    for(long i=0;i<num_elements;i++)
    {
        if(i<=start_elem)
        {
            reconst[i] = dat[i];
        }
        else
        {
            T * in = new T[num_in];
            for(long j=i-num_in,k=0;j<i;j++,k++)
            {
                in[k] = (prct[j] - (-0.05))/(0.05 - (-0.05));
                in[k] = (in[k]<0)?0:in[k];
                in[k] = (in[k]>1)?1:in[k];
            }
            T * out = perceptron->model(num_in,1,in);
            T pred = out[0];
            pred *= (0.05 - (-0.05));
            pred -= 0.05;
            reconst[i] = (1+pred)*reconst[i-1];
        }
    }
}

template<typename T>
void visualize_signal ( long num_elems , T * dat )
{
    T max_dat = 0;
    for(long k=0;k<num_elems;k++)
    {
        if(max_dat<dat[k])max_dat=dat[k];
    }
    glColor3f(1,1,1);
    T dx = 2.0 / (num_elems-1);
    glBegin(GL_LINES);
    for(long k=0;k+1<num_elems;k++)
    {
        glVertex3f( -1 + k*dx
                  , dat[k] / max_dat
                  , 0
                  );
        glVertex3f( -1 + (k+1)*dx
                  , dat[k+1] / max_dat
                  , 0
                  );
    }
    glEnd();
}

template<typename T>
void visualize_reconstruction ( Perceptron<T> * perceptron , long num_elems , T * dat , T * prct , long num_in , long start_elem )
{
    T max_dat = 0;
    for(long k=0;k<num_elems;k++)
    {
        if(max_dat<dat[k])max_dat=dat[k];
    }
    T * reconstruction = new T[num_elems];
    reconstruct_model(perceptron,dat,prct,reconstruction,num_in,start_elem,num_elems);
    glColor3f(1,1,1);
    T dx = 2.0 / (num_elems-1);
    glBegin(GL_LINES);
    for(long k=0;k+1<num_elems;k++)
    {
        glVertex3f( -1 + k*dx
                  , reconstruction[k] / max_dat
                  , 0
                  );
        glVertex3f( -1 + (k+1)*dx
                  , reconstruction[k+1] / max_dat
                  , 0
                  );
    }
    glEnd();
}

#endif

