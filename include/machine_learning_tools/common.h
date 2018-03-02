#ifndef COMMON_H
#define COMMON_H



template<typename T>
void set_val(T * dat,T val,long size)
{
  for(long i=0;i<size;i++)
  {
    dat[i] = val;
  }
}

template<typename T>
T norm(T * dat,long size)
{
  T ret = 0;
  for(long i=0;i<size;i++)
  {
    ret += dat[i]*dat[i];
  }
  return sqrt(ret);
}

template<typename T>
void zero(T * dat,long size)
{
  for(long i=0;i<size;i++)
  {
    dat[i] = 0;
  }
}

template<typename T>
void constant(T * dat,T val,long size)
{
  for(long i=0;i<size;i++)
  {
    dat[i] = (-1+2*((rand()%10000)/10000.0f))*val;
  }
}

template<typename T>
void add(T * A, T * dA, T epsilon, long size)
{
  for(long i=0;i<size;i++)
  {
    A[i] += epsilon * dA[i];
  }
}

template<typename T>
T * Gabor ( long nx
          , long ny 
          , T lambda
          , T theta
          , T phi
          , T sigma
          , T gamma
          )
{
    T * out = new T[nx*ny];
    long wx=nx/2;
    long wy=ny/2;
    for(long x=-wx,k=0;x<=wx;x++)
    for(long y=-wy;y<=wy;y++,k++)
    {
        double X = x*cos(theta) + y*sin(theta);
        double Y = y*cos(theta) - x*sin(theta);
        out[k] = exp(-(X*X+gamma*gamma*Y*Y)/(2*sigma*sigma)) * cos(2*M_PI*X/lambda + phi);
    }
    return out;
}

#endif

