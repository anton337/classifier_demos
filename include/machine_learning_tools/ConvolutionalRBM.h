#ifndef CONVOLUTIONAL_RBM_H
#define CONVOLUTIONAL_RBM_H

#include <vector>
#include <stdlib.h>
#include <boost/thread.hpp>

#include "BoltzmannMachine.h"

#include "common.h"

template<typename T>
struct crbm_gradient_info
{
  long n;
  long v;
  long h;
  long nx;
  long ny;
  long dx;
  long dy;
  long kx;
  long ky;
  long M;
  long K;
  T * vis0;
  T * hid0;
  T * vis;
  T * hid;
  T * dW;
  T * dc;
  T * db;
  T partial_err;
  T * partial_dW;
  T * partial_dc;
  T * partial_db;
  void init()
  {
    partial_err = 0;
    partial_dW = new T[M*K*kx*ky];
    for(int i=0;i<M*K*kx*ky;i++)partial_dW[i]=0;
    partial_dc = new T[K*dx*dy];
    for(int i=0;i<K*dx*dy;i++)partial_dc[i]=0;
    partial_db = new T[M*nx*ny];
    for(int i=0;i<M*nx*ny;i++)partial_db[i]=0;
  }
  void destroy()
  {
    delete [] partial_dW;
    delete [] partial_dc;
    delete [] partial_db;
  }
  void globalUpdate()
  {
    for(int i=0;i<M*K*kx*ky;i++)
        dW[i] += partial_dW[i];
    for(int i=0;i<K*dx*dy;i++)
        dc[i] += partial_dc[i];
    for(int i=0;i<M*nx*ny;i++)
        db[i] += partial_db[i];
  }
};

template<typename T>
void crbm_gradient_worker(crbm_gradient_info<T> * g,std::vector<std::pair<long,long> > const & vrtx)
{
  
  long Ky = g->ky;
  long Kx = g->kx;
  long ny = g->ny;
  long nx = g->nx;
  long dy = g->dy;
  long dx = g->dx;
  long M = g->M;
  long K = g->K;
  long wy = Ky/2;
  long wx = Kx/2;
  long h = K*dx*dy;
  long v = M*nx*ny;
  T factor = 1.0f / (g->n*dx*dy);
  T factorb= 1.0f / (g->n*nx*ny);
  T factorv= 1.0f / (g->v);
  
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t].first;
    long m = vrtx[t].second;
    // dW       [M x K x kx x ky]
    // vis      [M x nx x ny]
    // hid      [K x dx x dy]
    //T sum = 0;
    for(long z=0;z<K;z++)
    {
      //std::cout << M << '\t' << K << std::endl;
        for(long oy=0;oy<dy;oy++)
        for(long ox=0;ox<dx;ox++)
        for(long ky=-wy,i=0;ky<=wy;ky++)
        for(long kx=-wx;kx<=wx;kx++,i++)
        {
          //long iy = oy+wy+wy-ky;
          //long ix = ox+wx+wx-kx;
          long iy = oy+wy+ky;
          long ix = ox+wx+kx;
          // flipped here
          g->partial_dW[m*K*Kx*Ky+z*Kx*Ky+i] -= factor * (g->vis0[m*nx*ny+k*v+iy*nx+ix]*g->hid0[z*dx*dy+k*h+oy*dx+ox] - g->vis[m*nx*ny+k*v+iy*nx+ix]*g->hid[z*dx*dy+k*h+oy*dx+ox]);
          //sum += fabs(g->partial_dW[m*K*Kx*Ky+z*Kx*Ky+i]);
        }
        //for(long ky=-wy,i=0;ky<=wy;ky++)
        //for(long kx=-wx;kx<=wx;kx++,i++)
        //{
        //  //g->partial_dW[z*Kx*Ky+i] /= sum;
        //}
    }
    //std::cout << "sum:" << sum << std::endl;

    //for(long z=0;z<K;z++)
    //for(long oy=0,j=0;oy<dy;oy++)
    //for(long ox=0;ox<dx;ox++,j++)
    //{
    //  g->partial_dc[z*dx*dy+j] -= factor * (g->hid0[z*dx*dy+k*h+j]*g->hid0[z*dx*dy+k*h+j] - g->hid[z*dx*dy+k*h+j]*g->hid[z*dx*dy+k*h+j]);
    //}

    //for(long m=0;m<M;m++)
    //for(long iy=0,i=0;iy<ny;iy++)
    //for(long ix=0;ix<nx;ix++,i++)
    //{
    //  g->partial_db[i] -= factorb * (g->vis0[m*nx*ny+k*v+i]*g->vis0[m*nx*ny+k*v+i] - g->vis[m*nx*ny+k*v+i]*g->vis[m*nx*ny+k*v+i]);
    //}

    for(long iy=0,i=0;iy<ny;iy++)
    for(long ix=0;ix<nx;ix++,i++)
    {
      g->partial_err += factorv * fabs(g->vis0[m*nx*ny+k*v+i]-g->vis[m*nx*ny+k*v+i]);
    }
    
  }
   
}

template<typename T>
struct worker_dat
{
  long nx;
  long ny;
  long kx;
  long ky;
  long dx;
  long dy;
  long K;
  long M;
};

///////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                           //
//            [nx x ny]            [kx x ky]            [dx x dy]                            //
//                                                                                           //
//      _____________________                                                                //
//      |     |             |          W            _________________                        //
//      |  c  |             |                       |c|             |                        //
//      |____x|             |       _______         |               |                        //
//      |                   |       |     |         |               |                        //
//      |                   |   *   |  c  |    =    |               |                        //
//      |                   |       |____x|         |               |                        //
//      |                   |                       |               |                        //
//      |                   |                       |               |                        //
//      |                   |                       |_______________|                        //
//      |___________________|                                                                //
//                                                                                           //
//      kx should be odd                                                                     //
//                                                                                           //
//      dx = nx - 2*(kx/2) <- integer division                                               //
//                                                                                           //
///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void vis2hid_worker ( worker_dat<T> * g
                    , const T   * X     // [nx x ny]
                    ,       T   * H     // [dx x dy]
                    ,       T   * c     // [dx x dy]
                    ,       T   * W     // [kx x ky]
                    , std::vector<std::pair<long,long> > const & vrtx
                    )
{
  
  long nx = g->nx;
  long ny = g->ny;
  long Kx = g->kx;
  long Ky = g->ky;
  long dx = g->dx;
  long dy = g->dy;
  long K = g->K;
  long M = g->M;
  long wx = Kx/2;
  long wy = Ky/2;
  long h = K*dx*dy;
  long v = M*nx*ny;
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t].first;
    long z = vrtx[t].second;
    long j = z*dx*dy;
    {
      for(long oy=0;oy<dy;oy++)
      for(long ox=0;ox<dx;ox++,j++)
      {
        H[k*h+j] = c[j]; 
        for(long m=0;m<M;m++)
        for(long ky=-wy,i=0;ky<=wy;ky++)
        for(long kx=-wx;kx<=wx;kx++,i++)
        {
          long iy=oy+wy+ky;
          long ix=ox+wx+kx;
          H[k*h+j] += W[m*K*Kx*Ky+z*Kx*Ky+i] * X[m*nx*ny+k*v+nx*iy+ix];
        }
        //H[k*h+j] = 1.0f/(1.0f + exp(-H[k*h+j]));
        //H[k*h+j] = atan(H[k*h+j]);
        //H[k*h+j] = 10.0f*atan(0.1f*H[k*h+j]);
        H[k*h+j] = 100*atan(0.01*H[k*h+j]);
      }
    }
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                           //
//           [dx x dy]            [kx x ky]                 [nx x ny]                        //
//                                                                                           //
//                                                    _____________________                  //
//       _________________           W^*              | ___ |             |                  //
//       |c| |           |                            | |c| |             |                  //
//       |__x|           |         _______            |_____|             |                  //
//       |               |         |     |            |                   |                  //
//       |               |    *    |  c  |     =      |                   |                  //
//       |               |         |____x|            |                   |                  //
//       |               |                            |                   |                  //
//       |               |                            |                   |                  //
//       |_______________|                            |                   |                  //
//                                                    |___________________|                  //
//                                                                                           //
//                                                                                           //
///////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void hid2vis_worker ( worker_dat<T> * g
                    , const T   * H     // [dx x dy]
                    ,       T   * V     // [nx x ny]
                    ,       T   * b     // [nx x ny]
                    ,       T   * W     // [kx x ky]
                    , std::vector<std::pair<long,long> > const & vrtx
                    )
{
  
  long nx = g->nx;
  long ny = g->ny;
  long Kx = g->kx;
  long Ky = g->ky;
  long dx = g->dx;
  long dy = g->dy;
  long K = g->K;
  long M = g->M;
  long wx = Kx/2;
  long wy = Ky/2;
  long h = K*dx*dy;
  long v = M*nx*ny;
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t].first;
    long m = vrtx[t].second;
    long i = m*nx*ny;
    for(long iy=0;iy<ny;iy++)
    for(long ix=0;ix<nx;ix++,i++)
    {
      //if ( ix      >= wx 
      //  && ix+wx <  nx
      //  && iy      >= wy
      //  && iy+wy <  ny
      //   )
      {
        V[k*v+i] = b[i];
        for(long z=0;z<K;z++)
        for(long ky=-wy,fy=2*wy,ty=0;ky<=wy;ky++,fy--,ty++)
        for(long kx=-wx,fx=2*wx,tx=0;kx<=wx;kx++,fx--,tx++)
        {
          long oy=iy-wy+ky;
          long ox=ix-wx+kx;
          if(oy>=0&&oy<dy&&ox>=0&&ox<dx)
          {
            V[k*v+i] += W[m*K*Kx*Ky+z*Kx*Ky+Kx*fy+fx] * H[k*h+z*dx*dy+dx*oy+ox]; // W is flipped here!
            //V[k*v+i] += W[z*Kx*Ky+Kx*ty+tx] * H[k*h+z*dx*dy+dx*oy+ox]; 
          }
        }
        //V[k*v+i] = 1.0f/(1.0f + exp(-V[k*v+i]));
        //V[k*v+i] = atan(V[k*v+i]);
        //V[k*v+i] = 10.0f*atan(0.1f*V[k*v+i]);
        V[k*v+i] = 100*atan(0.01*V[k*v+i]);
      }
      //else
      //{
      //  // do nothing, keep original values of V
      //  //V[k*v+i] = 0;
      //}
    }
  }
  
}

template<typename T>
struct ConvolutionalRBM : public BoltzmannMachine<T>
{

  T final_error;

  std::vector<T> errs;
  std::vector<T> test_errs;

  long h; // number hidden elements
  long v; // number visible elements
  long n; // number of samples
  long nx;
  long ny;
  long dx;
  long dy;
  long kx;
  long ky;
  long M;
  long K;
  T * c; // bias term for hidden state, R^h
  T * b; // bias term for visible state, R^v
  T * W; // weight matrix R^h*v
  T * X; // input data, binary [0,1], v*n

  T * vis0;
  T * hid0;
  T * vis;
  T * hid;
  T * dW;
  T * dc;
  T * db;

  ConvolutionalRBM ( long _v
                   , long _h
                   , long _nx
                   , long _ny
                   , long _dx
                   , long _dy
                   , long _kx
                   , long _ky
                   , long _M
                   , long _K
                   , T * _W
                   , T * _b
                   , T * _c
                   , long _n
                   , T * _X
                   )
  : BoltzmannMachine <T> ( CONVOLUTIONAL_RESTRICTED_BOLTZMANN_MACHINE_TYPE )
  {
    //for(long k=0;k<100;k++)
    //  std::cout << _X[k] << "\t";
    //std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    nx = _nx;
    ny = _ny;
    dx = _dx;
    dy = _dy;
    kx = _kx;
    ky = _ky;
    M = _M;
    K = _K;
    n = _n;
    c = _c;
    b = _b;
    W = _W;

    // error checks
    if(kx % 2 == 0){std::cout << "kx needs to be odd" << std::endl;exit(1);}
    if(ky % 2 == 0){std::cout << "ky needs to be odd" << std::endl;exit(1);}
    if(v != nx*ny) {std::cout << "v should be nx*ny" << std::endl;exit(1);}
    if(h != K*dx*dy) {std::cout << "h should be K*dx*dy" << std::endl;exit(1);}
    if(dx + 2*(kx/2) != nx) {std::cout << "nx should be dx + 2*(kx/2)" << std::endl;exit(1);}
    if(dy + 2*(ky/2) != ny) {std::cout << "ny should be dy + 2*(ky/2)" << std::endl;exit(1);}

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }

  void change_input_size ( long _nx, long _ny, long _dx, long _dy, long _n, T * dat )
  {
    dx = _dx;
    dy = _dy;
    nx = _nx;
    ny = _ny;
    n = _n;
    h = K*dx*dy;
    v = M*nx*ny;
    delete [] c;
    delete [] b;
    c = new T[K*dx*dy];
    b = new T[M*nx*ny];
    zero(c,K*dx*dy);
    zero(b,M*nx*ny);
    X = dat;
  }

  ConvolutionalRBM ( long _v
                   , long _h
                   , long _nx
                   , long _ny
                   , long _dx
                   , long _dy
                   , long _kx
                   , long _ky
                   , long _M
                   , long _K
                   , long _n
                   , T* _X
                   , bool gabor = true
                   , long _Nrot = 5
                   )
  {
    //for(long k=0;k<100;k++)
    //  std::cout << _X[k] << "\t";
    //std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    nx = _nx;
    ny = _ny;
    dx = _dx;
    dy = _dy;
    kx = _kx;
    ky = _ky;
    M = _M;
    K = _K;
    n = _n;

    //std::cout << "   M :    " << M << std::endl;

    // error checks
    if(kx % 2 == 0){std::cout << "kx needs to be odd" << std::endl;exit(1);}
    if(ky % 2 == 0){std::cout << "ky needs to be odd" << std::endl;exit(1);}
    if(v != M*nx*ny) {std::cout << "v should be M*nx*ny" << std::endl;exit(1);}
    if(h != K*dx*dy) {std::cout << "h should be K*dx*dy" << std::endl;exit(1);}
    if(dx + 2*(kx/2) != nx) {std::cout << "nx should be dx + 2*(kx/2)" << std::endl;exit(1);}
    if(dy + 2*(ky/2) != ny) {std::cout << "ny should be dy + 2*(ky/2)" << std::endl;exit(1);}

    c = new T[K*dx*dy];
    b = new T[M*nx*ny];
    W = new T[M*K*kx*ky];
    set_val<T>(c,0.0f,K*dx*dy);
    set_val<T>(b,0.0f,M*nx*ny);
    constant<T>(W,0,M*K*kx*ky);

    for(long m=0,i=0;m<M;m++)
    for(long k=0;k<K;k++)
    {
      //if(k==0 || gabor==false)
      if(gabor==false)
      {
        for(long x=0,j=0;x<kx;x++)
        for(long y=0;y<ky;y++,i++,j++)
        {
            //W[i] = (x==kx/2&&y+k/M==ky/2)?1.0/(M*_Nrot):0;
            W[i] = (x==kx/2&&y==ky/2)?0.01:0;
            //if(gabor==false && m!=(k%M))
            //if(gabor==false && m!=k)
            //if(gabor==false && (m!=0 || k!=0))
            if(gabor==false)
            W[i] = ((0.01)*(1.0 - 2*(rand()%10000)/10000.0f));
        }
      }
      else
      {
        long Nrot = _Nrot;
        long J = Nrot*(k)/(K);
        //long J = 1;
        std::cout << J << std::endl;
        T * gab = Gabor < T > ( kx                              // nx
                              , ky                              // ny
                              , (double)kx                      // lambda
                              , Nrot*2*M_PI*(double)(k)/(K)     // theta
                              , (J+1)*M_PI/2                    // phi
                              , (double)kx/4                    // sigma
                              , 1.0                             // gamma
                              );
        for(long x=0,j=0;x<kx;x++)
        for(long y=0;y<ky;y++,i++,j++)
        {
            //W[i] = (x==kx/2&&y==ky/2)?1.0:0;
            W[i] = 4*(gab[j])/(kx*ky);
        }
        delete [] gab;
      }
    }
    //char ch;
    //std::cin >> ch;
    //exit(1);

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }

  void init(int offset)
  {
    boost::posix_time::ptime time_start(boost::posix_time::microsec_clock::local_time());
    if(vis0==NULL)vis0 = new T[n*v];
    if(hid0==NULL)hid0 = new T[n*h];
    if(vis==NULL)vis = new T[n*v];
    if(hid==NULL)hid = new T[n*h];
    if(dW==NULL)dW = new T[M*K*kx*ky];
    if(dc==NULL)dc = new T[K*dx*dy];
    if(db==NULL)db = new T[M*nx*ny];

    //std::cout << "n*v=" << n*v << std::endl;
    //std::cout << "offset=" << offset << std::endl;
    for(long i=0,size=n*v;i<size;i++)
    {
      vis0[i] = X[i+offset];
    }

    vis2hid(vis0,hid0);
    boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration(time_end - time_start);
    //std::cout << "init timing:" << duration << '\n';
  }

  void cd(long nGS,T epsilon,int offset=0,bool bottleneck=false)
  {
    //std::cout << " M : " << M << std::endl;
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());
    //std::cout << "cd" << std::endl;

    // CD Contrastive divergence (Hinton's CD(k))
    //   [dW, db, dc, act] = cd(self, X) returns the gradients of
    //   the weihgts, visible and hidden biases using Hinton's
    //   approximated CD. The sum of the average hidden units
    //   activity is returned in act as well.

    for(long i=0;i<n*v;i++)
    {
      vis[i] = vis0[i];
    }
    vis2hid(vis0,hid0);
    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    //std::cout << "cd timing 1:" << duration10 << '\n';

    for (long iter = 1;iter<=nGS;iter++)
    {
      //std::cout << "iter=" << iter << std::endl;
      // sampling
      vis2hid(vis,hid);
      hid2vis(hid,vis);
    }
    //return;

    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    //std::cout << "cd timing 2:" << duration21 << '\n';
  
    zero(dW,M*K*kx*ky);
    zero(dc,K*dx*dy);
    zero(db,M*nx*ny);
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    //std::cout << "cd timing 3:" << duration32 << '\n';
    T * err = new T(0);
    gradient_update(n,vis0,hid0,vis,hid,dW,dc,db,err);
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    //std::cout << "cd timing 4:" << duration43 << '\n';
    //*err = sqrt(*err);
    //for(int t=2;t<3&&t<errs.size();t++)
    //  *err += (errs[errs.size()+1-t]-*err)/t;
    final_error = *err;
    errs.push_back(*err);
    test_errs.push_back(*err);
    static int cnt2 = 0;
    //if(cnt2%100==0)
    //std::cout << "rbm error=" << *err << std::endl;
    cnt2++;
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    //std::cout << "cd timing 5:" << duration54 << '\n';
    //std::cout << "epsilon = " << epsilon << std::endl;
    add(W,dW,-epsilon,M*K*kx*ky);
    //add(c,dc,-epsilon,K*dx*dy);
    //add(b,db,-epsilon,M*nx*ny);

    //std::cout << "dW norm = " << norm(dW,v*h) << std::endl;
    //std::cout << "dc norm = " << norm(dc,h) << std::endl;
    //std::cout << "db norm = " << norm(db,v) << std::endl;
    //std::cout << "W norm = " << norm(W,v*h) << std::endl;
    //std::cout << "c norm = " << norm(c,h) << std::endl;
    //std::cout << "b norm = " << norm(b,v) << std::endl;
    //std::cout << "err = " << *err << std::endl;
    delete err;

    boost::posix_time::ptime time_6(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration65(time_6 - time_5);
    //std::cout << "cd timing 6:" << duration65 << '\n';
    //char ch;
    //std::cin >> ch;
  }

  void sigmoid(T * p,T * X,long n)
  {
    for(long i=0;i<n;i++)
    {
      p[i] = 1.0f/(1.0f + exp(-X[i]));
    }
  }

  void vis2hid_simple(const T * X,T * H)
  {
    {
      for(long j=0;j<h;j++)
      {
        H[j] = c[j]; 
        for(long i=0;i<v;i++)
        {
          H[j] += W[i*h+j] * X[i];
        }
        H[j] = 1.0f/(1.0f + exp(-H[j]));
      }
    }
  }

  void hid2vis_simple(const T * H,T * V)
  {
    {
      for(long i=0;i<v;i++)
      {
        V[i] = b[i]; 
        for(long j=0;j<h;j++)
        {
          V[i] += W[i*h+j] * H[j];
        }
        V[i] = 1.0f/(1.0f + exp(-V[i]));
      }
    }
  }

  void gradient_update ( long n
                       , T * vis0
                       , T * hid0
                       , T * vis
                       , T * hid
                       , T * dW
                       , T * dc
                       , T * db
                       , T * err
                       )
  {
    //std::cout << "M:" << M << std::endl;
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());

    std::vector<boost::thread*> threads;
    std::vector<std::vector<std::pair<long,long> > > vrtx(boost::thread::hardware_concurrency());
    std::vector<crbm_gradient_info<T>*> g;

    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    //std::cout << "gradient update timing 1:" << duration10 << '\n';

    for(long i=0;i<n;i++)
    {
      for(long j=0;j<M;j++)
      {
        vrtx[(i+j)%vrtx.size()].push_back(std::pair<long,long>(i,j));
      }
    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    //std::cout << "gradient update timing 2:" << duration21 << '\n';
    for(long i=0;i<vrtx.size();i++)
    {
      g.push_back(new crbm_gradient_info<T>());
    }
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    //std::cout << "gradient update timing 3:" << duration32 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      g[thread]->n = n;
      g[thread]->v = v;
      g[thread]->h = h;
      g[thread]->nx = nx;
      g[thread]->ny = ny;
      g[thread]->dx = dx;
      g[thread]->dy = dy;
      g[thread]->kx = kx;
      g[thread]->ky = ky;
      g[thread]->K = K;
      g[thread]->M = M;
      g[thread]->vis0 = vis0;
      g[thread]->hid0 = hid0;
      g[thread]->vis = vis;
      g[thread]->hid = hid;
      g[thread]->dW = dW;
      g[thread]->dc = dc;
      g[thread]->db = db;
      g[thread]->init();
      threads.push_back(new boost::thread(crbm_gradient_worker<T>,g[thread],vrtx[thread]));
    }
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    //std::cout << "gradient update timing 4:" << duration43 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
      g[thread]->globalUpdate();
      *err += g[thread]->partial_err;
      g[thread]->destroy();
      delete g[thread];
    }
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    //std::cout << "gradient update timing 5:" << duration54 << '\n';
    threads.clear();
    vrtx.clear();
    g.clear();
  }
  
  void hid2vis(const T * H,T * V)
  {
    worker_dat<T> * D = new worker_dat<T>();
    D->nx = nx;
    D->ny = ny;
    D->dx = dx;
    D->dy = dy;
    D->kx = kx;
    D->ky = ky;
    D->K = K;
    D->M = M;
    std::vector<boost::thread*> threads;
    std::vector<std::vector<std::pair<long,long> > > vrtx(boost::thread::hardware_concurrency());
    for(long i=0;i<n;i++)
    {
      for(long j=0;j<M;j++)
      {
        vrtx[(i+j)%vrtx.size()].push_back(std::pair<long,long>(i,j));
      }
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(hid2vis_worker<T>,D,H,V,b,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
    threads.clear();
    vrtx.clear();
    delete D;
  }

  void vis2hid(const T * X,T * H)
  {
    worker_dat<T> * D = new worker_dat<T>();
    D->nx = nx;
    D->ny = ny;
    D->dx = dx;
    D->dy = dy;
    D->kx = kx;
    D->ky = ky;
    D->K = K;
    D->M = M;
    std::vector<boost::thread*> threads;
    std::vector<std::vector<std::pair<long,long> > > vrtx(boost::thread::hardware_concurrency());
    for(long i=0;i<n;i++)
    {
      for(long j=0;j<K;j++)
      {
        vrtx[(i+j)%vrtx.size()].push_back(std::pair<long,long>(i,j));
      }
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(vis2hid_worker<T>,D,X,H,c,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
    threads.clear();
    vrtx.clear();
    delete D;
  }

};

#endif

