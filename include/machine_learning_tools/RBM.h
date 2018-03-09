#ifndef RBM_H
#define RBM_H

#include "BoltzmannMachine.h"

#include "common.h"

template<typename T>
struct gradient_info
{
  long n;
  long v;
  long h;
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
    partial_dW = new T[h*v];
    for(int i=0;i<h*v;i++)partial_dW[i]=0;
    partial_dc = new T[h];
    for(int i=0;i<h;i++)partial_dc[i]=0;
    partial_db = new T[v];
    for(int i=0;i<v;i++)partial_db[i]=0;
  }
  void destroy()
  {
    delete [] partial_dW;
    delete [] partial_dc;
    delete [] partial_db;
  }
  void globalUpdate()
  {
    for(int i=0;i<h*v;i++)
        dW[i] += partial_dW[i];
    for(int i=0;i<h;i++)
        dc[i] += partial_dc[i];
    for(int i=0;i<v;i++)
        db[i] += partial_db[i];
  }
};

template<typename T>
void gradient_worker(gradient_info<T> * g,std::vector<long> const & vrtx)
{
  T factor = 1.0f / g->n;
  T factorv= 1.0f / (g->v*g->v);
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long i=0;i<g->v;i++)
    {
      for(long j=0;j<g->h;j++)
      {
        g->partial_dW[i*g->h+j] -= factor * (g->vis0[k*g->v+i]*g->hid0[k*g->h+j] - g->vis[k*g->v+i]*g->hid[k*g->h+j]);
      }
    }

    for(long j=0;j<g->h;j++)
    {
      g->partial_dc[j] -= factor * (g->hid0[k*g->h+j]*g->hid0[k*g->h+j] - g->hid[k*g->h+j]*g->hid[k*g->h+j]);
    }

    for(long i=0;i<g->v;i++)
    {
      g->partial_db[i] -= factor * (g->vis0[k*g->v+i]*g->vis0[k*g->v+i] - g->vis[k*g->v+i]*g->vis[k*g->v+i]);
    }

    for(long i=0;i<g->v;i++)
    {
      g->partial_err += factorv * (g->vis0[k*g->v+i]-g->vis[k*g->v+i])*(g->vis0[k*g->v+i]-g->vis[k*g->v+i]);
    }
  }
}

template<typename T>
void vis2hid_worker(const T * X,T * H,long h,long v,T * c,T * W,std::vector<long> const & vrtx)
{
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long j=0;j<h;j++)
    {
      H[k*h+j] = c[j]; 
      for(long i=0;i<v;i++)
      {
        H[k*h+j] += W[i*h+j] * X[k*v+i];
      }
      H[k*h+j] = 1.0f/(1.0f + exp(-H[k*h+j]));
    }
  }
}

template<typename T>
void hid2vis_worker(const T * H,T * V,long h,long v,T * b,T * W,std::vector<long> const & vrtx)
{
  for(long t=0;t<vrtx.size();t++)
  {
    long k = vrtx[t];
    for(long i=0;i<v;i++)
    {
      V[k*v+i] = b[i]; 
      for(long j=0;j<h;j++)
      {
        V[k*v+i] += W[i*h+j] * H[k*h+j];
      }
      V[k*v+i] = 1.0f/(1.0f + exp(-V[k*v+i]));
    }
  }
}

template<typename T>
struct RBM : public BoltzmannMachine<T>
{

  std::vector<T> errs;
  std::vector<T> test_errs;

  long h; // number hidden elements
  long v; // number visible elements
  long n; // number of samples
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

  RBM(long _v,long _h,T * _W,T * _b,T * _c,long _n,T * _X) 
  : BoltzmannMachine ( RESTRICTED_BOLTZMANN_MACHINE_TYPE )
  {
    //for(long k=0;k<100;k++)
    //  std::cout << _X[k] << "\t";
    //std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = _c;
    b = _b;
    W = _W;

    vis0 = NULL;
    hid0 = NULL;
    vis = NULL;
    hid = NULL;
    dW = NULL;
    dc = NULL;
    db = NULL;
  }
  RBM(long _v,long _h,long _n,T* _X)
  {
    //for(long k=0;k<100;k++)
    //  std::cout << _X[k] << "\t";
    //std::cout << "\n";
    X = _X;
    h = _h;
    v = _v;
    n = _n;
    c = new T[h];
    b = new T[v];
    W = new T[h*v];
    constant<T>(c,0.5f,h);
    constant<T>(b,0.5f,v);
    constant<T>(W,0.5f,v*h);

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
    if(dW==NULL)dW = new T[h*v];
    if(dc==NULL)dc = new T[h];
    if(db==NULL)db = new T[v];

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
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());
    //std::cout << "cd" << std::endl;

    // CD Contrastive divergence (Hlongon's CD(k))
    //   [dW, db, dc, act] = cd(self, X) returns the gradients of
    //   the weihgts, visible and hidden biases using Hlongon's
    //   approximated CD. The sum of the average hidden units
    //   activity is returned in act as well.

    for(long i=0;i<n*h;i++)
    {
      hid[i] = hid0[i];
    }
    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    //std::cout << "cd timing 1:" << duration10 << '\n';

    for (long iter = 1;iter<=nGS;iter++)
    {
      //std::cout << "iter=" << iter << std::endl;
      // sampling
      hid2vis(hid,vis);
      vis2hid(vis,hid);

// Preview stuff
#if 0
      long off = dat_offset%(n);
      long offv = off*v;
      long offh = off*h;
      long off_preview = off*(3*WIN*WIN+10);
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis_preview[k] = vis[offv+k];
          vis_previewG[k] = vis[offv+k+WIN*WIN];
          vis_previewB[k] = vis[offv+k+2*WIN*WIN];
        }
      }
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis1_preview[k] = orig_arr[offset+off_preview+k];
          vis1_previewG[k] = orig_arr[offset+off_preview+k+WIN*WIN];
          vis1_previewB[k] = orig_arr[offset+off_preview+k+2*WIN*WIN];
        }
      }
      for(long x=0,k=0;x<WIN;x++)
      {
        for(long y=0;y<WIN;y++,k++)
        {
          vis0_preview[k] = vis0[offv+k];
          vis0_previewG[k] = vis0[offv+k+WIN*WIN];
          vis0_previewB[k] = vis0[offv+k+2*WIN*WIN];
        }
      }
#endif

    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    //std::cout << "cd timing 2:" << duration21 << '\n';
  
    zero(dW,v*h);
    zero(dc,h);
    zero(db,v);
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    //std::cout << "cd timing 3:" << duration32 << '\n';
    T * err = new T(0);
    gradient_update(n,vis0,hid0,vis,hid,dW,dc,db,err);
    boost::posix_time::ptime time_4(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration43(time_4 - time_3);
    //std::cout << "cd timing 4:" << duration43 << '\n';
    *err = sqrt(*err);
    //for(int t=2;t<3&&t<errs.size();t++)
    //  *err += (errs[errs.size()+1-t]-*err)/t;
    errs.push_back(*err);
    test_errs.push_back(*err);
    static int cnt2 = 0;
    //if(cnt2%100==0)
    std::cout << "rbm error=" << *err << std::endl;
    cnt2++;
    boost::posix_time::ptime time_5(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration54(time_5 - time_4);
    //std::cout << "cd timing 5:" << duration54 << '\n';
    //std::cout << "epsilon = " << epsilon << std::endl;
    add(W,dW,-epsilon,v*h);
    add(c,dc,-epsilon,h);
    add(b,db,-epsilon,v);

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

  void vis2hid(const T * X,T * H)
  {
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(vis2hid_worker<T>,X,H,h,v,c,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
    threads.clear();
    vrtx.clear();
  }

  void gradient_update(long n,T * vis0,T * hid0,T * vis,T * hid,T * dW,T * dc,T * db,T * err)
  {
    boost::posix_time::ptime time_0(boost::posix_time::microsec_clock::local_time());

    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    std::vector<gradient_info<T>*> g;

    boost::posix_time::ptime time_1(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration10(time_1 - time_0);
    //std::cout << "gradient update timing 1:" << duration10 << '\n';

    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    boost::posix_time::ptime time_2(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration21(time_2 - time_1);
    //std::cout << "gradient update timing 2:" << duration21 << '\n';
    for(long i=0;i<vrtx.size();i++)
    {
      g.push_back(new gradient_info<T>());
    }
    boost::posix_time::ptime time_3(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration duration32(time_3 - time_2);
    //std::cout << "gradient update timing 3:" << duration32 << '\n';
    for(long thread=0;thread<vrtx.size();thread++)
    {
      g[thread]->n = n;
      g[thread]->v = v;
      g[thread]->h = h;
      g[thread]->vis0 = vis0;
      g[thread]->hid0 = hid0;
      g[thread]->vis = vis;
      g[thread]->hid = hid;
      g[thread]->dW = dW;
      g[thread]->dc = dc;
      g[thread]->db = db;
      g[thread]->init();
      threads.push_back(new boost::thread(gradient_worker<T>,g[thread],vrtx[thread]));
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
    std::vector<boost::thread*> threads;
    std::vector<std::vector<long> > vrtx(boost::thread::hardware_concurrency());
    for(long i=0;i<n;i++)
    {
      vrtx[i%vrtx.size()].push_back(i);
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads.push_back(new boost::thread(hid2vis_worker<T>,H,V,h,v,b,W,vrtx[thread]));
    }
    for(long thread=0;thread<vrtx.size();thread++)
    {
      threads[thread]->join();
      delete threads[thread];
    }
    threads.clear();
    vrtx.clear();
  }

};

#endif

