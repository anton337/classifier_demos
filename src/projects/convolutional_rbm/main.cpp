#include <iostream>
#include "ConvolutionalRBM.h"
#include "text.h"
#include "readBMP.h"

void clear() 
{
  // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
  std::cout << "\x1B[2J\x1B[H";
}

ConvolutionalRBM<double> * rbm = NULL;

int main(int argc,char ** argv)
{
  std::cout << "Convolutional RBM Test." << std::endl;
  srand(time(0));
  if(argc>0)
  {
    Image * img = new Image();
    ImageLoad(argv[1],img);
    long nx = 20;
    long ny = 20;
    long kx = 3;
    long ky = 3;
    long dx = nx-2*(kx/2);
    long dy = ny-2*(ky/2);
    long K = 8;
    double * dat_full = img->get_doubles(nx,ny);
    for(long x=0,k=0;x<nx;x++)
    {
        for(long y=0;y<ny;y++,k++)
        {
            std::cout << ((dat_full[k]>0.5)?'*':' ');
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    long full_n = img->get_size()/(nx*ny);
    long n = 32;
    double * dat = new double[n*nx*ny];
    for(long i=0,k=0;i<n;i++)
    {
        for(long x=0;x<nx;x++)
        {
            for(long y=0;y<ny;y++,k++)
            {
                dat[k] = dat_full[((987*i)%full_n)*nx*ny+(nx-1-x)*ny+y];
            }
        }
    }
    rbm = new ConvolutionalRBM<double> ( nx*ny
                                       , K*dx*dy
                                       , nx
                                       , ny
                                       , dx
                                       , dy
                                       , kx
                                       , ky
                                       , K
                                       , n
                                       , dat
                                       );
    double eps = 1e-1;
    bool init = true;
    double init_error;
    for(long i=0;;i++)
    {
      rbm -> init(0);
      rbm -> cd(1,eps,0);
      if(init)
      {
        init = false;
        init_error = rbm->final_error;
      }
      if(i%10==0)
      {
        clear();
        std::cout << "\033[1;31mconvolutional rbm\033[0m\n";
        //for(long z=0,k=0;z<K;z++)
        //{
        //  for(long x=0;x<kx;x++)
        //  {
        //    for(long y=0;y<ky;y++,k++)
        //    {
        //      std::cout << ((rbm->W[k]>0)?'*':' ');
        //    }
        //    std::cout << '\n';
        //  }
        //  std::cout << '\n';
        //}
        long ind = i%n;
        long ind_vis = ind*nx*ny;
        long ind_hid = ind*K*dx*dy;
        for(long x=0,k=0;x<nx;x++)
        {
            for(long y=0;y<ny;y++,k++)
            {
                std::cout << ((rbm->vis0[k+ind_vis]>0.5)?'*':'.');
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        {
            double min_val = 0;
            double max_val = 0;
            for(long x=0,k=0;x<nx;x++)
            {
                for(long y=0;y<ny;y++,k++)
                {
                    min_val = (min_val<rbm->vis[k+ind_vis])?min_val:rbm->vis[k+ind_vis];
                    max_val = (max_val>rbm->vis[k+ind_vis])?max_val:rbm->vis[k+ind_vis];
                }
            }
            for(long x=0,k=0;x<nx;x++)
            {
                for(long y=0;y<ny;y++,k++)
                {
                    std::cout << (((rbm->vis[k+ind_vis]-min_val)/(max_val-min_val)>0.5)?'*':'.');
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        {
            double min_val = 0;
            double max_val = 0;
            for(long x=0,k=0;x<dx;x++)
            {
                for(long y=0;y<dy;y++,k++)
                {
                    min_val = (min_val<rbm->hid[k+ind_hid])?min_val:rbm->hid[k+ind_hid];
                    max_val = (max_val>rbm->hid[k+ind_hid])?max_val:rbm->hid[k+ind_hid];
                }
            }
            for(long x=0,k=0;x<dx;x++)
            {
                for(long y=0;y<dy;y++,k++)
                {
                    std::cout << (((rbm->hid[k+ind_hid]-min_val)/(max_val-min_val)>0.5)?'*':'.');
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        for(long x=0,k=0;x<kx;x++)
        {
            for(long y=0;y<ky;y++,k++)
            {
                std::cout << rbm->W[k] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(long x=0,k=0;x<kx;x++)
        {
            for(long y=0;y<ky;y++,k++)
            {
                std::cout << rbm->dW[k] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << n << '\t' << init_error << '\t' << rbm -> final_error << std::endl;
      }
    }
  }
  else
  {
    std::cout << "Please specify input name [bmp format]." << std::endl;
    exit(1);
  }
    std::cout << "Done!" << std::endl;
  return 0;
}

