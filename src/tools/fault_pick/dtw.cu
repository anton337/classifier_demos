#include <stdio.h>
#include <iostream>
#include <vector>
#include <boost/thread.hpp>

__global__
void dtw ( int n
         , int p_num_samp
         , int win
         , float * A
         , float * B
         , int * toggle
         , float * S
         )
{

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if ( i < n )
  {

    // init

    int length;
    int num_samp;
    int num_layers;
    int width;
    float ** dist;
    
    num_samp = p_num_samp;
    length = num_samp;
    num_layers = 2*num_samp - 1;
    width = 2*win + 1;
    dist = new float *[3];

    for(int k=0;k<3;k++)
    {
      dist[k] = new float [width];
      for(int i=0;i<width;i++)
      {
        dist[k][i] = 0;
      }
    }
    
    int N=num_layers*width;
    for(int k=0;k<num_layers;k++)
    {
      for(int j=0;j<width;j++)
      {
        toggle[i*N+width*k+j] = 0;
      }
    }
    

    int inter = 2;
    int min_toggle = -1;
    int s1,s2;
    int L,L_1,L_2;
    for ( int l=0; l < num_layers; l++ )
    {
      L = l%3;
      L_1 = (l-1+3)%3;
      L_2 = (l-2+3)%3;
      if ( l % 2 == 0 )
      {
        for ( int w=0; w < width; w++ )
        {
          s1 = -win + (l/2) + w;
          if ( s1 >= 0 && s1 >= (l/2) - win && s1 < num_samp)
          {
            s2 = l - s1;
            if ( s2 >= 0 && s2 >= (l/2) - win && s2 < num_samp)
            {
              min_toggle = -1;
              dist[L][w] = 10000;
              if ( l > 0 )
              {
                if (dist[L_1][w] <= dist[L][w])
                {
                  dist[L][w] = dist[L_1][w];
                  min_toggle = 0;
                }
                if ( w > 0 )
                {
                  if (dist[L_1][w-1] <= dist[L][w])
                  {
                    dist[L][w] = dist[L_1][w-1];
                    min_toggle = 1;
                  }
                }
              }
              if ( l > 1 )
              {
                if (dist[L_2][w] <= dist[L][w])
                {
                  dist[L][w] = dist[L_2][w];
                  min_toggle = 2;
                }
              }
              if(min_toggle==-1)
              {
                dist[L][w] = 0;
              }
              toggle[i*N+width*l+w] = min_toggle;
              for(int k=-inter;k<=inter;k++)
              {
                int count = 0;
                if (s1+k>=0&&s1+k<length&&s2+k>=0&&s2+k<length)
                {
                count++;
                dist[L][w] += fabs(A[i*p_num_samp+s1+k]-B[i*p_num_samp+s2+k]);//metric(A[s1+k],B[s2+k]);
                }
                dist[L][w] /= count+0.01f;
              }
            }
          }
        }
      }
      else
      {
        for ( int w=0; w < width; w++ )
        {
          s1 = -win + (l/2)+1 + w;
          if ( s1 >= 0 && s1 >= (l/2)+1 - win && s1 < num_samp)
          {
            s2 = l - s1;
            if ( s2 >= 0 && s2 >= (l/2)+1 - win && s2 < num_samp)
            {
              min_toggle = -1;
              dist[L][w] = 10000;
              if ( l > 0 )
              {
                if (dist[L_1][w] <= dist[L][w])
                {
                  dist[L][w] = dist[L_1][w];
                  min_toggle = 0;
                }
                if ( w+1 < width )
                {
                  if (dist[L_1][w+1] <= dist[L][w])
                  {
                    dist[L][w] = dist[L_1][w+1];
                    min_toggle = 3;
                  }
                }
              }
              if ( l > 1 )
              {
                if (dist[L_2][w] <= dist[L][w])
                {
                  dist[L][w] = dist[L_2][w];
                  min_toggle = 2;
                }
              }
              if (min_toggle==-1)
              {
                dist[L][w] = 0;
              }
              toggle[i*N+width*l+w] = min_toggle;
              for(int k=-inter;k<=inter;k++)
              {
                int count = 0;
                if (s1+k>=0&&s1+k<length&&s2+k>=0&&s2+k<length)
                {
                count++;
                dist[L][w] += fabs(A[i*p_num_samp+s1+k]-B[i*p_num_samp+s2+k]);//metric(A[s1+k],B[s2+k]);
                }
                dist[L][w] /= count+0.01f;
              }
            }
          }
        }
      }
    }
    

    for(int k=0;k<p_num_samp;k++)
    {
      S[i*p_num_samp+k] = 0;
    }
    int S1=1;
    int S2=1;
    int Layer = num_layers-1;
    int Win = win;
    int iter = 0;
    while(S1>0&&S2>0&&iter<2*length)
    {
      iter++;
      S1 = -win + (Layer/2) + (Layer%2) + Win;
      S2 = Layer - S1;
      S[i*p_num_samp+S1] = S2-S1;
      if(toggle[i*N+width*Layer+Win] == 0)
      {
        Layer--;
      }
      else
      if(toggle[i*N+width*Layer+Win] == 1)
      {
        Layer--;
        Win--;
      }
      else
      if(toggle[i*N+width*Layer+Win] == 2)
      {
        Layer-=2;
      }
      else
      if(toggle[i*N+width*Layer+Win] == 3)
      {
        Layer--;
        Win++;
      }
    }


    for(int k=0;k<3;k++)
    {
      delete [] dist[k];
    }
    delete [] dist;

  }

}

void dtw_cpu ( std::vector<int> indices
             , int n
             , int p_num_samp
             , int win
             , float * p_A
             , float * p_B
             , float * p_S
             )
{

  for(int k=0;k<indices.size();k++)
  {
    int i = indices[k];

    // init

    int length;
    int num_samp;
    int num_layers;
    int width;
    float ** dist;
    int ** toggle;

    float * A = &p_A[i*p_num_samp];
    float * B = &p_B[i*p_num_samp];
    float * S = &p_S[i*p_num_samp];
    
    num_samp = p_num_samp;
    length = num_samp;
    num_layers = 2*num_samp - 1;
    width = 2*win + 1;
    dist = new float *[3];
    for(int k=0;k<3;k++)
    {
      dist[k] = new float [width];
      for(int i=0;i<width;i++)
      {
        dist[k][i] = 0;
      }
    }
    toggle = new int *[num_layers];
    for(int k=0;k<num_layers;k++)
    {
      toggle[k] = new int [width];
      for(int i=0;i<width;i++)
      {
        toggle[k][i] = 0;
      }
    }


    int inter = 2;
    int min_toggle = -1;
    int s1,s2;
    int L,L_1,L_2;
    for ( int l=0; l < num_layers; l++ )
    {
      L = l%3;
      L_1 = (l-1+3)%3;
      L_2 = (l-2+3)%3;
      if ( l % 2 == 0 )
      {
        for ( int w=0; w < width; w++ )
        {
          s1 = -win + (l/2) + w;
          if ( s1 >= 0 && s1 >= (l/2) - win && s1 < num_samp)
          {
            s2 = l - s1;
            if ( s2 >= 0 && s2 >= (l/2) - win && s2 < num_samp)
            {
              min_toggle = -1;
              dist[L][w] = 10000;
              if ( l > 0 )
              {
                if (dist[L_1][w] <= dist[L][w])
                {
                  dist[L][w] = dist[L_1][w];
                  min_toggle = 0;
                }
                if ( w > 0 )
                {
                  if (dist[L_1][w-1] <= dist[L][w])
                  {
                    dist[L][w] = dist[L_1][w-1];
                    min_toggle = 1;
                  }
                }
              }
              if ( l > 1 )
              {
                if (dist[L_2][w] <= dist[L][w])
                {
                  dist[L][w] = dist[L_2][w];
                  min_toggle = 2;
                }
              }
              if(min_toggle==-1)
              {
                dist[L][w] = 0;
              }
              toggle[l][w] = min_toggle;
              for(int k=-inter;k<=inter;k++)
              {
                int count = 0;
                if (s1+k>=0&&s1+k<length&&s2+k>=0&&s2+k<length)
                {
                count++;
                dist[L][w] += fabs(A[s1+k]-B[s2+k]);//metric(A[s1+k],B[s2+k]);
                }
                dist[L][w] /= count+0.01f;
              }
            }
          }
        }
      }
      else
      {
        for ( int w=0; w < width; w++ )
        {
          s1 = -win + (l/2)+1 + w;
          if ( s1 >= 0 && s1 >= (l/2)+1 - win && s1 < num_samp)
          {
            s2 = l - s1;
            if ( s2 >= 0 && s2 >= (l/2)+1 - win && s2 < num_samp)
            {
              min_toggle = -1;
              dist[L][w] = 10000;
              if ( l > 0 )
              {
                if (dist[L_1][w] <= dist[L][w])
                {
                  dist[L][w] = dist[L_1][w];
                  min_toggle = 0;
                }
                if ( w+1 < width )
                {
                  if (dist[L_1][w+1] <= dist[L][w])
                  {
                    dist[L][w] = dist[L_1][w+1];
                    min_toggle = 3;
                  }
                }
              }
              if ( l > 1 )
              {
                if (dist[L_2][w] <= dist[L][w])
                {
                  dist[L][w] = dist[L_2][w];
                  min_toggle = 2;
                }
              }
              if (min_toggle==-1)
              {
                dist[L][w] = 0;
              }
              toggle[l][w] = min_toggle;
              for(int k=-inter;k<=inter;k++)
              {
                int count = 0;
                if (s1+k>=0&&s1+k<length&&s2+k>=0&&s2+k<length)
                {
                count++;
                dist[L][w] += fabs(A[s1+k]-B[s2+k]);//metric(A[s1+k],B[s2+k]);
                }
                dist[L][w] /= count+0.01f;
              }
            }
          }
        }
      }
    }
    

    int S1=1;
    int S2=1;
    int Layer = num_layers-1;
    int Win = win;
    int iter = 0;
    for(int k=0;k<p_num_samp;k++)
    {
      S[k] = 0;
    }
    while(S1>0&&S2>0&&iter<2*length)
    {
      iter++;
      S1 = -win + (Layer/2) + (Layer%2) + Win;
      S2 = Layer - S1;
      S[S1] = S2-S1;
      if(toggle[Layer][Win] == 0)
      {
        Layer--;
      }
      else
      if(toggle[Layer][Win] == 1)
      {
        Layer--;
        Win--;
      }
      else
      if(toggle[Layer][Win] == 2)
      {
        Layer-=2;
      }
      else
      if(toggle[Layer][Win] == 3)
      {
        Layer--;
        Win++;
      }
    }


    for(int k=0;k<3;k++)
    {
      delete [] dist[k];
    }
    delete [] dist;
    for(int k=0;k<num_layers;k++)
    {
      delete [] toggle[k];
    }
    delete [] toggle;

  }

}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

int main(void)
{


  int nt = 2*1024;
  int nx = 12*200;//1024;
  int win = 100;

  float *A, *B, *S, *S_cpu, *d_A, *d_B, *d_S;
  int *d_toggle;
  A = (float*)malloc(nt*nx*sizeof(float));
  B = (float*)malloc(nt*nx*sizeof(float));
  S = (float*)malloc(nt*nx*sizeof(float));
  S_cpu = (float*)malloc(nt*nx*sizeof(float));
  int num_layers = 2*nt - 1;
  int width = 2*win + 1;
  cudaMalloc(&d_A, nt*nx*sizeof(float)); 
  cudaMalloc(&d_B, nt*nx*sizeof(float)); 
  cudaMalloc(&d_S, nt*nx*sizeof(float)); 
  cudaMalloc(&d_toggle, nx*num_layers*width*sizeof(float));

  for(int x=0,k=0;x<nx;x++)
  for(int t=0;t<nt;t++,k++)
  {
    A[k] = (x+1)*t/(float)nt;
    B[k] = (x+1)*(t+(x+(int)sqrt(t)+1)%30)/(float)nt;
    S[k] = 0;
  }

  cudaMemcpy(d_A,A,nt*nx*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,B,nt*nx*sizeof(float),cudaMemcpyHostToDevice);

  std::cout << "begin GPU" << std::endl;
  clock_t start_gpu = clock();
  dtw<<<(nx+255)/256,256>>>(nx,nt,win,d_A,d_B,d_toggle,d_S);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  clock_t end_gpu = clock();
  std::cout << "end GPU" << std::endl;
  std::cout << "GPU time:" << end_gpu - start_gpu << std::endl;

  cudaMemcpy(S,d_S,nt*nx*sizeof(float),cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_S);
  cudaFree(d_toggle);

  int nthreads = 8;
  std::vector<boost::thread*> threads;
  std::vector<std::vector<int> > indices(nthreads);
  for(int i=0;i<nx;i++)
  {
    indices[i%nthreads].push_back(i);
  }
  std::cout << "begin CPU" << std::endl;
  clock_t start_cpu = clock();
  for(int i=0;i<nthreads;i++)
  {
    threads.push_back(new boost::thread(dtw_cpu,indices[i],nx,nt,win,A,B,S_cpu));
  }
  for(int i=0;i<nthreads;i++)
  {
    threads[i]->join();
  }
  clock_t end_cpu = clock();
  std::cout << "end CPU" << std::endl;
  std::cout << "CPU time:" << end_cpu - start_cpu << std::endl;
  std::cout << "speedup:" << (float)(end_cpu - start_cpu)/(float)(end_gpu - start_gpu) << "X" << std::endl;
  double Error = 0;
  for(int i=0;i<nt*nx;i++)
  {
    Error += fabs(S[i]-S_cpu[i]);
  }
  std::cout << "Error:" << Error << std::endl;

  return 0;
}

