#ifndef DTW_H
#define DTW_H

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

#endif

