#include <iostream>
#include <boost/thread.hpp>
#include "sep_reader.h"
#include "best_fit_plane.h"
#define FLOAT_DISPLAY
#include "visualization.h"
#include "dtw.h"

std::string filename_dat = "/home/antonk/data/oxy.hdr";
std::string filename_afi = "/home/antonk/data/oxy-afi.hdr";
SEPReader reader_dat(filename_dat.c_str());
SEPReader reader_afi(filename_afi.c_str());
int ox = reader_dat.o3;
int oy = reader_dat.o2;
int oz = reader_dat.o1;
int nx = reader_dat.n3;
int ny = reader_dat.n2;
int nz = reader_dat.n1;
float * dat         = new float[nz*ny];
float * afi         = new float[nz*ny];
float * dat_flipped = new float[3*nz*ny]; // tri color
float * full        = new float[3L*nz*ny*nx];
int WIN = 15;
float * win_factor     = NULL;
float * win_factor_rev = NULL;
VisualizeDataArrayColor < float > * viz_in_dat = NULL;
VisualizeDataArrayColor < float > * viz_slice_dat = NULL;
VisualizeDataArrayColor < float > * viz_slice_dat2 = NULL;
VisualizeDataArrayColor < float > * viz_slice_dat3 = NULL;
VisualizeDataArrayColor < float > * viz_slice_dat4 = NULL;
int winx = 1000;
int winy = 500;
int probex = 300;
int probey = 300;
int probe  = probex/2;
int probe_samp  = probey/2;
float * dat_dtw_energy = new float[3*probex*probey];
float * dat_slice_m2 = new float[3*probex*probey];
float * dat_slice_m1 = new float[3*probex*probey];
float * dat_slice_p1 = new float[3*probex*probey];
float * dat_slice_p2 = new float[3*probex*probey];
float * dat_p1 = new float[probex*probey];
float * dat_m1 = new float[probex*probey];
float * dat_dtw_tmp = new float[probex*probey];
float * dat_dtw = new float[3*probex*probey];
float * dat_dtw_rev_tmp = new float[probex*probey];
float * dat_dtw_rev = new float[3*probex*probey];

float min_val = 100000000;
float max_val =-100000000;

int view_miny = 0;
int view_minz = 0;
int view_widthy = ny;
int view_widthz = nz;

long x = 0;
bool do_load = true;

std::vector<Point<float> > pts;
BestFitPlane<float> fit;

void load()
{
  while(true)
  {
    usleep(1000);
    if(key_up)
    {
        view_minz-=view_widthz*0.1;
        if(view_minz<0)view_minz=0;
        key_up = false;
        do_load = true;
    }
    if(key_down)
    {
        view_minz+=view_widthz*0.1;
        if(view_minz+view_widthz>nz)view_minz=nz-view_widthz;
        key_down = false;
        do_load = true;
    }
    if(key_UP)
    {
        view_widthz*=1.1;
        if(view_widthz>nz)view_widthz=nz;
        if(view_minz+view_widthz>nz)view_minz=nz-view_widthz;
        key_UP = false;
        do_load = true;
    }
    if(key_DOWN)
    {
        view_widthz/=1.1;
        key_DOWN = false;
        do_load = true;
    }
    if(key_left)
    {
        view_miny-=view_widthy*0.1;
        if(view_miny<0)view_miny=0;
        key_left = false;
        do_load = true;
    }
    if(key_right)
    {
        view_miny+=view_widthy*0.1;
        if(view_miny+view_widthy>ny)view_miny=ny-view_widthy;
        key_right = false;
        do_load = true;
    }
    if(key_LEFT)
    {
        view_widthy*=1.1;
        if(view_widthy>ny)view_widthy=ny;
        if(view_miny+view_widthy>ny)view_miny=ny-view_widthy;
        key_LEFT = false;
        do_load = true;
    }
    if(key_RIGHT)
    {
        view_widthy/=1.1;
        key_RIGHT = false;
        do_load = true;
    }
    if(left_selected)
    {
      left_selected = false;
      if((float)mouse_x/winy <= 1.0)
      {
        float X = x;
        float Z = view_minz + ((float)view_widthz/(float)nz)*nz*((float)mouse_y/winy);
        float Y = view_miny + ((float)view_widthy/(float)ny)*ny*(((float)winx/winy)*((float)mouse_x/winx));
        std::cout << X << '\t' << Y << '\t' << Z << std::endl;
        pts.push_back(Point<float>(X,Y,Z));
        fit.init(pts);
        do_load = true;
        for(long k=0;k<3*probex*probey;k++)
        {
          dat_slice_m2[k] = 0;
          dat_slice_m1[k] = 0;
          dat_slice_p1[k] = 0;
          dat_slice_p2[k] = 0;
          dat_dtw[k] = 0;
          dat_dtw_rev[k] = 0;
          dat_dtw_energy[k] = 0;
        }
      }
      else
      {
        std::cout << "Not clicking" << std::endl;
        probe = (int)(2*probex*(((float)mouse_x - winy)/winy))%probex;
        probe_samp = (int)(2*probey - 2*probey*((float)mouse_y/winy))%probey;
        std::cout << "probe:" << probe << '\t' << probe_samp << std::endl;
        do_load = true;
      }
    }
    if(right_selected)
    {
      right_selected = false;
      do_load = true;
      pts.clear();
      fit.init(pts);
      for(long k=0;k<3*probex*probey;k++)
      {
        dat_slice_m2[k] = 0;
        dat_slice_m1[k] = 0;
        dat_slice_p1[k] = 0;
        dat_slice_p2[k] = 0;
        dat_dtw[k] = 0;
        dat_dtw_rev[k] = 0;
        dat_dtw_energy[k] = 0;
      }
    }
    if(change_pos_index)
    {
      x++;
      if(x>=nx)
      {
        x=nx-1;
      }
      do_load = true;
    }
    if(change_neg_index)
    {
      x--;
      if(x<0)
      {
        x=0;
      }
      do_load = true;
    }
    if(do_load)
    {
      do_load = false;
      reader_dat.read_sepval ( &dat[0]
                             , reader_dat.o1
                             , reader_dat.o2
                             , reader_dat.o3 + x
                             , reader_dat.n1
                             , reader_dat.n2
                             , 1 // reader.n3
                             );
      reader_afi.read_sepval ( &afi[0]
                             , reader_afi.o1
                             , reader_afi.o2
                             , reader_afi.o3 + x
                             , reader_afi.n1
                             , reader_afi.n2
                             , 1 // reader.n3
                             );
      for(long k=0;k<ny*nz;k++)
      {
        min_val = (dat[k]<min_val)?dat[k]:min_val;
        max_val = (dat[k]>max_val)?dat[k]:max_val;
      }
      float r,g,b;
      long Y,Z;
      float facty = (float)ny / (float)view_widthy;
      float factz = (float)nz / (float)view_widthz;
      for(long y=0,k=0;y<ny;y++)
        for(long z=0;z<nz;z++,k++)
        {
          r = (dat[k]-min_val)/(max_val-min_val);
          g = r;
          b = afi[k];
          if(b<0.3||z<nz/3||z>2*nz/3)b=0;else b=1;
          full[3L*ny*nz*x+ny*nz*0+(y+(nz-1-z)*ny)] = r;
          full[3L*ny*nz*x+ny*nz*1+(y+(nz-1-z)*ny)] = g;
          full[3L*ny*nz*x+ny*nz*2+(y+(nz-1-z)*ny)] = b;
          Y = ((float)y - view_miny) * facty;
          Z = ((float)z - view_minz) * factz;
          if(Y>=0&&Y<ny&&Z>=0&&Z<nz)
          {
            for(long _y=Y;_y<=Y+facty;_y++)
            for(long _z=Z;_z<=Z+factz;_z++)
            if(_y>=0&&_y<ny&&_z>=0&&_z<nz)
            {
                dat_flipped[ny*nz*0+(_y+(nz-1-_z)*ny)] = r;
                dat_flipped[ny*nz*1+(_y+(nz-1-_z)*ny)] = g;
                dat_flipped[ny*nz*2+(_y+(nz-1-_z)*ny)] = b;
                if(pts.size()>=3)
                if ( fit.distance(x,y,z) < 1 
                  && (x-fit.anchorx)*(x-fit.anchorx) 
                   + (y-fit.anchory)*(y-fit.anchory) 
                   + (z-fit.anchorz)*(z-fit.anchorz) 
                   < probex*probex/4
                   )
                {
                  dat_flipped[ny*nz*0+(_y+(nz-1-_z)*ny)] = 1;
                }
            }
          }
        }
      
      long X;
      int W = 1;
      for(long t=0;t<pts.size();t++)
      {
        X = pts[t].x;
        Y = ((float)pts[t].y - view_miny) * facty;
        Z = ((float)pts[t].z - view_minz) * factz;
        if(Y>=0&&Y<ny&&Z>=0&&Z<nz)
        {
          for(long _y=Y-W;_y<=Y+W*facty;_y++)
          for(long _z=Z-W;_z<=Z+W*factz;_z++)
          if(X>=x-W&&X<=x+W&&_y>=0&&_y<ny&&_z>=0&&_z<nz)
          {
            dat_flipped[ny*nz*0+(_y+(nz-1-_z)*ny)] = 0;
            //dat_flipped[ny*nz*1+(_y+(nz-1-_z)*ny)] = 0;
            dat_flipped[ny*nz*2+(_y+(nz-1-_z)*ny)] = 0;
          }
        }
      }
      
      if(pts.size()>=3)
      {
        for(long _x=0,k=0;_x<probex;_x++)
        for(long _y=0;_y<probey;_y++,k++)
        {
          Point<float> pt = fit.get_projection((_y-probey/2),(_x-probex/2));
          if(fit.is_inside(pt))
          {
            //Point<float> p = pt;
            Point<float> p;
            fit.get_projection(pt,p);
            {
              long X = p.x + 10*fit.normx;
              long Y = p.y + 10*fit.normy;
              long Z = p.z;
              if(/*x==X&&*/X>=0&&X<nx&&Y>=0&&Y<ny&&Z>=0&&Z<nz)
              {
                dat_slice_p1[probex*probey*0+k] = (_y!=probe&&_x!=probe_samp)?(((_x+probex)%50>2&&(_y+probey)%50>2)?full[3L*ny*nz*X+ny*nz*1+(Y+(nz-1-Z)*ny)]:0):1;
                dat_slice_p1[probex*probey*1+k] = full[3L*ny*nz*X+ny*nz*1+(Y+(nz-1-Z)*ny)];
                dat_slice_p1[probex*probey*2+k] = full[3L*ny*nz*X+ny*nz*2+(Y+(nz-1-Z)*ny)];
                if(x==X)
                {
                    long _Y = ((float)Y - view_miny) * facty;
                    long _Z = ((float)Z - view_minz) * factz;
                    for(long __y=_Y-W;__y<=_Y+W;__y++)
                    for(long __z=_Z-W;__z<=_Z+W;__z++)
                    if(__y>=0&&__y<ny&&__z>=0&&__z<nz)
                    dat_flipped[ny*nz*0+(__y+(nz-1-__z)*ny)] = 1;
                }
              }
              else
              {
                if(_y==probe || _x==probe_samp)dat_slice_p1[probex*probey*0+k] = 1;
              }
            }
            {
              long X = p.x - 10*fit.normx;
              long Y = p.y - 10*fit.normy;
              long Z = p.z;
              if(/*x==X&&*/X>=0&&X<nx&&Y>=0&&Y<ny&&Z>=0&&Z<nz)
              {
                dat_slice_m1[probex*probey*0+k] = (_y!=probe&&_x!=probe_samp-dat_dtw_tmp[_x+_y*probex])?(((_x+probex)%50>2&&(_y+probey)%50>2)?full[3L*ny*nz*X+ny*nz*1+(Y+(nz-1-Z)*ny)]:0):1;
                dat_slice_m1[probex*probey*1+k] = full[3L*ny*nz*X+ny*nz*1+(Y+(nz-1-Z)*ny)];
                dat_slice_m1[probex*probey*2+k] = full[3L*ny*nz*X+ny*nz*2+(Y+(nz-1-Z)*ny)];
                if(x==X)
                {
                    long _Y = ((float)Y - view_miny) * facty;
                    long _Z = ((float)Z - view_minz) * factz;
                    for(long __y=_Y-W;__y<=_Y+W;__y++)
                    for(long __z=_Z-W;__z<=_Z+W;__z++)
                    if(__y>=0&&__y<ny&&__z>=0&&__z<nz)
                    dat_flipped[ny*nz*0+(__y+(nz-1-__z)*ny)] = 1;
                }
              }
              else
              {
                if(_y==probe || _x==probe_samp)dat_slice_m1[probex*probey*0+k] = 1;
              }
            }
          }
        }
        for(long _x=0,k=0;_x<probex;_x++)
        for(long _y=0;_y<probey;_y++,k++)
        {
          dat_p1[_x+probex*_y] = dat_slice_p1[probex*probey+k];
          dat_m1[_x+probex*_y] = dat_slice_m1[probex*probey+k];
        }
        std::vector<int> indices;
        for(long i=0;i<probex;i++)
        {
          indices.push_back(i);
        }
        for(long i=0;i<probex*probey;i++)
        {
            dat_dtw_energy[i] = 0;
        }
        dtw_cpu ( indices
                , probex
                , probey
                , WIN
                , win_factor
                , dat_m1
                , dat_p1
                , dat_dtw_tmp
                , probe
                , dat_dtw_energy
                );
        for(long _x=0,k=0;_x<probex;_x++)
        for(long _y=0;_y<probey;_y++,k++)
        {
          dat_dtw[_x+_y*probex]                 = 0.5 + 0.02*((_x!=probe&&_y!=probe_samp)?dat_dtw_tmp[k]:1-dat_dtw_tmp[k]);
          dat_dtw[_x+_y*probex+probex*probey]   = 0.5 + 0.02*(dat_dtw_tmp[k]);
          dat_dtw[_x+_y*probex+2*probex*probey] = 0.5 + 0.02*(dat_dtw_tmp[k]);
          //dat_dtw_energy[k] *= 10;
          dat_dtw_energy[k+probex*probey] = dat_dtw_energy[k];
          dat_dtw_energy[k+2*probex*probey] = dat_dtw_energy[k];
        }
        /*
        dtw_cpu ( indices
                , probex
                , probey
                , WIN
                , win_factor_rev
                , dat_p1
                , dat_m1
                , dat_dtw_rev_tmp
                , probex/2
                , dat_dtw_energy
                );
        for(long _x=0,k=0;_x<probex;_x++)
        for(long _y=0;_y<probey;_y++,k++)
        {
          dat_dtw_rev_tmp[k] /= 15;
          dat_dtw_rev_tmp[k] += 0.5;
          dat_dtw_rev[_x+_y*probex]                 = dat_dtw_rev_tmp[k];
          dat_dtw_rev[_x+_y*probex+probex*probey]   = dat_dtw_rev_tmp[k];
          dat_dtw_rev[_x+_y*probex+2*probex*probey] = dat_dtw_rev_tmp[k];
        }
        */
      }
      
    }
  }
}

void reshape ( int width, int height ) {
    height = width / 2;
    winx = width;
    winy = height;
        /* define the viewport transformation */
        glViewport(0,0,width,height);

}

int main(int argc,char ** argv)
{
    std::cout << "Fault Pick Tool." << std::endl;
    win_factor = new float[2*WIN+1];
    win_factor_rev = new float[2*WIN+1];
    for(long i=-WIN,k=0;i<=WIN;i++,k++)
    {
        win_factor_rev  [k] = 0;//0.005*fabs(i);//(i<0)?10:1;
        win_factor      [k] = 0;//0.005*fabs(i);//(i>0)?10:1;
    }
    viz_in_dat = new VisualizeDataArrayColor < float > ( 3*ny*nz
                                                       , 3*ny*nz
                                                       , 3*ny*nz
                                                       , ny
                                                       , nz
                                                       , dat_flipped
                                                       , -1 , 0 , -1 , 1
                                                       );
    viz_slice_dat = new VisualizeDataArrayColor < float > ( 3*probex*probey
                                                          , 3*probex*probey
                                                          , 3*probex*probey
                                                          , probex
                                                          , probey
                                                          , dat_slice_p1
                                                          , 0 , .5 , -1 , 0
                                                          );
    viz_slice_dat2 = new VisualizeDataArrayColor < float > ( 3*probex*probey
                                                           , 3*probex*probey
                                                           , 3*probex*probey
                                                           , probex
                                                           , probey
                                                           , dat_slice_m1
                                                           , .5 , 1 , -1 , 0
                                                           );
    viz_slice_dat3 = new VisualizeDataArrayColor < float > ( 3*probex*probey
                                                          , 3*probex*probey
                                                          , 3*probex*probey
                                                          , probex
                                                          , probey
                                                          , dat_dtw
                                                          , 0 , .5 , 0 , 1
                                                          );
    viz_slice_dat4 = new VisualizeDataArrayColor < float > ( 3*probex*probey
                                                           , 3*probex*probey
                                                           , 3*probex*probey
                                                           , probex
                                                           , probey
                                                           , dat_dtw_energy
                                                           , .5 , 1 , 0 , 1
                                                           );
    addDisplay ( viz_in_dat );
    addDisplay ( viz_slice_dat );
    addDisplay ( viz_slice_dat2 );
    addDisplay ( viz_slice_dat3 );
    addDisplay ( viz_slice_dat4 );
    new boost::thread(load);

    glutInit(&argc, argv);
    glutInitWindowSize(winx,winy);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutCreateWindow("DTW");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialInput);
    glutMouseFunc(OnMouseClick);
    init();
    glutMainLoop();
    std::cout << "Finished." << std::endl;
    return 0;
}

