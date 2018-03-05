#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include <vector>
#include <map>
#include "best_fit_plane.h"

template<typename T>
struct Point
{
  T x,y,z;
  Point(T _x,T _y,T _z)
    : x(_x), y(_y), z(_z)
  {

  }
};

template<typename T>
struct Triangle
{
    Point<T> a,b,c;
    Triangle(Point<T> const & _a,Point<T> const & _b,Point<T> const & _c)
        : a(_a) , b(_b) , c(_c)
    {

    }
    void get_normal(T & x,T & y,T & z)
    {
        x = (b.y-a.y)*(c.z-a.z) - (b.z-a.z)*(c.y-a.y);
        y = (b.z-a.z)*(c.x-a.x) - (b.x-a.x)*(c.z-a.z);
        z = (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
        T normal = sqrtf(x*x+y*y+z*z);
        if(normal > 1e-10)
        {
            x/=normal;
            y/=normal;
            z/=normal;
        }
        else
        {
            x=0;
            y=0;
            z=1;
        }
    }
    T get_time_to_intercept ( T const & ox
                            , T const & oy
                            , T const & oz
                            , T const & vx
                            , T const & vy
                            , T const & vz
                            )
    {
        T nx,ny,nz;
        get_normal(nx,ny,nz);
        T t = ((a.x*ox+a.y*oy+a.z*oz) - (nx*ox+ny*oy+nz*oz)) / ((a.x*vx+a.y*vy+a.z*vz) - (nx*vx+ny*vy+nz*vz));
        return t;
    }
    void get_projection ( T const & ox
                        , T const & oy
                        , T const & oz
                        , T const & vx
                        , T const & vy
                        , T const & vz
                        , T       & px
                        , T       & py
                        , T       & pz
                        )
    {
        T t = get_time_to_intercept();
        px = ox + t * vx;
        py = oy + t * vy;
        pz = oz + t * vz;
    }
    float sign ( T const & p1x
               , T const & p1y
               , T const & p1z
               , T const & p2x
               , T const & p2y
               , T const & p2z
               , T const & p3x
               , T const & p3y
               , T const & p3z
               )
    {
        return (p1x - p3x)*(p2y - p3y) - (p2x - p3x)*(p1y - p3y);
    }
    bool is_inside ( T const & x
                   , T const & y
                   , T const & z
                   )
    {
        bool b1,b2,b3;
        b1 = sign(x,y,z,a.x,a.y,a.z,b.x,b.y,b.z) < 0.0f;
        b2 = sign(x,y,z,b.x,b.y,b.z,c.x,c.y,c.z) < 0.0f;
        b3 = sign(x,y,z,c.x,c.y,c.z,a.x,a.y,a.z) < 0.0f;
        return (b1 == b2)&&(b2 == b3);
    }
};

template<typename T>
struct Triangulation
{
    std::vector<Triangle<T> > triangles;

    bool ready;

    Triangulation()
    {
        ready = false;
    }

    void clear()
    {
        triangles.clear();
        ready = false;
    }

    void get_normal ( T & x , T & y , T & z )
    {
        x = 0;
        y = 0;
        z = 0;
        if(ready)
        {
            T tmpx, tmpy, tmpz;
            for(long i=0;i<triangles.size();i++)
            {
                triangles[i].get_normal(tmpx,tmpy,tmpz);
                x += tmpx;
                y += tmpy;
                z += tmpz;
            }
            T normal = sqrtf(x*x+y*y+z*z);
            x /= normal;
            y /= normal;
            z /= normal;
        }
    }

    void triangulate ( std::vector<Point<T> > const & pts
                     )
    {
        ready = false;
        triangles . clear ();
        std::map<int, std::vector<Point<T> > > lines;
        for(long i=0;i<pts.size();i++)
        {
            if(lines.find((int)pts[i].x) == lines.end())
            {
                lines[(int)pts[i].x] = std::vector<Point<T> >();
            }
            lines[(int)pts[i].x].push_back(pts[i]);
        }
        if(lines.size()>=2)
        {
            typename std::map<int,std::vector<Point<T> > >::iterator it_curr = lines.begin();
            typename std::map<int,std::vector<Point<T> > >::iterator it_next = lines.begin();
            it_next++;
            for(;it_next!=lines.end();)
            {
                bool side = 0;
                for(long i1=0,i2=0;;)
                {
                    if(side==0)
                    {
                        if(i1+1>=it_curr->second.size())
                        {
                            break;
                        }
                        Point<T> a = it_curr->second[i1];
                        Point<T> b = it_curr->second[i1+1];
                        Point<T> c = it_next->second[i2];
                        triangles.push_back(Triangle<T>(a,b,c));
                        i2++;
                        if(i2>=it_next->second.size())
                        {
                            break;
                        }
                        side=1;
                    }
                    else
                    {
                        if(i2+1>=it_next->second.size())
                        {
                            break;
                        }
                        Point<T> a = it_next->second[i2];
                        Point<T> b = it_next->second[i2+1];
                        Point<T> c = it_curr->second[i1];
                        triangles.push_back(Triangle<T>(a,c,b));
                        i1++;
                        if(i1>=it_next->second.size())
                        {
                            break;
                        }
                        side=0;
                    }
                }
                it_curr++;
                it_next++;
            }
            ready = true;
        }
    }
};

#endif

