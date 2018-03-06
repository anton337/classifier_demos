#ifndef TRIANGULATE_H
#define TRIANGULATE_H

#include <vector>
#include <map>
#include "delauny.h"
#include "triangulate.h"
#include "best_fit_plane.h"

template<typename T>
struct Point
{
  T x,y,z;
  Point(T _x,T _y,T _z)
    : x(_x), y(_y), z(_z)
  {

  }
  Point()
    : x(0), y(0), z(0)
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
        T t = nx*(a.x-ox) + ny*(a.y-oy) + nz*(a.z-oz);
        t /= nx*vx + ny*vy + nz*vz;
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
        T t = get_time_to_intercept(ox,oy,oz,vx,vy,vz);
        px = ox + t * vx;
        py = oy + t * vy;
        pz = oz + t * vz;
    }
    T sign ( T const & p1x
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
        T ret = (p1x - p3x)*(p2y - p3y) - (p2x - p3x)*(p1y - p3y);
        //std::cout << ret << std::endl;
        return ret;
    }
    bool is_inside ( T const & x
                   , T const & y
                   , T const & z
                   )
    {
        bool b1,b2,b3;
        b1 = sign(x,y,z,a.x,a.y,a.z,b.x,b.y,b.z) > 0.0f;
        b2 = sign(x,y,z,b.x,b.y,b.z,c.x,c.y,c.z) > 0.0f;
        b3 = sign(x,y,z,c.x,c.y,c.z,a.x,a.y,a.z) > 0.0f;
        //std::cout << "x:" << x << '\t' << y << '\t' << z << std::endl;
        //std::cout << "a:" << a.x << '\t' << a.y << '\t' << a.z << std::endl;
        //std::cout << "b:" << b.x << '\t' << b.y << '\t' << b.z << std::endl;
        //std::cout << "c:" << c.x << '\t' << c.y << '\t' << c.z << std::endl;
        //std::cout << "bool:" << b1 << '\t' << b2 << '\t' << b3 << std::endl;
        return (b1 == b2)&&(b2 == b3);
    }
    bool is_inside ( T const & x
                   , T const & y
                   , T const & z
                   , T const & nx
                   , T const & ny
                   , T const & nz
                   )
    {
        T px,py,pz;
        get_projection(x,y,z,nx,ny,nz,px,py,pz);
        return is_inside(px,py,pz);
        //return is_inside(x,y,z);
    }
};

template<typename T>
struct Triangulation
{
    std::vector<Triangle<T> > triangles;

    bool ready;

    bool is_inside ( T const & x
                   , T const & y
                   , T const & z
                   , T const & nx
                   , T const & ny
                   , T const & nz
                   )
    {
        if(ready)
        {
          for(long i=0;i<triangles.size();i++)
          {
            if(triangles[i].is_inside(x,y,z,nx,ny,nz))
              return true;
          }
        }
        return false;
    }

    void get_projection ( T const & x
                        , T const & y
                        , T const & z
                        , T const & nx
                        , T const & ny
                        , T const & nz
                        , T       & px
                        , T       & py
                        , T       & pz
                        )
    {
      px = 0;
      py = 0;
      pz = 0;
        if(ready)
        {
          for(long i=0;i<triangles.size();i++)
          {
            if(triangles[i].is_inside(x,y,z,nx,ny,nz))
            {
              triangles[i].get_projection(x,y,z,nx,ny,nz,px,py,pz);
              return;
            }
          }
        }
    }

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
        std::vector<Point2D> pts2d;
        std::map<std::pair<int,int>,int> Y;
        for(long i=0;i<pts.size();i++)
        {
          pts2d.push_back(Point2D((int)pts[i].x,(int)pts[i].z));
          Y[std::pair<int,int>((int)pts[i].x,(int)pts[i].z)] = (int)pts[i].y;
        }
        std::vector<std::vector<Point2D> > tri;
        delauny(pts2d,tri);
        for(long i=0;i<tri.size();i++)
        {
          triangles . push_back ( Triangle<T> ( Point<T> ( boost::polygon::x(tri[i][0]) , Y[std::pair<int,int>(boost::polygon::x(tri[i][0]), boost::polygon::y(tri[i][0]))] , boost::polygon::y(tri[i][0]) ) 
                                              , Point<T> ( boost::polygon::x(tri[i][1]) , Y[std::pair<int,int>(boost::polygon::x(tri[i][1]), boost::polygon::y(tri[i][1]))] , boost::polygon::y(tri[i][1]) )
                                              , Point<T> ( boost::polygon::x(tri[i][2]) , Y[std::pair<int,int>(boost::polygon::x(tri[i][2]), boost::polygon::y(tri[i][2]))] , boost::polygon::y(tri[i][2]) )
                                              ) 
                                );
        }
        if(triangles . size() > 0)
        {
          ready = true;
        }
    }

    /*
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
    */
};

#endif

