#include <vector>
#include <cassert>
#include <iostream>
#include "boost/polygon/voronoi.hpp"

typedef boost::polygon::point_data<double> Point;

void delauny()
{
    std::vector<Point> vertices;
    // add your input vertices
  
    for(long i=0;i<40;i++)
    {
      vertices.push_back(Point(-100+200*(rand()%100)/100.0,-100+200*(rand()%100)/100.0));
    }

    boost::polygon::voronoi_diagram<double> vd;
    boost::polygon::construct_voronoi(vertices.begin(), vertices.end(), &vd);
  
    std::cout << "Start:" << std::endl;
    int count = 0;
    for (const auto& vertex: vd.vertices()) {
        std::vector<Point> triangle;
        auto edge = vertex.incident_edge();
        do {
            auto cell = edge->cell();
            assert(cell->contains_point());
    
            triangle.push_back(vertices[cell->source_index()]);
            if (triangle.size() == 3) {
                // process output triangles
                std::cout << "Got triangle:" << std::endl
                          << boost::polygon::x(triangle[0]) 
                  << '\t' << boost::polygon::y(triangle[0]) << std::endl
                          << boost::polygon::x(triangle[1]) 
                  << '\t' << boost::polygon::y(triangle[1]) << std::endl
                          << boost::polygon::x(triangle[2]) 
                  << '\t' << boost::polygon::y(triangle[2]) << std::endl;
                triangle.erase(triangle.begin() + 1);
            }
    
            edge = edge->rot_next();
        } while (edge != vertex.incident_edge());
    }
}

int main()
{
    delauny();
    return 0;
}
