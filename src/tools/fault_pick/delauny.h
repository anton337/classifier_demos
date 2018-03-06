#ifndef DELAUNY_H
#define DELAUNY_H

#include <vector>
#include <cassert>
#include <iostream>
#include "boost/polygon/voronoi.hpp"

typedef boost::polygon::point_data<double> Point2D;

void delauny(std::vector<Point2D> const & vertices,std::vector<std::vector<Point2D> > & triangles)
{
    boost::polygon::voronoi_diagram<double> vd;
    boost::polygon::construct_voronoi(vertices.begin(), vertices.end(), &vd);
    int count = 0;
    for (const auto& vertex: vd.vertices()) {
        std::vector<Point2D> triangle;
        auto edge = vertex.incident_edge();
        do {
            auto cell = edge->cell();
            assert(cell->contains_point());
    
            triangle.push_back(vertices[cell->source_index()]);
            if (triangle.size() == 3) {
                // process output triangles
                //std::cout << "Got triangle:" << std::endl
                //          << boost::polygon::x(triangle[0]) 
                //  << '\t' << boost::polygon::y(triangle[0]) << std::endl
                //          << boost::polygon::x(triangle[1]) 
                //  << '\t' << boost::polygon::y(triangle[1]) << std::endl
                //          << boost::polygon::x(triangle[2]) 
                //  << '\t' << boost::polygon::y(triangle[2]) << std::endl;
                triangles.push_back(triangle);
                triangle.erase(triangle.begin() + 1);
            }
    
            edge = edge->rot_next();
        } while (edge != vertex.incident_edge());
    }
}

#endif

