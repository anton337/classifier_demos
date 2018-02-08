#ifndef DISPLAY_H
#define DISPLAY_H

template<typename T>
struct Display
{

    T min_x;
    T max_x;
    T min_y;
    T max_y;

    Display ( T p_min_x
            , T p_max_x
            , T p_min_y
            , T p_max_y
            )
    : min_x ( p_min_x )
    , max_x ( p_max_x )
    , min_y ( p_min_y )
    , max_y ( p_max_y )
    {

    }

    virtual void update() = 0;

    virtual void signal ( bool pos 
                        , bool neg 
                        , bool up 
                        , bool down 
                        )
    {


    }

};

#endif

