#ifndef VISUALIZE_DATA_ARRAY
#define VISUALIZE_DATA_ARRAY

double * viz_dat = NULL;

long viz_selection = 0;

long n_elems = 0;

long n_vars = 0;

long n_x = 0;

long n_y = 0;

template<typename T>
void set_viz_data ( long num_elems
                  , long num_vars
                  , long nx
                  , long ny
                  , T * dat
                  )
{
    n_elems = num_elems;
    n_vars  = num_vars;
    n_x = nx;
    n_y = ny;
    viz_dat = dat;
}

template<typename T>
void visualize_data_array ( long selection
                          , long num_elems
                          , long num_vars
                          , long nx
                          , long ny
                          , T * dat
                          )
{

    if(dat == NULL)
    {
        std::cout << "Data array not initialized" << std::endl;
        exit(1);
    }

    if(num_elems % num_vars != 0)
    {
        std::cout << "Number of elements is not divisible by number of variables" << std::endl;
        exit(1);
    }

    if(nx*ny != num_vars)
    {
        std::cout << "nx * ny is not equal to number of variables" << std::endl;
        exit(1);
    }

    if(selection*num_vars >= num_elems)
    {
        std::cout << "selection index is outside of data range" << std::endl;
        exit(1);
    }
    
    float val;
    T dx = 2.0/nx;
    T dy = 2.0/ny;
    glBegin(GL_QUADS);
    for(long y=0,k=selection*num_vars;y<ny;y++)
    {
        for(long x=0;x<nx;x++,k++)
        {
            val = dat[k];
            glColor3f(val,val,val);
            glVertex3f( -1 + dx*x
                      , -1 + dy*y
                      , 0
                      );
            glVertex3f( -1 + dx*(x+1)
                      , -1 + dy*y
                      , 0
                      );
            glVertex3f( -1 + dx*(x+1)
                      , -1 + dy*(y+1)
                      , 0
                      );
            glVertex3f( -1 + dx*x
                      , -1 + dy*(y+1)
                      , 0
                      );
        }
    }
    glEnd();

}

#endif

