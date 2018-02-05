#ifndef VISUALIZE_ERRORS
#define VISUALIZE_ERRORS

template<typename T>
void visualize_errors ( std::vector<T> errs )
{
    float max_err = 0;
    for(long k=0;k<errs.size();k++)
    {
        if(max_err<errs[k])max_err=errs[k];
    }
    glColor3f(1,1,1);
    glBegin(GL_LINES);
    for(long k=0;k+1<errs.size();k++)
    {
        glVertex3f( -1 + 2*k / ((float)errs.size()-1)
                  , errs[k] / max_err
                  , 0
                  );
        glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
                  , errs[k+1] / max_err
                  , 0
                  );
        glVertex3f( -1 + 2*k / ((float)errs.size()-1)
                  , 0
                  , 0
                  );
        glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
                  , 0
                  , 0
                  );
        glVertex3f( -1 + 2*k / ((float)errs.size()-1)
                  , 0
                  , 0
                  );
        glVertex3f( -1 + 2*k / ((float)errs.size()-1)
                  , errs[k] / max_err
                  , 0
                  );
        glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
                  , 0
                  , 0
                  );
        glVertex3f( -1 + 2*(k+1) / ((float)errs.size()-1)
                  , errs[k+1] / max_err
                  , 0
                  );
    }
}

#endif

