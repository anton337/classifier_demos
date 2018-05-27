#include <iostream>
#include <vector>
#include <math.h>
#include <unistd.h>
#include <GL/glut.h>


class
segment
{
  float AX,AY;
  float BX,BY;
  float nx;
  float ny;
  float R;
  public:
  segment(float _AX,float _AY,float _BX,float _BY)
    : AX(_AX), AY(_AY), BX(_BX), BY(_BY)
  {
    nx = (BY - AY);
    ny =-(BX - AX);
    float n = sqrt(nx*nx+ny*ny);
    nx /= n;
    ny /= n;
    R = sqrt((BX - AX)*(BX - AX) + (BY - AY)*(BY - AY));
  }
  void draw()
  {
    glVertex3f(AX,AY,0);
    glVertex3f(BX,BY,0);
  }
  float distance(float orig_x,float orig_y,float dir_x,float dir_y) const
  {
    float d = -(dir_x*nx + dir_y*ny);
    float dir = sqrt(dir_x*dir_x + dir_y*dir_y);
    if(fabs(d) < 1e-10) return -1;
    float t = ((orig_x - AX)*nx + (orig_y - AY)*ny) / d;
    float dx = orig_x + dir_x*t - (AX+BX)*.5;
    float dy = orig_y + dir_y*t - (AY+BY)*.5;
    float r = sqrt(dx*dx + dy*dy);
    if(r > R*.5)return -1;
    if(t > 0 && t < .1)return 1;
    if(t > 0)return -1;
    return t;
  }
};

std::vector<segment> segments;

void
initRoad()
{
  float R=1;
  float delta=2*M_PI/70;
  for(float th=0;th<M_PI;th+=delta){
    segments.push_back ( segment ( (R+.1)*(1+0.2*cos((th-0.5*delta)*6))*cos(th-0.5*delta)
                                 , (R+.1)*(1+0.2*cos((th-0.5*delta)*6))*sin(th-0.5*delta)
                                 , (R+.1)*(1+0.2*cos((th+0.5*delta)*6))*cos(th+0.5*delta)
                                 , (R+.1)*(1+0.2*cos((th+0.5*delta)*6))*sin(th+0.5*delta)
                                 )
                       );
    segments.push_back ( segment ( (R-.1)*(1+(0.25+th*0.05)*cos((th-0.5*delta)*6))*cos(th-0.5*delta)
                                 , (R-.1)*(1+(0.25+th*0.05)*cos((th-0.5*delta)*6))*sin(th-0.5*delta)
                                 , (R-.1)*(1+(0.25+th*0.05)*cos((th+0.5*delta)*6))*cos(th+0.5*delta)
                                 , (R-.1)*(1+(0.25+th*0.05)*cos((th+0.5*delta)*6))*sin(th+0.5*delta)
                                 )
                       );
  }
}

class
car
{
  float init_x,init_y;
  float x,y;
  float vx,vy;
  float ax,ay;
  float th;
  bool live;
  int iter;
  std::vector<float> data;
  std::vector<float> param;
  public:
  float distance_traveled(){
    return sqrt((x-init_x)*(x-init_x) + (y-init_y)*(y-init_y));
  }
  int get_num_iters(){
    return iter;
  }
  std::vector<float> const & get_params(){
    return param;
  }
  void set_params(int i,float val){
    param[i] = val;
  }
  public:
  car(float _x,float _y,float _th)
  {
    init(_x,_y,_th);
  }
  bool is_live(){return live;}
  void init(float _x,float _y,float _th){
    data.resize(3*2+1);
    param.resize(3*2+1);
    for(int i=0;i<param.size();i++){
      param[i] = 0.1*(1-2*(rand()%1000)/1000.0f);
    }
    x = _x;
    y = _y;
    init_x = x;
    init_y = y;
    th = _th;
    vx = 0;
    vy = 0;
    ax = 0;
    ay = 0;
    live = true;
    iter = 0;
  }
  private:
  float get_closest(std::vector<segment> const & segments,float orig_x,float orig_y,float dir_x,float dir_y){
    float D = -1e10;
    float d;
    for(int i=0;i<segments.size();i++){
      d = segments[i].distance(orig_x,orig_y,dir_x,dir_y);
      D = (d > D)?d:D;
    }
    if(D > 0.9)live = false;
    return D;
  }
  void collect_data(std::vector<segment> const & segments){
    int I = 3;
    float D = 0.2;
    for(int i=-I,k=0;i<=I;i++,k++){
      float dth = 0.5*i*M_PI/I;
      float dx = -D*cos(th+dth);
      float dy = -D*sin(th+dth);
      data[k] = get_closest(segments,x,y,dx,dy);
    }
  }
  float model(){
    float D = 0;
    for(int i=0;i<param.size();i++){
      D += param[i]*data[i];
    }
    return D;
  }
  public:
  void simulate(){
    collect_data(segments);
    if(live){
      float V = 0.001;
      ax = V*cos(th);
      ay = V*sin(th);
      vx += ax;
      vy += ay;
      x  += vx;
      y  += vy;
      vx *= 0.75;
      vy *= 0.75;
      th += model();
      iter ++;
    }
  }
  void draw()
  {
    glVertex3f(x+0.03*cos(th),y+0.03*sin(th),0);
    glVertex3f(x-0.03*cos(th),y-0.03*sin(th),0);
    glVertex3f(x+0.03*sin(th),y-0.03*cos(th),0);
    glVertex3f(x+0.03*cos(th),y+0.03*sin(th),0);
    glVertex3f(x-0.03*sin(th),y+0.03*cos(th),0);
    glVertex3f(x+0.03*cos(th),y+0.03*sin(th),0);
    //int I = 3;
    //float D = 0.2;
    //for(int i=-I,k=0;k<data.size();i++,k++){
    //  float dth = 0.5*i*M_PI/I;
    //  float dx = -data[k]*D*cos(th+dth);
    //  float dy = -data[k]*D*sin(th+dth);
    //  glVertex3f(x,y,0);
    //  glVertex3f(x+dx,y+dy,0);
    //}
  }
};

std::vector<car*> drone;

int n_iters = 1000;

int g_iter = 0;

bool all_dead = false;

bool go = false;

std::vector<float> g_params;

void
drawCars(void)
{
  {
    for(int i=0;i<drone.size();i++){
      drone[i] -> simulate();
    }
    glBegin(GL_LINES);
    glColor3f(1,1,1);
    for(int i=0;i<segments.size();i++){
      segments[i].draw();
    }
    for(int i=0;i<drone.size();i++){
      drone[i] -> draw();
    }
    glEnd();
    all_dead = true;
    for(int i=0;i<drone.size();i++){
      if(drone[i] -> is_live() == true){
        all_dead = false;
      }
    }
  }
  g_iter++;
  if(g_iter > n_iters || all_dead == true){
    g_iter = 0;
    float max_val = 0;
    float err_val;
    for(int i=0;i<drone.size();i++){
      err_val = drone[i]->distance_traveled();
      if(err_val > max_val){
        max_val = err_val;
        g_params.resize(drone[i]->get_params().size());
        for(int j=0;j<drone[i]->get_params().size();j++){
          g_params[j] = drone[i]->get_params()[j];
        }
      }
    }
    for(int i=0;i<drone.size();i++){
      drone[i] -> init(1.2,0,M_PI/2);
      for(int j=0;j<drone[i]->get_params().size();j++){
        drone[i]->set_params(j,g_params[j] + i*1e-2*(1-2*(rand()%1000)/1000.0f)/drone.size());
      }
    }
  }
}

void
display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if(go){
    drawCars();
  }
  glutSwapBuffers();
}

void idle(void)
{
  glutPostRedisplay();
}

void
init(void)
{

  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 5.0,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */

  /* Adjust cube position to be asthetic angle. */
  glTranslatef(0.0, 0.0, -1.0);
}

void
keyboard(unsigned char Key, int x, int y)
{
  switch(Key)
  {
    case ' ':go = true;break;
    case 27:
      {
        exit(1);
        break;
      }
  };
}

int
main(int argc, char **argv)
{
  srand(time(0));
  for(int i=0;i<10;i++){
    drone.push_back(new car(1.2,0,M_PI/2));
  }
  std::cout << "Car Simulation" << std::endl;
  std::cout << "press space to start..." << std::endl;
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("Car Simulation");
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  glutKeyboardFunc(keyboard);
  initRoad();
  init();
  glutMainLoop();
  return 0;             /* ANSI C requires main to return int. */
}

