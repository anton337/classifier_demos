#ifndef HOPFIELD_NETWORK_H
#define HOPFIELD_NETWORK_H

template < typename T >
struct HopfieldNode;

template < typename T >
struct HopfieldEdge
{
  HopfieldNode < T > * start;
  HopfieldNode < T > * end;
  T weight;
};

template < typename T >
struct HopfieldNode
{
  std::vector < HopfieldEdge < T > * > edges;
  T capacity;
  T value;
  void update()
  {
    T sum = 0;
    for(long i=0;i<edges.size();i++)
    {
      sum += edges[i]->weight * edges[i]->end->value;
    }
    if(sum >= capacity)
    {
      value = 1;
    }
    else
    {
      value = -1;
    }
  }
};

template < typename T >
struct HopfieldNetwork
{

  std::vector < HopfieldEdge < T > * > edges;
  std::vector < HopfieldEdge < T > * > nodes;

  T get_energy()
  {
    T KE = 0;
    for(long i=0;i<edges.size();i++)
    {
      KE += edges[i]->weight * edges[i]->start->value * edges[i]->end->value;
    }
    T PE = 0;
    for(long i=0;i<nodes.size();i++)
    {
      PE += nodes[i]->capacity * nodes[i]->value;
    }
    T E = -0.5*KE - PE;
    return E;
  }

  void update()
  {
    for(long i=0;i<nodes.size();i++)
    {
      nodes[i]->update();
    }
  }

  void Hebbian_learning(long n_nodes, T * dat)
  {

  }

  void Storkey_learning(long n_nodes, T * dat)
  {

  }

  void train(long n_nodes, T * dat)
  {
    if(n_nodes != nodes.size())
    {
      std::cout << "Hopfield network, number of nodes does not match." << std::endl;
      exit(1);
    }
    if(dat == NULL)
    {
      std::cout << "Hopfield network, training un-initialized array." << std::endl;
      exit(1);
    }

  }

};

#endif

