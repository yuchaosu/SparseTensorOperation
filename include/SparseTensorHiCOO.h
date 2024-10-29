#ifndef SPARSETENSORHICOO_H
#define SPARSETENSORHICOO_H

#include <vector>
#include <iostream>

class SparseTensorHiCOO
{
public:
    SparseTensorHiCOO(int dim1, int dim2, int dim3, int block_size);

    void addElement(int i, int j, int k, double value);
    
    void print() const;


    std::vector<int> blockCoord1, blockCoord2, blockCoord3;
    std::vector<int> elemCoord1, elemCoord2, elemCoord3;
    std::vector<double> values;
    int dim1, dim2, dim3;
    int blockSize;
};

#endif