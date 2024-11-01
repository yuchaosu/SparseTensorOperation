#ifndef SPARSETENSORCOO_H
#define SPARSETENSORCOO_H

#include <vector>
#include <iostream>

class SparseTensorCOO {
    public:
        SparseTensorCOO(int dim1, int dim2, int dim3);
        
        void addElement(int i, int j, int k, double value);
        
        void print() const;

        void sort();

        void reorder(const std::vector<size_t>& idx);


        std::vector<int> coords1, coords2, coords3;
        std::vector<double> values;
        int dim1, dim2, dim3;
};

#endif