//SparseTensorCOO.cpp
#include "SparseTensorCOO.h"
#include <algorithm>
#include <tuple>


    SparseTensorCOO::SparseTensorCOO(int dim1, int dim2, int dim3)
                    : dim1(dim1), dim2(dim2), dim3(dim3) {}

    void SparseTensorCOO::addElements(int i, int j, int k, double value) {
        //If the value is 0, return
        if (value == 0)
            return;

        //Add non-zero elements
        coords1.emplace_back(i);
        coords2.emplace_back(j);
        coords3.emplace_back(k);
        values.emplace_back(value);
    }

    void SparseTensorCOO::print() const {
        for (size_t i = 0; i < values.size(); ++i) {
        std::cout << "(" << coords1[i] << ", " << coords2[i] << ", "
                  << coords3[i] << ") -> " << values[i] << "\n";
        }
    }                  
