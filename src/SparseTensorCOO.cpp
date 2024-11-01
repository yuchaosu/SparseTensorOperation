//SparseTensorCOO.cpp
#include "SparseTensorCOO.h"
#include <algorithm>
#include <tuple>


    SparseTensorCOO::SparseTensorCOO(int dim1, int dim2, int dim3)
                    : dim1(dim1), dim2(dim2), dim3(dim3) {}

    void SparseTensorCOO::addElement(int i, int j, int k, double value) {
        //If the value is 0, return
        if (value == 0.0)
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

    void SparseTensorCOO::sort() {
        std::vector<size_t> idx(values.size());
        for (size_t i = 0; i < idx.size(); ++i)
            idx[i] = i;
        
        //Comparator Overloading
        auto comparator = [this](size_t a, size_t b) {
            if (coords1[a] != coords1[b])
                return coords1[a] < coords1[b];
            if (coords2[a] != coords2[b])
                return coords2[a] < coords2[b];
            return coords3[a] < coords3[b];
        };
        
        std::sort(idx.begin(), idx.end(), comparator);

        reorder(idx);

    }

    void SparseTensorCOO::reorder(const std::vector<size_t>& idx) {
        std::vector<int> c1(coords1.size()), c2(coords2.size()), c3(coords3.size());
        std::vector<double> vals(values.size());

    
        for (size_t i = 0; i < idx.size(); ++i) {
            c1[i] = coords1[idx[i]];
            c2[i] = coords2[idx[i]];
            c3[i] = coords3[idx[i]];
            vals[i] = values[idx[i]];
        }

        // Assign back
        coords1 = std::move(c1);
        coords2 = std::move(c2);
        coords3 = std::move(c3);
        values = std::move(vals);

    }          
