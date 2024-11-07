//SparseTensorHiCOO.cpp
#include "SparseTensorHiCOO.h"
#include <algorithm>
#include <tuple>

SparseTensorHiCOO::SparseTensorHiCOO(int dim1, int dim2, int dim3, int blockSize)
                    : dim1(dim1), dim2(dim2), dim3(dim3), blockSize(blockSize) {}


void SparseTensorHiCOO::addElement(int i, int j, int k, double val) {
    if (val == 0.0)
        return;
    int blockI = i / blockSize;
    int blockJ = j / blockSize;
    int blockK = k / blockSize;

    int elemI = i % blockSize;
    int elemJ = j % blockSize;
    int elemK = k % blockSize;

    blockCoord1.emplace_back(blockI);
    blockCoord2.emplace_back(blockJ);
    blockCoord3.emplace_back(blockK);

    elemCoord1.emplace_back(elemI);
    elemCoord2.emplace_back(elemJ);
    elemCoord3.emplace_back(elemK);

    values.emplace_back(val);
}

void SparseTensorHiCOO::print() const {
    std::cout << "HiCOO Tensor Entries:\n";
    for (size_t idx = 0; idx < values.size(); ++idx) {
        int i = blockCoord1[idx] * blockSize + elemCoord1[idx];
        int j = blockCoord2[idx] * blockSize + elemCoord2[idx];
        int k = blockCoord3[idx] * blockSize + elemCoord3[idx];
        std::cout << "(" << i << ", " << j << ", " << k << ") -> " << values[idx] << "\n";
    }
}
