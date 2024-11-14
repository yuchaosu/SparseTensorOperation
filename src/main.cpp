#include "SparseTensorHiCOO.h"
#include "SparseTensorCOO.h"
#include "omp.h"
#include <map>
#include <tuple>
#include <stdexcept>
#include <algorithm>

SparseTensorCOO tensorElementWiseMultiply(const SparseTensorCOO& A,
                                          const SparseTensorCOO& B) {
    if (A.dim1 != B.dim1 || A.dim2 != B.dim2 || A.dim3 != B.dim3)
    {
        throw std::invalid_argument("Tensor Dimensions Do Not Match");
    }

    SparseTensorCOO C(A.dim1, A.dim2, A.dim3);

    SparseTensorCOO sortedA = A;
    SparseTensorCOO sortedB = B;
    sortedA.sort();
    sortedB.sort();

    size_t sizeA = sortedA.values.size();
    size_t sizeB = sortedB.values.size();

    // Parallel merging is complex due to data dependencies.
    // Here we attempt to parallelize using OpenMP tasks.

    #ifdef USE_OPENMP
    std::vector<int> C_coords1, C_coords2, C_coords3;
    std::vector<double> C_values;

    #pragma omp parallel
    {
        std::vector<int> local_coords1, local_coords2, local_coords3;
        std::vector<double> local_values;

        #pragma omp single nowait
        {
            size_t i = 0, j = 0;
            while (i < sizeA && j < sizeB) {
                int ai = sortedA.coords1[i];
                int aj = sortedA.coords2[i];
                int ak = sortedA.coords3[i];

                int bi = sortedB.coords1[j];
                int bj = sortedB.coords2[j];
                int bk = sortedB.coords3[j];

                if (ai == bi && aj == bj && ak == bk) {
                    double valA = sortedA.values[i];
                    double valB = sortedB.values[j];

                    #pragma omp task firstprivate(ai, aj, ak, valA, valB)
                    {
                        double val = valA * valB;

                        #pragma omp critical
                        {
                            C.addElement(ai, aj, ak, val);
                        }
                    }
                    ++i;
                    ++j;
                } else if (ai < bi ||
                           (ai == bi && (aj < bj || (aj == bj && ak < bk)))) {
                    ++i;
                } else {
                    ++j;
                }
            }
        }
    }
    #else
    size_t i = 0, j = 0;
    while (i < sizeA && j < sizeB)
    {
        int ai = sortedA.coords1[i];
        int aj = sortedA.coords2[i];
        int ak = sortedA.coords3[i];

        int bi = sortedB.coords1[j];
        int bj = sortedB.coords2[j];
        int bk = sortedB.coords3[j];

        if (ai == bi && aj == bj && ak == bk) {
            double val = sortedA.values[i] * sortedB.values[j];
            C.addElement(ai, aj, ak, val);
            ++i;
            ++j;
        } else if (ai < bi ||
                   (ai == bi && (aj < bj || (aj == bj && ak < bk)))) {
            ++i;
        } else {
            ++j;
        }
    }
    #endif

    return C;
}


SparseTensorHiCOO tensorElementWiseMultiply(const SparseTensorHiCOO& A,
                                            const SparseTensorHiCOO& B) {
    if (A.dim1 != B.dim1 || A.dim2 != B.dim2 ||
        A.dim3 != B.dim3 || A.blockSize != B.blockSize)
        throw std::invalid_argument(
            "Tensor dimensions or block sizes do not match"
        );
    
    SparseTensorHiCOO C(A.dim1, A.dim2, A.dim3, A.blockSize);

    // Create a map to quickly find elements in B
    std::map<std::tuple<int, int, int, int, int, int>, double> BElements;
    for (size_t idx = 0; idx < B.values.size(); ++idx) {
        auto key = std::make_tuple(
            B.blockCoord1[idx], B.blockCoord2[idx], B.blockCoord3[idx],
            B.elemCoord1[idx], B.elemCoord2[idx], B.elemCoord3[idx]);
        BElements[key] = B.values[idx];
    }

    // Perform element-wise multiplication
    std::vector<int> local_blockCoord1, local_blockCoord2, local_blockCoord3;
    std::vector<int> local_elemCoord1, local_elemCoord2, local_elemCoord3;
    std::vector<double> local_values;

    size_t n = A.values.size();

    #ifdef USE_OPENMP
    #pragma omp parallel
    {
        std::vector<int> thread_blockCoord1, thread_blockCoord2, thread_blockCoord3;
        std::vector<int> thread_elemCoord1, thread_elemCoord2, thread_elemCoord3;
        std::vector<double> thread_values;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < n; ++idx) {
            auto key = std::make_tuple(
                A.blockCoord1[idx], A.blockCoord2[idx], A.blockCoord3[idx],
                A.elemCoord1[idx], A.elemCoord2[idx], A.elemCoord3[idx]);

            auto it = BElements.find(key);
            if (it != BElements.end()) {
                double val = A.values[idx] * it->second;
                if (val != 0.0) {
                    thread_blockCoord1.push_back(A.blockCoord1[idx]);
                    thread_blockCoord2.push_back(A.blockCoord2[idx]);
                    thread_blockCoord3.push_back(A.blockCoord3[idx]);
                    thread_elemCoord1.push_back(A.elemCoord1[idx]);
                    thread_elemCoord2.push_back(A.elemCoord2[idx]);
                    thread_elemCoord3.push_back(A.elemCoord3[idx]);
                    thread_values.push_back(val);
                }
            }
        }

        #pragma omp critical
        {
            local_blockCoord1.insert(local_blockCoord1.end(), thread_blockCoord1.begin(), thread_blockCoord1.end());
            local_blockCoord2.insert(local_blockCoord2.end(), thread_blockCoord2.begin(), thread_blockCoord2.end());
            local_blockCoord3.insert(local_blockCoord3.end(), thread_blockCoord3.begin(), thread_blockCoord3.end());
            local_elemCoord1.insert(local_elemCoord1.end(), thread_elemCoord1.begin(), thread_elemCoord1.end());
            local_elemCoord2.insert(local_elemCoord2.end(), thread_elemCoord2.begin(), thread_elemCoord2.end());
            local_elemCoord3.insert(local_elemCoord3.end(), thread_elemCoord3.begin(), thread_elemCoord3.end());
            local_values.insert(local_values.end(), thread_values.begin(), thread_values.end());
        }
    }
    #else
    for (size_t idx = 0; idx < n; ++idx) {
        auto key = std::make_tuple(
            A.blockCoord1[idx], A.blockCoord2[idx], A.blockCoord3[idx],
            A.elemCoord1[idx], A.elemCoord2[idx], A.elemCoord3[idx]);

        auto it = BElements.find(key);
        if (it != BElements.end()) {
            double val = A.values[idx] * it->second;
            if (val != 0.0) {
                local_blockCoord1.push_back(A.blockCoord1[idx]);
                local_blockCoord2.push_back(A.blockCoord2[idx]);
                local_blockCoord3.push_back(A.blockCoord3[idx]);
                local_elemCoord1.push_back(A.elemCoord1[idx]);
                local_elemCoord2.push_back(A.elemCoord2[idx]);
                local_elemCoord3.push_back(A.elemCoord3[idx]);
                local_values.push_back(val);
            }
        }
    }
    #endif

    // Assign results to C
    C.blockCoord1 = std::move(local_blockCoord1);
    C.blockCoord2 = std::move(local_blockCoord2);
    C.blockCoord3 = std::move(local_blockCoord3);
    C.elemCoord1 = std::move(local_elemCoord1);
    C.elemCoord2 = std::move(local_elemCoord2);
    C.elemCoord3 = std::move(local_elemCoord3);
    C.values = std::move(local_values);

    return C;
}

SparseTensorCOO tensorTimesVector(const SparseTensorCOO& A,
                                  const std::vector<double>& v, int mode) {
    if (mode < 1 || mode > 3)
        throw std::invalid_argument("Mode must be between 1 and 3");
    

    int outDim1 = A.dim1, outDim2 = A.dim2, outDim3 = A.dim3;
    if (mode == 1)
        outDim1 = 1;
    else if (mode == 2)
        outDim2 = 1;
    else if (mode == 3)
        outDim3 = 1;

    SparseTensorCOO C(outDim1, outDim2, outDim3);

    size_t nnz = A.values.size();

    std::map<std::tuple<int, int, int>, double> outElements;

    #ifdef USE_OPENMP
    #pragma omp parallel
    {
        std::map<std::tuple<int, int, int>, double> localOutElements;

        #pragma omp for nowait schedule(static)
        for (size_t i = 0; i < nnz; ++i){
            double val = A.values[i];
            int idx1 = A.coords1[i];
            int idx2 = A.coords2[i];
            int idx3 = A.coords3[i];

            double vecVal = 0.0;
            if (mode == 1 && idx1 < static_cast<int>(v.size()))
                vecVal = v[idx1];
            else if (mode == 2 && idx2 < static_cast<int>(v.size()))
                vecVal = v[idx2];
            else if (mode == 3 && idx3 < static_cast<int>(v.size()))
                vecVal = v[idx3];
            else 
                continue;

            val *= vecVal;

            int outIdx1 = idx1, outIdx2 = idx2, outIdx3 = idx3;
            if (mode == 1)
                outIdx1 = 0;
            else if (mode == 2)
                outIdx2 = 0;
            else if (mode == 3)
                outIdx3 = 0;

            std::tuple<int, int, int> key = std::make_tuple(outIdx1, outIdx2, outIdx3);

            localOutElements[key] += val;
        }

        #pragma omp critical
        {
            for (const auto& kv : localOutElements)
            {
                outElements[kv.first] += kv.second;
            }
        }
    }
    #else
    for (size_t i = 0; i < nnz; ++i){
        double val = A.values[i];
        int idx1 = A.coords1[i];
        int idx2 = A.coords2[i];
        int idx3 = A.coords3[i];

        double vecVal = 0.0;
        if (mode == 1 && idx1 < static_cast<int>(v.size()))
            vecVal = v[idx1];
        else if (mode == 2 && idx2 < static_cast<int>(v.size()))
            vecVal = v[idx2];
        else if (mode == 3 && idx3 < static_cast<int>(v.size()))
            vecVal = v[idx3];
        else 
            continue;

        val *= vecVal;

        int outIdx1 = idx1, outIdx2 = idx2, outIdx3 = idx3;
        if (mode == 1)
            outIdx1 = 0;
        else if (mode == 2)
            outIdx2 = 0;
        else if (mode == 3)
            outIdx3 = 0;

        std::tuple<int, int, int> key = std::make_tuple(outIdx1, outIdx2, outIdx3);

        outElements[key] += val;
    }
    #endif

    for (const auto& kv : outElements)
    {
        int i = std::get<0>(kv.first);
        int j = std::get<1>(kv.first);
        int k = std::get<2>(kv.first);

        double val = kv.second;

        C.addElement(i, j, k, val);
    }
    
    return C;
}


SparseTensorHiCOO tensorTimesVector(const SparseTensorHiCOO& A,
                                    const std::vector<double>& v, int mode) {
    if (mode < 1 || mode > 3) {
        throw std::invalid_argument("Mode must be between 1 and 3.");
    }

    int m1 = A.dim1, m2 = A.dim2, m3 = A.dim3;
    int out_m1 = m1, out_m2 = m2, out_m3 = m3;
    if (mode == 1)
        out_m1 = 1;
    else if (mode == 2)
        out_m2 = 1;
    else if (mode == 3)
        out_m3 = 1;

    SparseTensorHiCOO C(out_m1, out_m2, out_m3, A.blockSize);

    std::map<std::tuple<int, int, int, int, int, int>, double> out_elements;

    size_t nnz = A.values.size();

    #ifdef USE_OPENMP
    #pragma omp parallel
    {
        std::map<std::tuple<int, int, int, int, int, int>, double> local_out_elements;

        #pragma omp for nowait schedule(static)
        for (size_t idx = 0; idx < nnz; ++idx) {
            double val = A.values[idx];

            int blockI = A.blockCoord1[idx];
            int blockJ = A.blockCoord2[idx];
            int blockK = A.blockCoord3[idx];

            int elemI = A.elemCoord1[idx];
            int elemJ = A.elemCoord2[idx];
            int elemK = A.elemCoord3[idx];

            int global_i = blockI * A.blockSize + elemI;
            int global_j = blockJ * A.blockSize + elemJ;
            int global_k = blockK * A.blockSize + elemK;

            double vec_val = 0.0;
            if (mode == 1 && global_i < static_cast<int>(v.size()))
                vec_val = v[global_i];
            else if (mode == 2 && global_j < static_cast<int>(v.size()))
                vec_val = v[global_j];
            else if (mode == 3 && global_k < static_cast<int>(v.size()))
                vec_val = v[global_k];
            else
                continue; // Index out of bounds

            val *= vec_val;

            // Reduce along the mode
            int out_global_i = global_i, out_global_j = global_j, out_global_k = global_k;
            if (mode == 1)
                out_global_i = 0;
            else if (mode == 2)
                out_global_j = 0;
            else if (mode == 3)
                out_global_k = 0;

            int out_block_i = out_global_i / A.blockSize;
            int out_block_j = out_global_j / A.blockSize;
            int out_block_k = out_global_k / A.blockSize;

            int out_elem_i = out_global_i % A.blockSize;
            int out_elem_j = out_global_j % A.blockSize;
            int out_elem_k = out_global_k % A.blockSize;

            auto key = std::make_tuple(
                out_block_i, out_block_j, out_block_k,
                out_elem_i, out_elem_j, out_elem_k);

            // Accumulate
            local_out_elements[key] += val;
        }

        // Merge local results into global map
        #pragma omp critical
        {
            for (const auto& kv : local_out_elements) {
                out_elements[kv.first] += kv.second;
            }
        }
    }
    #else
    for (size_t idx = 0; idx < nnz; ++idx) {
        double val = A.values[idx];

        int blockI = A.blockCoord1[idx];
        int blockJ = A.blockCoord2[idx];
        int blockK = A.blockCoord3[idx];

        int elemI = A.elemCoord1[idx];
        int elemJ = A.elemCoord2[idx];
        int elemK = A.elemCoord3[idx];

        int global_i = blockI * A.blockSize + elemI;
        int global_j = blockJ * A.blockSize + elemJ;
        int global_k = blockK * A.blockSize + elemK;

        double vec_val = 0.0;
        if (mode == 1 && global_i < static_cast<int>(v.size()))
            vec_val = v[global_i];
        else if (mode == 2 && global_j < static_cast<int>(v.size()))
            vec_val = v[global_j];
        else if (mode == 3 && global_k < static_cast<int>(v.size()))
            vec_val = v[global_k];
        else
            continue; // Index out of bounds

        val *= vec_val;

        // Reduce along the mode
        int out_global_i = global_i, out_global_j = global_j, out_global_k = global_k;
        if (mode == 1)
            out_global_i = 0;
        else if (mode == 2)
            out_global_j = 0;
        else if (mode == 3)
            out_global_k = 0;

        int out_block_i = out_global_i / A.blockSize;
        int out_block_j = out_global_j / A.blockSize;
        int out_block_k = out_global_k / A.blockSize;

        int out_elem_i = out_global_i % A.blockSize;
        int out_elem_j = out_global_j % A.blockSize;
        int out_elem_k = out_global_k % A.blockSize;

        auto key = std::make_tuple(
            out_block_i, out_block_j, out_block_k,
            out_elem_i, out_elem_j, out_elem_k);

        // Accumulate
        out_elements[key] += val;
    }
    #endif

    // Convert map to HiCOOTensor
    for (const auto& kv : out_elements) {
        int block_i = std::get<0>(kv.first);
        int block_j = std::get<1>(kv.first);
        int block_k = std::get<2>(kv.first);
        int elem_i = std::get<3>(kv.first);
        int elem_j = std::get<4>(kv.first);
        int elem_k = std::get<5>(kv.first);
        double val = kv.second;
        if (val != 0.0) {
            C.blockCoord1.emplace_back(block_i);
            C.blockCoord2.emplace_back(block_j);
            C.blockCoord3.emplace_back(block_k);
            C.elemCoord1.emplace_back(elem_i);
            C.elemCoord2.emplace_back(elem_j);
            C.elemCoord3.emplace_back(elem_k);
            C.values.emplace_back(val);
        }
    }

    return C;
}


int main(int argc, char *argv[]) {

    // Using SparseTensorCOO (same as before)
    {
        // Define two sparse tensors A and B
        SparseTensorCOO A(4, 4, 4);
        SparseTensorCOO B(4, 4, 4);

        // Add elements to tensor A
        A.addElement(0, 1, 2, 3.0);
        A.addElement(1, 2, 3, 4.0);
        A.addElement(2, 3, 0, 5.0);

        // Add elements to tensor B
        B.addElement(0, 1, 2, 2.0);
        B.addElement(1, 2, 3, 3.0);
        B.addElement(3, 0, 1, 4.0);

        // Perform element-wise multiplication
        SparseTensorCOO C = tensorElementWiseMultiply(A, B);

        std::cout << "Result of TEW (Element-wise Multiplication) using SparseTensorCOO:\n";
        C.print();

        // Define a vector for TTV operation
        std::vector<double> v = {1.0, 2.0, 3.0, 4.0};

        // Perform TTV along mode 1
        SparseTensorCOO D = tensorTimesVector(A, v, 1);

        std::cout << "\nResult of TTV along mode 1 using SparseTensorCOO:\n";
        D.print();
    }

    // Using HiCOOTensor
    {
        // Define two HiCOO tensors A and B with block size 2
        SparseTensorHiCOO A(4, 4, 4, 2);
        SparseTensorHiCOO B(4, 4, 4, 2);

        // Add elements to tensor A
        A.addElement(0, 1, 2, 3.0);
        A.addElement(1, 2, 3, 4.0);
        A.addElement(2, 3, 0, 5.0);

        // Add elements to tensor B
        B.addElement(0, 1, 2, 2.0);
        B.addElement(1, 2, 3, 3.0);
        B.addElement(3, 0, 1, 4.0);

        // Perform element-wise multiplication
        SparseTensorHiCOO C = tensorElementWiseMultiply(A, B);

        std::cout << "\nResult of TEW (Element-wise Multiplication) using HiCOOTensor:\n";
        C.print();

        // Define a vector for TTV operation
        std::vector<double> v = {1.0, 2.0, 3.0, 4.0};

        // Perform TTV along mode 1
        SparseTensorHiCOO D = tensorTimesVector(A, v, 1);

        std::cout << "\nResult of TTV along mode 1 using HiCOOTensor:\n";
        D.print();
    }

    return 0;


}