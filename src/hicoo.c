#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <omp.h> // OpenMP 支持

typedef struct {
    uint64_t len;   /// length
    uint64_t cap;   /// capacity
    double *data;   /// data
} sptValueVector;

typedef struct {
    uint64_t len;   /// length
    uint64_t cap;   /// capacity
    uint64_t *data; /// data
} sptIndexVector;

typedef struct {
    uint64_t len;   /// length
    uint64_t cap;   /// capacity
    uint64_t *data; /// data
} sptBlockIndexVector;

typedef struct {
    uint64_t nmodes;      /// # modes
    uint64_t *ndims;      /// size of each mode, length nmodes
    uint64_t nnz;         /// # non-zeros
    uint16_t sb_bits;     /// block size by nnz
    sptIndexVector bptr;  /// block pointers to all nonzeros
    sptBlockIndexVector *binds; /// block indices within each group
    sptIndexVector *einds;      /// element indices within each block 
    sptValueVector values;      /// non-zero values, length nnz
} sptSparseTensorHiCOO;

void sptFreeIndexVector(sptIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}

void sptFreeValueVector(sptValueVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}

void sptFreeBlockIndexVector(sptBlockIndexVector *vec) {
    free(vec->data);
    vec->len = 0;
    vec->cap = 0;
}

void sptFreeSparseTensorHiCOO(sptSparseTensorHiCOO *tsr) {
    sptFreeIndexVector(&tsr->bptr);
    for (uint32_t i = 0; i < tsr->nmodes; ++i) {
        sptFreeBlockIndexVector(&tsr->binds[i]);
        sptFreeIndexVector(&tsr->einds[i]);
    }
    free(tsr->binds);
    free(tsr->einds);
    free(tsr->ndims);
    sptFreeValueVector(&tsr->values);
}

int sptNewIndexVector(sptIndexVector *vec, uint64_t len, uint64_t cap) {
    vec->len = len;
    vec->cap = cap < 2 ? 2 : cap;
    vec->data = malloc(vec->cap * sizeof *vec->data);
    if (!vec->data) return -1;
    memset(vec->data, 0, vec->cap * sizeof *vec->data);
    return 0;
}

int sptNewValueVector(sptValueVector *vec, uint64_t len, uint64_t cap) {
    vec->len = len;
    vec->cap = cap < 2 ? 2 : cap;
    vec->data = malloc(vec->cap * sizeof *vec->data);
    if (!vec->data) return -1;
    memset(vec->data, 0, vec->cap * sizeof *vec->data);
    return 0;
}

int sptNewBlockIndexVector(sptBlockIndexVector *vec, uint64_t len, uint64_t cap) {
    vec->len = len;
    vec->cap = cap < 2 ? 2 : cap;
    vec->data = malloc(vec->cap * sizeof *vec->data);
    if (!vec->data) return -1;
    memset(vec->data, 0, vec->cap * sizeof *vec->data);
    return 0;
}

int sptNewSparseTensorHiCOO(sptSparseTensorHiCOO *tsr, uint32_t nmodes, const uint32_t ndims[], uint64_t nnz, uint16_t sb_bits) {
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    if (!tsr->ndims) return -1;
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = nnz;
    tsr->sb_bits = sb_bits;

    uint64_t nblocks = (nnz + (1UL << sb_bits) - 1) >> sb_bits;
    if (sptNewIndexVector(&tsr->bptr, nblocks + 1, nblocks + 1) != 0) return -1;

    tsr->binds = malloc(nmodes * sizeof *tsr->binds);
    tsr->einds = malloc(nmodes * sizeof *tsr->einds);
    if (!tsr->binds || !tsr->einds) return -1;

    for (uint32_t i = 0; i < nmodes; ++i) {
        if (sptNewBlockIndexVector(&tsr->binds[i], nblocks, nblocks) != 0) return -1;
        if (sptNewIndexVector(&tsr->einds[i], nnz, nnz) != 0) return -1;
    }

    if (sptNewValueVector(&tsr->values, nnz, nnz) != 0) return -1;

    return 0;
}

int newHiCOOSpt(sptSparseTensorHiCOO *tsr, uint32_t nmodes, const uint32_t ndims[], uint64_t nnz, uint16_t sb_bits) {
    int result = sptNewSparseTensorHiCOO(tsr, nmodes, ndims, nnz, sb_bits);
    if (result != 0) return result;

    srand((unsigned)time(NULL));
    uint64_t nblocks = (nnz + (1UL << sb_bits) - 1) >> sb_bits;

    for (uint64_t b = 0; b < nblocks; ++b) {
        tsr->bptr.data[b] = b << sb_bits;
        for (uint32_t m = 0; m < nmodes; ++m) {
            tsr->binds[m].data[b] = rand() % ndims[m];
        }
    }

    for (uint64_t i = 0; i < nnz; ++i) {
        for (uint32_t m = 0; m < nmodes; ++m) {
            tsr->einds[m].data[i] = rand() % (1UL << sb_bits);
        }
        tsr->values.data[i] = (double)rand() / RAND_MAX;
    }

    return 0;
}

int sptSparseTensorHiCOOMulVector(sptSparseTensorHiCOO *Y, sptSparseTensorHiCOO *X, const sptValueVector *V, uint32_t mode) {
    uint64_t i;

    #pragma omp parallel for if (USE_OPENMP) schedule(static)
    for (i = 0; i < X->nnz; ++i) {
        Y->values.data[i] = X->values.data[i] * V->data[X->einds[mode].data[i]];
    }

    return 0;
}

int sptSparseTensorHiCOOMulMatrix(sptSparseTensorHiCOO *Y, sptSparseTensorHiCOO *X, const sptValueVector *M, uint32_t mode) {
    uint64_t i;

    #pragma omp parallel for if (USE_OPENMP) schedule(static)
    for (i = 0; i < X->nnz; ++i) {
        Y->values.data[i] = X->values.data[i] * M->data[X->einds[mode].data[i]];
    }

    return 0;
}

void generateRandomMatrix(sptValueVector *matrix, uint64_t rows, uint64_t cols) {
    sptNewValueVector(matrix, rows * cols, rows * cols);
    for (uint64_t i = 0; i < rows * cols; ++i) {
        matrix->data[i] = (double)rand() / RAND_MAX;
    }
}

void generateRandomVector(sptValueVector *vector, uint64_t len) {
    sptNewValueVector(vector, len, len);
    for (uint64_t i = 0; i < len; ++i) {
        vector->data[i] = (double)rand() / RAND_MAX;
    }
}

double getElapsedTime(clock_t start, clock_t end) {
    return ((double)(end - start)) / CLOCKS_PER_SEC;
}

int sptSparseTensorHiCOOMulScalar(sptSparseTensorHiCOO *Y, sptSparseTensorHiCOO *X, double scalar) {
    if (X->nmodes != Y->nmodes) {
        fprintf(stderr, "Tensor dimensions do not match for scalar multiplication.\n");
        return -1;
    }

    #pragma omp parallel for if (USE_OPENMP) schedule(static)
    for (uint64_t i = 0; i < X->nnz; ++i) {
        Y->values.data[i] = X->values.data[i] * scalar;
    }

    memcpy(Y->bptr.data, X->bptr.data, X->bptr.len * sizeof(uint64_t));

    for (uint32_t m = 0; m < X->nmodes; ++m) {
        memcpy(Y->binds[m].data, X->binds[m].data, X->binds[m].len * sizeof(uint64_t));
        memcpy(Y->einds[m].data, X->einds[m].data, X->einds[m].len * sizeof(uint64_t));
    }

    return 0;
}

int main() {
    sptSparseTensorHiCOO X, Y, Z;
    sptValueVector M, V;

    uint32_t nmodes = 5;
    uint64_t nnz = 26021945;
    uint32_t dims[] = {165427, 11374, 2, 100, 89};
    uint16_t sb_bits = 10;

    if (newHiCOOSpt(&X, nmodes, dims, nnz, sb_bits) != 0) {
        fprintf(stderr, "Failed to allocate HiCOO tensor X\n");
        return -1;
    }


    // 生成随机矩阵和向量
    generateRandomMatrix(&M, dims[1], 10000);
    generateRandomVector(&V, dims[0]);

    // 分配结果张量空间
    if (sptNewSparseTensorHiCOO(&Y, nmodes, dims, nnz, sb_bits) != 0 ||
        sptNewSparseTensorHiCOO(&Z, nmodes, dims, nnz, sb_bits) != 0) {
        fprintf(stderr, "Failed to allocate tensor Y\n");
        sptFreeSparseTensorHiCOO(&X);
        sptFreeValueVector(&M);
        sptFreeValueVector(&V);
        return -1;
    }

    // OpenMP 线程设置

    //printf("threads used: %d\n", threads);



    clock_t start, end;

    // SpTTV 测试
    start = clock();
    sptSparseTensorHiCOOMulVector(&Y, &X, &V, 0);
    end = clock();
    printf("SpTTV time: %f seconds\n", getElapsedTime(start, end));
    double throughput = X.nnz / getElapsedTime(start, end);
    printf("Throughput: %f non-zero elements per second\n", throughput);
    // SpTTM 测试
    start = clock();
    sptSparseTensorHiCOOMulMatrix(&Y, &X, &M, 1);
    end = clock();
    printf("SpTTM time: %f seconds\n", getElapsedTime(start, end));
     throughput = X.nnz / getElapsedTime(start, end);
    printf("Throughput: %f non-zero elements per second\n", throughput);

    double scalar = 2.5;
    start = clock();
    sptSparseTensorHiCOOMulScalar(&Z, &X, scalar);
    end = clock();
    printf("SpTTS time: %f seconds\n", getElapsedTime(start, end));

     throughput = X.nnz / getElapsedTime(start, end);
    printf("Throughput: %f non-zero elements per second\n", throughput);

    // 释放资源
    sptFreeSparseTensorHiCOO(&X);
    sptFreeSparseTensorHiCOO(&Y);
    sptFreeSparseTensorHiCOO(&Z);
    sptFreeValueVector(&M);
    sptFreeValueVector(&V);
}
