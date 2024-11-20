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
    uint64_t nmodes;      /// # modes
    uint64_t *ndims;      /// size of each mode, length nmodes
    uint64_t nnz;         /// # non-zeros
    sptIndexVector *inds; /// indices of each element, length [nmodes][nnz]
    sptValueVector values; /// non-zero values, length nnz
} sptSparseTensor;

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

void sptFreeSparseTensor(sptSparseTensor *tsr) {
    for (uint32_t i = 0; i < tsr->nmodes; ++i) {
        sptFreeIndexVector(&tsr->inds[i]);
    }
    free(tsr->inds);
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

int sptNewSparseTensorWithNnz(sptSparseTensor *tsr, uint32_t nmodes, const uint32_t ndims[], uint64_t nnz) {
    tsr->nmodes = nmodes;
    tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    if (!tsr->ndims) return -1;
    memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    tsr->nnz = nnz;
    tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    if (!tsr->inds) return -1;
    for (uint32_t i = 0; i < nmodes; ++i) {
        if (sptNewIndexVector(&tsr->inds[i], nnz, nnz) != 0) return -1;
    }
    if (sptNewValueVector(&tsr->values, nnz, nnz) != 0) return -1;
    return 0;
}

int newSpt(sptSparseTensor *tsr, uint32_t nmodes, const uint32_t ndims[], uint64_t nnz) {
    int result = sptNewSparseTensorWithNnz(tsr, nmodes, ndims, nnz);
    if (result != 0) return result;

    srand((unsigned)time(NULL));
    for (uint64_t i = 0; i < nnz; ++i) {
        for (uint32_t m = 0; m < nmodes; ++m) {
            tsr->inds[m].data[i] = rand() % ndims[m];
        }
        tsr->values.data[i] = (double)rand() / RAND_MAX;
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

int sptSparseTensorMulVector(sptSparseTensor *Y, sptSparseTensor *X, const sptValueVector *V, uint32_t mode) {
    uint64_t i;

    //#pragma omp parallel for if (USE_OPENMP) schedule(dynamic, 1000)
    #pragma omp parallel for if (USE_OPENMP) schedule(static)
    for (i = 0; i < X->nnz; ++i) {
        Y->values.data[i] = X->values.data[i] * V->data[X->inds[mode].data[i]];
    }

    return 0;
}

int sptSparseTensorMulMatrix(sptSparseTensor *Y, sptSparseTensor *X, const sptValueVector *M, uint32_t mode) {
    uint64_t i;

    #pragma omp parallel for if (USE_OPENMP) schedule(static)
    for (i = 0; i < X->nnz; ++i) {
        Y->values.data[i] = X->values.data[i] * M->data[X->inds[mode].data[i]];
    }

    return 0;
}

int loadTensorFromFile(sptSparseTensor *tensor, const char *filename, uint32_t nmodes, uint64_t nnz) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening file");
        return -1;
    }

    uint64_t nz = 0;
    while (nz < nnz) {
        for (uint32_t i = 0; i < nmodes; ++i) {
            if (fscanf(fp, "%" SCNu64, &tensor->inds[i].data[nz]) != 1) {
                fprintf(stderr, "Error reading index at line %" PRIu64 "\n", nz + 1);
                fclose(fp);
                return -1;
            }
        }
        if (fscanf(fp, "%lf", &tensor->values.data[nz]) != 1) {
            fprintf(stderr, "Error reading value at line %" PRIu64 "\n", nz + 1);
            fclose(fp);
            return -1;
        }
        nz++;
    }

    fclose(fp);
    return 0;
}


int sptSparseTensorMulScalar(sptSparseTensor *Y, sptSparseTensor *X, double scalar) {
    if (X->nmodes != Y->nmodes) {
        fprintf(stderr, "Tensor dimensions do not match for scalar multiplication.\n");
        return -1;
    }

    #pragma omp parallel for if (USE_OPENMP) schedule(static)
    for (uint64_t i = 0; i < X->nnz; ++i) {
        Y->values.data[i] = X->values.data[i] * scalar;
    }

    for (uint32_t m = 0; m < X->nmodes; ++m) {
        memcpy(Y->inds[m].data, X->inds[m].data, X->nnz * sizeof(uint64_t));
    }

    return 0;
}

int main() {
    sptSparseTensor X, Y, Z;
    sptValueVector M, V;

    // 预定义张量参数
    uint32_t nmodes = 5;  // 张量的模式数
    uint64_t nnz = 26021945;  // 非零元素数量
    uint32_t dims[] = {165427, 11374, 2, 100, 89}; // 张量的每个维度大小

    // 分配张量空间
    if (sptNewSparseTensorWithNnz(&X, nmodes, dims, nnz) != 0) {
        fprintf(stderr, "Failed to allocate tensor X\n");
        return -1;
    }

    // 从文件加载数据
    if (loadTensorFromFile(&X, "./vast-2015-mc1-5d.tns", nmodes, nnz) != 0) {
        fprintf(stderr, "Failed to load tensor data from file\n");
        sptFreeSparseTensor(&X);
        return -1;
    }

    // 生成随机矩阵和向量
    generateRandomMatrix(&M, dims[1], 10000);
    generateRandomVector(&V, dims[0]);

    // 分配结果张量空间
    if (sptNewSparseTensorWithNnz(&Y, nmodes - 1, dims, nnz) != 0 ||
        sptNewSparseTensorWithNnz(&Z, nmodes, dims, nnz) != 0) {
        fprintf(stderr, "Failed to allocate tensor Y\n");
        sptFreeSparseTensor(&X);
        sptFreeSparseTensor(&Z);
        sptFreeValueVector(&M);
        sptFreeValueVector(&V);
        return -1;
    }

    // OpenMP 线程设置

    //printf("threads used: %d\n", threads);



    clock_t start, end;

    // SpTTV 测试
    start = clock();
    sptSparseTensorMulVector(&Y, &X, &V, 0);
    end = clock();
    printf("SpTTV time: %f seconds\n", getElapsedTime(start, end));
    double throughput = X.nnz / getElapsedTime(start, end);
    printf("Throughput: %f non-zero elements per second\n", throughput);
    // SpTTM 测试
    start = clock();
    sptSparseTensorMulMatrix(&Y, &X, &M, 1);
    end = clock();
    printf("SpTTM time: %f seconds\n", getElapsedTime(start, end));
     throughput = X.nnz / getElapsedTime(start, end);
    printf("Throughput: %f non-zero elements per second\n", throughput);

    double scalar = 2.5;
    start = clock();
    sptSparseTensorMulScalar(&Z, &X, scalar);
    end = clock();
    printf("SpTTS time: %f seconds\n", getElapsedTime(start, end));

     throughput = X.nnz / getElapsedTime(start, end);
    printf("Throughput: %f non-zero elements per second\n", throughput);

    // 释放资源
    sptFreeSparseTensor(&X);
    sptFreeSparseTensor(&Y);
    sptFreeSparseTensor(&Z);
    sptFreeValueVector(&M);
    sptFreeValueVector(&V);

    return 0;
}
