#include <cstdio>

#define N 1000

__global__ void add(const int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int main() {
    int ha[N];
    int hb[N];
    int *da;
    int *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));
    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }

    int a = 1;
    printf("%d", a);

    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);
    add<<<N, 1>>>(da, db);
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }
    cudaFree(da);
    cudaFree(db);
    return 0;
}