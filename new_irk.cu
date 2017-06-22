
#include <typeinfo> // for usage of C++ typeid
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>

#include "cusolverSp.h"
#include <cusolverDn.h>
#include "cusolverSp.h"
#include "cublas_v2.h"

#include <helper_cuda.h>
#include "cusparse_v2.h"
#include <iostream>
using namespace std;
  
//profiling the code
#define GPUERRCHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define TIME_INDIVIDUAL_LIBRARY_CALLS

#define DBICGSTAB_MAX_ULP_ERR   100
#define DBICGSTAB_EPS           1.E-14f //9e-2

#define CLEANUP()                       \
do {                                    \
    if (x)          free (x);           \
    if (f)          free (f);           \
    if (r)          free (r);           \
    if (rw)         free (rw);          \
    if (p)          free (p);           \
    if (pw)         free (pw);          \
    if (s)          free (s);           \
    if (t)          free (t);           \
    if (v)          free (v);           \
    if (tx)         free (tx);          \
    if (Aval)       free(Aval);         \
    if (AcolsIndex) free(AcolsIndex);   \
    if (ArowsIndex) free(ArowsIndex);   \
    if (Mval)       free(Mval);         \
    if (devPtrX)    checkCudaErrors(cudaFree (devPtrX));                    \
    if (devPtrF)    checkCudaErrors(cudaFree (devPtrF));                    \
    if (devPtrR)    checkCudaErrors(cudaFree (devPtrR));                    \
    if (devPtrRW)   checkCudaErrors(cudaFree (devPtrRW));                   \
    if (devPtrP)    checkCudaErrors(cudaFree (devPtrP));                    \
    if (devPtrS)    checkCudaErrors(cudaFree (devPtrS));                    \
    if (devPtrT)    checkCudaErrors(cudaFree (devPtrT));                    \
    if (devPtrV)    checkCudaErrors(cudaFree (devPtrV));                    \
    if (devPtrAval) checkCudaErrors(cudaFree (devPtrAval));                 \
    if (devPtrAcolsIndex) checkCudaErrors(cudaFree (devPtrAcolsIndex));     \
    if (devPtrArowsIndex) checkCudaErrors(cudaFree (devPtrArowsIndex));     \
    if (devPtrMval)       checkCudaErrors(cudaFree (devPtrMval));           \
    if (stream)           checkCudaErrors(cudaStreamDestroy(stream));       \
    if (cublasHandle)     checkCudaErrors(cublasDestroy(cublasHandle));     \
    if (cusparseHandle)   checkCudaErrors(cusparseDestroy(cusparseHandle)); \
    fflush (stdout);                                    \
} while (0)

using namespace std;
#define BLOCK_SIZE 32

extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }


void gpu_blas_mmul(const double *A, const double*B, double *C, const int m, const int k, const int n) {
  int lda=m,ldb=k,ldc=m;
  const double alf = 1;
  const double bet = 0;
  const double *alpha = &alf;
  const double *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Do the actual multiplication
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  // Destroy the handle
  cublasDestroy(handle);
}

__global__ void copy_kernel(const double * __restrict d_in1, double * __restrict d_out1, const double * __restrict d_in2, double * __restrict d_out2, const int M, const int N) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < N) && (j < N)) {
        d_out1[j * N + i] = d_in1[j * M + i];
        d_out2[j * N + i] = d_in2[j * M + i];
    }
}

int bicg(double h_A1_dense[] ,double x0[] , double B[] , double M[] , int  rmaxit , double rtol, int n ){
double *r0 = (double*)malloc(n*n*sizeof(double ));
double *r0_tilde = (double*)malloc(n*n*sizeof(*r0_tilde));
double *r = (double*)malloc(n*n*sizeof(double));
double *r_temp_tilde = (double*)malloc(n*n*sizeof(*r_temp_tilde));
double *z = (double*)malloc(n*sizeof(*z));
double *z_tilde = (double*)malloc(n*sizeof(*z_tilde));
double *p = (double*)malloc(n*sizeof(*p));
double *p_tilde = (double*)malloc(n*sizeof(*p_tilde));
double *q = (double*)malloc(n*sizeof(*q));
double *q_tilde = (double*)malloc(n*sizeof(*q_tilde));
double normr=0;double normb=0; int tot_iter=0;
double resid;
double tol,betta;
int work_size=0;
double tempp=0;
// r=b-Ax;
double *h_A_trans= (double*)malloc(n*n*sizeof(double));
for(int i=0;i<n;i++)for(int j=0;j<n;j++){
  h_A_trans[i*n+j]=h_A1_dense[j*n+i];
}



// C=Ax
cublasHandle_t Blas_handle;
cublasCreate(&Blas_handle); 
cusolverDnHandle_t solver_handle;
cusolverDnCreate(&solver_handle);

int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;


double *h_C=(double*)malloc(n*sizeof(*h_C));
nr_rows_A=n;nr_cols_A=n;nr_rows_B=n;nr_cols_B=1;nr_rows_C=n;nr_cols_C=1;

double *d_A1;cudaMalloc(&d_A1,nr_rows_A * nr_cols_A * sizeof(*d_A1));
double *d_B;cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(*d_B));
double *d_C;cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(*d_C));
//int d_A1_ColIndices ;cudaMalloc(&d_A1_ColIndices, nnzA * sizeof(*d_A1_ColIndices));
cudaMemcpy(d_A1,h_A1_dense,nr_rows_A * nr_cols_A * sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(d_B,B,nr_rows_B * nr_cols_B * sizeof(double),cudaMemcpyHostToDevice);

gpu_blas_mmul(d_A1, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(double),cudaMemcpyDeviceToHost);
for(int i=0;i<n;i++){r[i]=B[i]-h_C[i];r_temp_tilde[i]=r[i];r0[i]=r[i];r0_tilde[i]=r[i];    normr+=r[i]*r[i];normb+=B[i]*B[i];}
normr=sqrt(normr);normb=sqrt(normb);
if(!normb)normb=1;
resid=normr/normb;
if(resid<rtol){
  tol=resid;
  tot_iter=0;
  return 0;

}  
double rho1=0;double rho2=0;

double *d_M; cudaMalloc(&d_M, nr_rows_A*nr_cols_A*sizeof(*d_M));
cudaMemcpy(d_M, M, nr_rows_A * nr_cols_A * sizeof(double),cudaMemcpyHostToDevice );
// CUDA QR initialisation

double *d_TAU; cudaMalloc((void **)&d_TAU,min(nr_cols_A,nr_rows_A)*sizeof(double) );
cusolverDnDgeqrf_bufferSize(solver_handle,nr_rows_A,nr_cols_A,d_M,nr_rows_A,&work_size);
double *work; cudaMalloc(&work ,work_size*sizeof(double));

// CUDA GERF exec.
int *dev_info; cudaMalloc(&dev_info, sizeof(int));

cusolverDnDgeqrf(solver_handle, nr_rows_A, nr_cols_A, d_M,nr_rows_A, d_TAU, work, work_size, dev_info);
//cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *TAU, double *Workspace, int Lwork, int *devInfo

int dev_info_h=0;cudaMemcpy(&dev_info_h,dev_info, sizeof(int ), cudaMemcpyDeviceToHost);
if(dev_info_h!=0 )cout<<"uncussful exec of GERf"<<endl;
double *h_Q =(double *)malloc(nr_rows_A*nr_cols_A*sizeof(double));
memset(h_Q, 0,  nr_rows_A*nr_cols_A*sizeof(double));
for(int i=0;i<nr_rows_A;i++)h_Q[i+i*nr_rows_A]=1;
double *d_Q; cudaMalloc(&d_Q, nr_rows_A*nr_cols_A*sizeof(double));
cudaMemcpy(d_Q,h_Q,nr_rows_A*nr_cols_A*sizeof(double),cudaMemcpyHostToDevice);
//CUDA QR execution
cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT,CUBLAS_OP_N,nr_rows_A,nr_cols_A,
  min(nr_rows_A,nr_cols_A), d_M, nr_rows_A,d_TAU,d_Q,nr_rows_A,work,work_size,dev_info);


// 






  double *d_r; cudaMalloc(&d_r, nr_rows_A*nr_cols_A*sizeof(double));

   double *d_p; cudaMalloc(&d_p, n*sizeof(double ));

   double *d_A1_trans;cudaMalloc(&d_A1_trans, n*n*sizeof(double ));
  double *d_R ; cudaMalloc(&d_R,nr_cols_A*nr_cols_A*sizeof(double));
  double  *h_Bl= (double *)malloc(nr_cols_A*nr_cols_A*sizeof(double));
  double  *d_Bl; cudaMalloc(&d_Bl,nr_cols_A*nr_cols_A*sizeof(double));
double *d_qq; cudaMalloc(&d_qq, n*sizeof(double ));

for(int i=0;i<rmaxit; i++){
  // solve Mz=r;//block solve
  for(int j=0;j<n;j++){r[j]=r0[j];r_temp_tilde[j]=r0_tilde[j];}
  cudaMemcpy(d_r,r,nr_rows_A*nr_cols_A*sizeof(double),cudaMemcpyHostToDevice);
  cusolverDnDormqr(solver_handle,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,nr_rows_A,nr_cols_A,min(nr_cols_A,nr_rows_A),
    d_M,nr_rows_A,d_TAU,d_r,nr_rows_A,work,work_size,dev_info);


  // at this point d_r contains the element Q^Tr 
  // only the first coloumn if d_r makes sense ...
  
  cudaMemcpy(r,d_r,nr_rows_A*nr_cols_A*sizeof(double ), cudaMemcpyDeviceToHost);
  dim3 Grid(iDivUp(nr_cols_A,BLOCK_SIZE),iDivUp(nr_cols_A,BLOCK_SIZE));
  dim3 Block(BLOCK_SIZE,BLOCK_SIZE);
  copy_kernel<<<Grid, Block>>>(d_M,d_R,d_r,d_Bl,nr_rows_A,nr_cols_A);



  // solving an upper triangular linear system
  const double alpha =1;
   cublasDtrsm(Blas_handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,nr_cols_A,nr_cols_A,
                &alpha,d_R,nr_cols_A,d_Bl,nr_cols_A);
   cudaMemcpy(h_Bl,d_Bl,nr_cols_A*nr_cols_A*sizeof(double), cudaMemcpyDeviceToHost);
    for(int j =0;j<n;j++ )z[j]=h_Bl[j];
      // solve Mtz=r_temp_tilde 
    for(int j =0;j<n;j++)z_tilde[j]=z[j];
    rho1=0;
    for(int j=0;j<n;j++ )rho1+=z[j]*r0_tilde[j];
    if(rho1==0){
      tol =normr/normb;
      tot_iter=i;
      return 1;
    }  

    if(i==0){
      for(int j=0;j<n;j++){
        p[j]=z[j];p_tilde[j]=z_tilde[j];
      }
    }
    else{
     betta=rho1/rho2;
     for(int j=0;j<n;j++) {
      p[j]=betta*p[j]+z[j];
      p_tilde[j]=betta*p_tilde[j]+z_tilde[j];

     }

    // q=Ap and q_tilde =At*p
   
   cudaMemcpy(d_p, p, n* sizeof(double),cudaMemcpyHostToDevice );

   
   
   gpu_blas_mmul(d_A1, d_p, d_qq, nr_rows_A, nr_cols_A, nr_cols_B);

   cudaMemcpy(q, d_qq, n* sizeof(double),cudaMemcpyDeviceToHost ); 
   cudaMemcpy(d_A1_trans,h_A_trans,n*n*sizeof(double),cudaMemcpyHostToDevice);
   cudaMemcpy(d_p, p_tilde, n* sizeof(double),cudaMemcpyHostToDevice );

   gpu_blas_mmul(d_A1_trans, d_p, d_qq, nr_rows_A, nr_cols_A, nr_cols_B);
   cudaMemcpy(q_tilde, d_qq, n* sizeof(double),cudaMemcpyDeviceToHost );

   tempp=0;
   for(int j=0;j<n;j++)tempp+=p_tilde[j]*q[j];
   double alphaa = -1*rho1/tempp;
   for(int j=0;j<n;j++){
       x0[j]+=alphaa*p[j];
       r0[j]-=alphaa*q[j];
       r0_tilde[j]-=alphaa*q_tilde[j];
      
      }
      rho2=rho1;
      normr=0;
      for(int j=0;j<n;j++)normr+=r0[j]*r0[j];
        normr=sqrt(normr);resid=normr/normb;
      if(resid<rtol){
          tol=resid;
          tot_iter=i;
          return 3;
      }


    }
    


  
}
tol =resid;

cudaFree(d_A1);
cudaFree(d_B);
cudaFree(d_C);
cudaFree(d_M);

cudaFree(d_Q);
cudaFree(d_qq);
cudaFree(d_TAU);
cudaFree(dev_info);

cudaFree(d_r);
cudaFree(d_R);
cudaFree(work);
cudaFree(d_Bl);


cudaFree(d_A1_trans);
free(r0);
free(r0_tilde);
free(r_temp_tilde);
free(r);
free(z);
free(z_tilde);
free(q_tilde);
free(q);
free(p);
free(p_tilde);
free(h_A_trans);
free(h_Q);
free(h_C);
free(h_Bl);
cublasDestroy(Blas_handle);
cusolverDnDestroy(solver_handle);



return 1;





}

int main(){
const int n=10;const int r=4;
const int N=n;

int rmaxit,max_iter,irka_iter;
double rtol,itol;

double *x0 = (double*)malloc(n*sizeof(double));    
cout<<"I am Here"<<endl;;
double *x0_tilde= (double*)malloc(n*sizeof(*x0_tilde));  
double *A=(double*)malloc(n*n*sizeof(double));
double *B=(double*)malloc(n*sizeof(*B));
double *C=(double*)malloc(n*sizeof(*C)); 
double *res=(double*)malloc(n*sizeof(*res));
double *res_tilde=(double*)malloc(n*sizeof(*res_tilde));
double *sig=(double*)malloc(r*sizeof(*sig));
double *sig_old=(double*)malloc(r*sizeof(*sig_old));
double *temp_v=(double*)malloc(n*sizeof(*temp_v));
double *temp_w=(double*)malloc(n*sizeof(*temp_w));
double *eye_n=(double*)malloc(n*n*sizeof(*eye_n));
double *V =(double *)malloc (n*n*sizeof(double));
double *W =(double *)malloc (n*n*sizeof(double));


double error=100007;
max_iter=100;
rmaxit  =100;
rtol= 0.0001;
itol= 0.0001;
srand((unsigned)time(0));
for(int i=0;i<n;i++){
  B[i]=rand()%10;B[i]/=10;
  C[i]=rand()%10;C[i]/=10;
  for(int j=0;j<n;j++){
    if(i==j)eye_n[i*n+j]=1;
    else eye_n[i*n+j]=0;
    double tempx=rand()%10;
    if(tempx>7 or tempx< 3){A[i*n+j]=(rand()%10);}
    else A[i*n+j]=0;
  }
}
//for(int i=0;i<n;i++){for(int j=0;j<n;j++){cout<<eye_n[i*n+j]<<" ";}cout<<endl;}
//initialize sparse matrix A
cusparseHandle_t handle; cusparseCreate(&handle);

double *d_A_dense;  cudaMalloc(&d_A_dense, n * n * sizeof(double));
double *d_EYE_dense ; cudaMalloc(&d_EYE_dense ,n*n*sizeof(double));

cudaMemcpy(d_A_dense, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
//cudaMemcpy(d_A_dense, A , n*n*sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(d_EYE_dense, eye_n , n*n*sizeof(double),cudaMemcpyHostToDevice);

cusparseMatDescr_t descrA;    cusparseCreateMatDescr(&descrA);
cusparseSetMatType    (descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);  



cusparseMatDescr_t descrEYE; cusparseCreateMatDescr(&descrEYE);
cusparseSetMatType    (descrEYE, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descrEYE, CUSPARSE_INDEX_BASE_ZERO);
int nnzA = 0;             // --- Number of nonzero elements in dense matrix A

const int lda = N;
int nnzEYE=0;
  int *d_nnzPerVectorA;   cudaMalloc(&d_nnzPerVectorA, n * sizeof(*d_nnzPerVectorA));
  cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, n, n, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA);



cout<<"nnzA is equal to "<<nnzA<<endl;
//int *d_nnzPerVectorA; cudaMalloc(&d_nnzPerVectorA,n*sizeof(*d_nnzPerVectorA));
//cusparseDnnz(handle,CUSPARSE_DIRECTION_ROW,n,n,descrA,d_A_dense,lda,d_nnzPerVectorA, &nnzA);


int *d_nnzPerVectorEYE; cudaMalloc(&d_nnzPerVectorEYE,n*sizeof(*d_nnzPerVectorEYE));
cusparseDnnz(handle,CUSPARSE_DIRECTION_ROW,n,n,descrEYE,d_EYE_dense,lda,d_nnzPerVectorEYE, &nnzEYE);


int *h_nnzPerVectorA = (int *)malloc(n * sizeof(*h_nnzPerVectorA));
cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, n * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost);


int *h_nnzPerVectorEYE = (int *)malloc(n * sizeof(*h_nnzPerVectorEYE));
cudaMemcpy(h_nnzPerVectorEYE, d_nnzPerVectorEYE, n * sizeof(*h_nnzPerVectorEYE), cudaMemcpyDeviceToHost);

// device side sparse matrix;
double *d_A ; cudaMalloc(&d_A, nnzA * sizeof(*d_A));
int *d_A_RowIndices ;cudaMalloc(&d_A_RowIndices, (n + 1) * sizeof(*d_A_RowIndices));
int *d_A_ColIndices ;cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices));
cusparseDdense2csr(handle, n, n, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices);

double *d_EYE ; cudaMalloc(&d_EYE, nnzEYE * sizeof(*d_EYE));
int *d_EYE_RowIndices ;cudaMalloc(&d_EYE_RowIndices, (n + 1) * sizeof(*d_EYE_RowIndices));
int *d_EYE_ColIndices ;cudaMalloc(&d_EYE_ColIndices, nnzEYE * sizeof(*d_EYE_ColIndices));
cusparseDdense2csr(handle, n, n, descrEYE, d_EYE_dense, lda, d_nnzPerVectorEYE, d_EYE, d_EYE_RowIndices, d_EYE_ColIndices);




// --- Host side sparse matrices
double *h_A = (double *)malloc(nnzA * sizeof(*h_A));

int *h_A_RowIndices = (int *)malloc((n + 1) * sizeof(*h_A_RowIndices));
int *h_A_ColIndices = (int *)malloc(nnzA * sizeof(*h_A_ColIndices));
cudaMemcpy(h_A, d_A, nnzA * sizeof(*h_A), cudaMemcpyDeviceToHost);
cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (n + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost);
cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost);



double *h_EYE = (double *)malloc(nnzEYE * sizeof(*h_EYE));

int *h_EYE_RowIndices = (int *)malloc((n + 1) * sizeof(*h_EYE_RowIndices));
int *h_EYE_ColIndices = (int *)malloc(nnzEYE * sizeof(*h_EYE_ColIndices));
cudaMemcpy(h_EYE, d_EYE, nnzEYE * sizeof(*h_EYE), cudaMemcpyDeviceToHost);
cudaMemcpy(h_EYE_RowIndices, d_EYE_RowIndices, (n + 1) * sizeof(*h_EYE_RowIndices), cudaMemcpyDeviceToHost);
cudaMemcpy(h_EYE_ColIndices, d_EYE_ColIndices, nnzEYE * sizeof(*h_EYE_ColIndices), cudaMemcpyDeviceToHost);


irka_iter=1;
//initialize sigma ...
double inii=0.5;double fini=7; 
for(int i=0;i<r;i++){
     sig[i]= log((inii + (fini-inii)*i/r ));
}

double *d_A1_dense;  cudaMalloc(&d_A1_dense, n * n * sizeof(*d_A1_dense));
cusparseMatDescr_t descrA1;    cusparseCreateMatDescr(&descrA1);
cusparseSetMatType   (descrA1, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descrA1, CUSPARSE_INDEX_BASE_ONE);
int *d_A1_RowIndices;  cudaMalloc(&d_A1_RowIndices, (n + 1) * sizeof(*d_A1_RowIndices));
  int *h_A1_RowIndices = (int *)malloc((n + 1) * sizeof(*h_A1_RowIndices));


 int baseA1, nnzA1 = 0;
  // nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnzA1;
 cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST); 


double *h_A1_dense = (double*)malloc(n * n * sizeof(*h_A1_dense));

double *h_A1 = (double *)malloc(nnzA1 * sizeof(*h_A1));   
int *h_A1_ColIndices = (int *)malloc(nnzA1 * sizeof(*h_A1_ColIndices));

cudaFree(d_nnzPerVectorA);
cudaFree(d_nnzPerVectorEYE);
//while loop

while(error>itol and irka_iter<max_iter){
 irka_iter++; 
 for(int i=0;i<n;i++)sig_old[i]=sig[i];
 for (int i=0;i<r;i++){
        cusparseXcsrgeamNnz(handle, n, n, 
                            descrEYE, nnzEYE, d_EYE_RowIndices, d_EYE_ColIndices, 
                            descrA, nnzA, d_A_RowIndices, d_A_ColIndices,
                            descrA1, d_A1_RowIndices, 
                            nnzTotalDevHostPtr);


          if (NULL != nnzTotalDevHostPtr){ nnzA1 = *nnzTotalDevHostPtr; }
          else {
           cudaMemcpy(&nnzA1,  d_A1_RowIndices + n, sizeof(int), cudaMemcpyDeviceToHost);
           cudaMemcpy(&baseA1, d_A1_RowIndices,     sizeof(int), cudaMemcpyDeviceToHost);
            nnzA1 -= baseA1;
          }
     

    int *d_A1_ColIndices; cudaMalloc(&d_A1_ColIndices, nnzA1 * sizeof(int));
    
    double *d_A1;         cudaMalloc(&d_A1, nnzA1 * sizeof(double));
   
       double alpha; double beta;
       alpha=sig[i];beta=-1;

       //////////////////////////////// maybe 
        cusparseDcsrgeam
                   (handle, n, n,
                    &alpha, 
                    descrEYE, nnzEYE, d_EYE, d_EYE_RowIndices, d_EYE_ColIndices,
                    &beta,
                    descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, 
                    descrA1, d_A1, d_A1_RowIndices, d_A1_ColIndices);
       
       cusparseDcsr2dense(handle, n, n, descrA1, d_A1, d_A1_RowIndices, d_A1_ColIndices, d_A1_dense, n);
       cudaMemcpy(h_A1 ,           d_A1,            nnzA1 * sizeof(*h_A1), cudaMemcpyDeviceToHost);
       cudaMemcpy(h_A1_RowIndices, d_A1_RowIndices, (n + 1) * sizeof(*h_A1_RowIndices), cudaMemcpyDeviceToHost);
       cudaMemcpy(h_A1_ColIndices, d_A1_ColIndices, nnzA1 * sizeof(*h_A1_ColIndices), cudaMemcpyDeviceToHost);
       cudaMemcpy(h_A1_dense, d_A1_dense, n * n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_A1);
       cudaFree(d_A1_ColIndices);
  
        // iintitialise x0 and x0_tilde
        for(int j =0;j<n;j++){
          x0[j]=((rand()+2)%7)/10;x0_tilde[j]=((rand()+7)%10)/10;
        } 
        int status1= bicg(h_A1_dense,x0,B,eye_n,rmaxit,rtol,n);
        int status2= bicg(h_A1_dense,x0_tilde,C,eye_n,rmaxit,rtol,n);
        for(int j=0;j<n;j++){V[j*n+i]=x0[j];W[n*j+i]=x0_tilde[j];}


 
  }
  // We have V and W matrix .... We need to orthogonalise them ....
  cusolverDnHandle_t solver_handle_m;
  cusolverDnCreate (&solver_handle_m);

  cublasHandle_t cublas_handle_m;
  cublasCreate(&cublas_handle_m);
  int work_size_m=0;
  int *devInfo_m;
  const int Nrows=n;
  const int Ncols=r;
  double *d_V; cudaMalloc(&d_V,Nrows*Ncols*sizeof(double));
  double *d_W; cudaMalloc(&d_W,Nrows*Ncols*sizeof(double));
  cudaMemcpy(d_V,V,Nrows*Ncols*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W,W,Nrows*Ncols*sizeof(double), cudaMemcpyHostToDevice);
  //Cuda Qr initialisation,
  double *d_TAU_V ; cudaMalloc(&d_TAU_V,min(Nrows,Ncols)*sizeof(double));
  double *d_TAU_W ; cudaMalloc(&d_TAU_W,min(Nrows,Ncols)*sizeof(double));

  cusolverDnDgeqrf_bufferSize(solver_handle_m, Nrows,Ncols, d_V ,n ,&work_size_m);

  double *work_V_m ; cudaMalloc(&work_V_m, work_size_m*sizeof(double ));
  double *work_W_m ; cudaMalloc(&work_W_m, work_size_m*sizeof(double ));
  // Cuda GERF exec...
// cusolverDnDgeqrf_bufferSize(        cusolverH,        m,        n,        d_A,       lda,        &lwork_geqrf);
//cusolver_status = cusolverDnDgeqrf( cusolverH, m, n, d_A, lda, d_tau, d_work, lwork, devInfo);


  cusolverDnDgeqrf(solver_handle_m,Nrows,Ncols,d_V,n,d_TAU_V,work_V_m,work_size_m,devInfo_m);
  int devInfo_V_h=0; cudaMemcpy(&devInfo_V_h,devInfo_m,sizeof(int),cudaMemcpyDeviceToHost);

  cusolverDnDgeqrf(solver_handle_m, Nrows , Ncols, d_W,n, d_TAU_W, work_W_m , work_size_m,devInfo_m);
  int devInfo_W_h=0; cudaMemcpy(&devInfo_W_h,devInfo_m,sizeof(int),cudaMemcpyDeviceToHost);
  
  if(devInfo_W_h!=0 or devInfo_V_h!=0){cout<<"Unsuccesful";}
  // At his point the upper triangular part of A contains the elemrnts of R.
  

  // Initialising Q matrix.
  double *h_Q_V= (double *)malloc(Nrows*Nrows*sizeof(double));
  double *h_Q_W= (double *)malloc(Nrows*Nrows*sizeof(double));
  for(int j=0;j<Nrows;j++)for(int i=0;i<Nrows;i++){if(j==i){h_Q_V[j+i*Nrows]=1;h_Q_W[j+i*Nrows]=1;}
                                                   else {h_Q_V[j+i*Nrows]=0;h_Q_W[j+i*Nrows]=0;}  }
  double *d_Q_V;cudaMalloc(&d_Q_V,Nrows*Nrows*sizeof(double));
  double *d_Q_W;cudaMalloc(&d_Q_W,Nrows*Nrows*sizeof(double));
  cudaMemcpy(d_Q_V,h_Q_V,Nrows*Nrows*sizeof(double),cudaMemcpyHostToDevice);cudaMemcpy(d_Q_W,h_Q_W,Nrows*Nrows*sizeof(double),cudaMemcpyHostToDevice);
  // CuDA QR execution 
  cusolverDnDormqr(solver_handle_m,CUBLAS_SIDE_LEFT,CUBLAS_OP_N,Nrows,Ncols,min(Nrows,Ncols),d_V,Nrows,d_TAU_V,d_Q_V,Nrows,work_V_m,work_size_m,devInfo_m);
  cusolverDnDormqr(solver_handle_m,CUBLAS_SIDE_LEFT,CUBLAS_OP_N,Nrows,Ncols,min(Nrows,Ncols),d_W,Nrows,d_TAU_W,d_Q_W,Nrows,work_W_m,work_size_m,devInfo_m);
  cudaMemcpy(h_Q_V,d_Q_V,Nrows*Nrows*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Q_W,d_Q_W,Nrows*Nrows*sizeof(double), cudaMemcpyDeviceToHost);
  for(int i=0;i<n;i++){
    for(int j=0;j<r;j++){
       V[i*n+j]=h_Q_V[i*n+j];
       W[i*n+j]=h_Q_W[i+j*n];// making it W^T

    }
  }
  // V and W have been orthogonalised
  // find Ared , Bred ...
  double *d_Q_V_mod;cudaMalloc(&d_Q_V_mod,Nrows*Ncols*sizeof(double));
  double *d_Q_W_mod;cudaMalloc(&d_Q_W_mod,Nrows*Ncols*sizeof(double));
  double *d_A_temp; cudaMalloc(&d_A_temp,Nrows*Ncols*sizeof(double));
  double *d_A_red; cudaMalloc(&d_A_red,Ncols*Ncols*sizeof(double));

  cudaMemcpy(d_Q_V_mod,V,Nrows*Ncols*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q_W_mod,W,Nrows*Ncols*sizeof(double), cudaMemcpyHostToDevice);
  gpu_blas_mmul(d_A_dense, d_Q_V_mod, d_A_temp, n, n, r);// q=Ap
  gpu_blas_mmul(d_Q_W_mod, d_A_temp, d_A_red, r, n, r);
  // d_A_red has the reduced Matrix ....
  double *d_B ;cudaMalloc(&d_B ,Nrows*sizeof(double));cudaMemcpy(d_B,B,Nrows*sizeof(double),cudaMemcpyHostToDevice);
  double *d_B_red ;cudaMalloc(&d_B_red ,Ncols*sizeof(double));
  gpu_blas_mmul(d_Q_W, d_B, d_B_red, r, n, 1);
  double *d_C ;cudaMalloc(&d_C,Nrows*sizeof(double ));
   double *d_C_red ;cudaMalloc(&d_C_red,Ncols*sizeof(double ));
  cudaMemcpy(d_C,C,Nrows*sizeof(double),cudaMemcpyHostToDevice);
  gpu_blas_mmul(d_C,d_Q_V_mod,d_C_red,1,n,r);
  double *B_red=(double *)malloc(r*sizeof(double));
  double *C_red=(double *)malloc(r*sizeof(double));
  double *A_red=(double *)malloc(r*r*sizeof(double));
  cudaMemcpy(A_red,d_A_red,r*r*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(B_red,d_B_red,r*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(C_red,d_C_red,r*sizeof(double),cudaMemcpyDeviceToHost);
 // we find the eiggen values of the Ared ... and change sigma ...

  double *eigv= (double *)malloc(r*sizeof(double ));
  double *eigvec= (double *)malloc(r*r*sizeof(double ));
   cusolverDnHandle_t cusolverH = NULL;
   cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
   int  lwork_eig = 0;
   int lda=r;cusolverDnCreate(&cusolverH);
  int  *dev_info_eig;cudaMalloc ((void**)&dev_info_eig, sizeof(int));
  double *d_eigv;cudaMalloc ((void**)&d_eigv, r*sizeof(double));
  double *d_eigvec;cudaMalloc ((void**)&d_eigvec, r*r*sizeof(int));
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
   cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER; 
   cusolver_status = cusolverDnDsyevd_bufferSize( cusolverH, jobz, uplo, r, d_A_red, lda, d_eigv, &lwork_eig);
   double *d_work_eig;  cudaMalloc((void**)&d_work_eig, sizeof(double)*lwork_eig);
   int *devInfo_eig = NULL;
   cusolverDnDsyevd( cusolverH, jobz, uplo, r, d_A_red, lda, d_eigv, d_work_eig, lwork_eig, devInfo_eig);
   cudaMemcpy(eigv,d_eigv,r*sizeof(double),cudaMemcpyDeviceToHost);
   cudaMemcpy(eigvec,d_A_red,r*r*sizeof(double),cudaMemcpyDeviceToHost);



  //
   double  norm_sigma=0;error=0;
   for(int j=0;j<r;j++){norm_sigma+=sig[j]*sig[j];sig[j]=eigv[j];error+=(sig[j]-sig_old[j])*(sig[j]-sig_old[j]);}
    error/=norm_sigma;

 
  cudaFree(d_V);
  cudaFree(d_W);
  
  cudaFree(d_TAU_V);
  cudaFree(d_TAU_W);
  
  cudaFree(work_V_m);
  cudaFree(work_W_m);


  cudaFree(d_Q_V);
  cudaFree(d_Q_W);
  
  cudaFree(d_Q_V_mod);
  cudaFree(d_Q_W_mod);

  cudaFree(d_A_temp);
  cudaFree(d_A_red);
  
  cudaFree(d_B);
  cudaFree(d_B_red);

  cudaFree(d_C);
  cudaFree(d_C_red);
  
  cudaFree(dev_info_eig);
  cudaFree(d_eigv);

  cudaFree(d_eigvec);
  cudaFree(d_work_eig);

  free(h_Q_V);
  free(h_Q_W);
  free(B_red);
  free(A_red);
  free(C_red);
  free(eigv);
  free(eigvec);
  cusolverDnDestroy(solver_handle_m);
  cublasDestroy(cublas_handle_m);
  
  

 }


//goes betwwn these two comments.
cusparseDestroyMatDescr(descrEYE);
cusparseDestroyMatDescr(descrA);
cusparseDestroy(handle);
cudaFree(d_A1_dense);
cudaFree(d_A1_RowIndices);
cudaFree(d_EYE);
cudaFree(d_EYE_ColIndices);
cudaFree(d_EYE_RowIndices);
cudaFree(d_A);
cudaFree(d_A_RowIndices);
cudaFree(d_A_ColIndices);
cudaFree(d_A_dense);
cudaFree(d_nnzPerVectorA);
cudaFree(d_nnzPerVectorEYE);


}


