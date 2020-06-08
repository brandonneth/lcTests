//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////
#include "RAJA/RAJA.hpp"

template <typename T>
T *allocate(std::size_t size)
{
  T *ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(
      cudaMallocManaged((void **)&ptr, sizeof(T) * size, cudaMemAttachGlobal));
#else
  ptr = new T[size];
#endif
  return ptr;
}

template <typename T>
void deallocate(T *&ptr)
{
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(ptr));
#else
    delete[] ptr;
#endif
    ptr = nullptr;
  }
}


//tests the execution of kernel objects created by the make_kernel function without modification
int test_kernel_exec() {

  int numErrors = 0;
  double * _a = allocate<double>(100);
  double * _b = allocate<double>(100);

  using VIEW_TYPE = RAJA::View<double, RAJA::Layout<1, int, 1>>;
  
  VIEW_TYPE a1(_a, 100);
  VIEW_TYPE b1(_b, 100);

  for(int i = 0; i < 100; i++) {
    _a[i] = 10;
  }
 
  using KERNEL_POL1 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::Lambda<0>
    >
  >;
  auto knl1 = RAJA::make_kernel<KERNEL_POL1>(RAJA::make_tuple(RAJA::RangeSegment(0,100)), [=](auto i) {b1(i) = a1(i) + i;});

  knl1();

  int error = 0;
  for(int i = 0; i < 100; i++) {
    if(_b[i] != 10 + i) {
      error = 1;
    }
  }

  numErrors += error;

  for(int i = 0; i < 100; i++) {
    _a[i] = i;
    _b[i] = -100;
  }

  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;

  VIEW_TYPE2 a2(_a, 10, 10);
  VIEW_TYPE2 b2(_b, 10, 10);

  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;
  
  auto knl2 = RAJA::make_kernel<KERNEL_POL2>(
    RAJA::make_tuple(
      RAJA::RangeSegment(1,9),
      RAJA::RangeSegment(0,10)),
    [=] (auto i, auto j) {
      b2(i,j) = (a2(i-1,j) + a2(i,j) + a2(i+1,j)) / 3;
    }
  ); 

  knl2();

  error = 0;
  for(int i = 1; i < 9; i++) {
    for(int j = 0; j < 10; j++) {
      if(b2(i,j) != (a2(i-1,j) + a2(i,j) + a2(i+1,j)) / 3) {
        error = 1;
      }
    }
  }

  numErrors += error;

  return numErrors;
}


//tests the execution of kernel objects created by the make_forall function without modification
int test_forall_exec() {
  double * _a = allocate<double>(100);
  double * _b = allocate<double>(100);

  using VIEW_TYPE = RAJA::View<double, RAJA::Layout<1, int, 1>>;
  
  VIEW_TYPE a(_a, 100);
  VIEW_TYPE b(_b, 100);

  for(int i = 0; i < 100; i++) {
    _a[i] = 10;
  }
 
  auto forall1 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [=](auto i) {b(i) = a(i) + i;});

  forall1();

  int error = 0;
  for(int i = 0; i < 100; i++) {
    if(_b[i] != 10 + i) {
      error = 1;
    }
  }

  deallocate(_a);
  deallocate(_b);
  return error;
}  

//tests the creation of chain objects
int test_chain_creation() {
  int numErrors = 0;

  double * _a = allocate<double>(100);
  double * _b = allocate<double>(100);

  using VIEW_TYPE1 = RAJA::View<double, RAJA::Layout<1, int, 1>>;
  
  VIEW_TYPE1 a1(_a, 100);
  VIEW_TYPE1 b1(_b, 100);

  auto knl1 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [=](auto i) {a1(i) = 1;});
  auto knl2 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [=](auto i) {b1(i) = 2;});
  auto knl3 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [=](auto i) {a1(i) = 2;});
  
  auto singleKnlChain = RAJA::chain(knl1);

  singleKnlChain();

  int error = 0;
  for(int i = 0; i < 100; i++) {
    if (_a[i] != 1) {
      error = 1;
    }
  }
  numErrors += error;

  auto doubleKnlChain = RAJA::chain(knl2, knl3);

  doubleKnlChain();
  
  error = 0;
  for(int i = 0; i < 100; i++) {
    if (_a[i] != 2 || _b[i] != 2 ) {
      error = 1;
    }
  }
  numErrors += error;

  return numErrors;
}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[])) {


  int numErrors = 0;
  
  int forallExecErrors = test_forall_exec();
  if(forallExecErrors != 0) {
    std::cout << "Forall Execution Errors: " << forallExecErrors << "\n";
    numErrors += forallExecErrors;

  }

  numErrors += test_kernel_exec();

  int chainCreationErrors = test_chain_creation();
  if(chainCreationErrors != 0) {
    std::cout << "Chain Creation Errors: " << chainCreationErrors << "\n";
    numErrors += chainCreationErrors;
  }

  std::cout << "Total error count: " << numErrors;

  return numErrors;
}
