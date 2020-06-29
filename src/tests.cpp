//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////
#include "RAJA/RAJA.hpp"


#include <iostream>
#include <sstream>
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
  auto knl1 = RAJA::make_kernel<KERNEL_POL1>(camp::make_tuple(RAJA::RangeSegment(0,100)), [&](auto i) {b1(i) = a1(i) + i;});

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
    camp::make_tuple(
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


//tests the symbolic execution of kernels
int test_sym_exec() {
  int numErrors = 0;
  double * _a = allocate<double>(100);
  double * _b = allocate<double>(100);

  using VIEW_TYPE = RAJA::View<double, RAJA::Layout<1, int, 1>>;
  
  VIEW_TYPE a(_a, 100);
  VIEW_TYPE b(_b, 100);

  for(int i = 0; i < 100; i++) {
    _a[i] = 10;
  }


 
  auto forall1 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [&](auto i) {b(i) = a(i) + i;});

  auto symbolicAccesses = forall1.execute_symbolically();
  
  int err1a = 1;
  int err1b = 1;
  int err1extra = 0;
  for( auto access : symbolicAccesses) {
    if (access.view == _a && access.isRead) {
      err1a = 0;
    }
    else if (access.view == _b && access.isWrite) { 
      err1b = 0;
    } else {
      err1extra = 1;
    }
  } 

  if(err1a) {
    std::cerr << "Symbolic Execution did not gather read of a(i)\n";
  }

  if(err1b) {
    std::cerr << "Symbolic Execution did not gather write of b(i)\n";
  }
  if(err1extra) {
    std::cerr << "Symbolic Execution collected more accesses than there were\n";
  }
  numErrors += err1a + err1b + err1extra;
  return numErrors;
} //test_sym_exec


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
 
  auto forall1 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [&](auto i) {b(i) = a(i) + i;});

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

  auto knl1 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [&](auto i) {a1(i) = 1; b1(i) = 0;});
  auto knl2 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [&](auto i) {b1(i) = 2;});
  auto knl3 = RAJA::make_forall<RAJA::seq_exec>(RAJA::RangeSegment(0,100), [&](auto i) {a1(i) = b1(i);});
  
  auto singleKnlChain = RAJA::chain(knl1);

  singleKnlChain();

  int error = 0;
  for(int i = 0; i < 100; i++) {
    if (_a[i] != 1) {
      error = 1;
    }
  }
  if(error == 1) {
    std::cerr << "Single chain creation failed to execute\n";
  }
  numErrors += error;

  auto doubleKnlChain = RAJA::chain(knl2, knl3);

  doubleKnlChain();
  
  int error1 = 0;
  int error2 = 0;
  for(int i = 0; i < 100; i++) {
    if (_b[i] != 2) {
      error1 = 1;
    }
    if (_a[i] != 2 ) {
      error2 = 1;
    }
  }
  numErrors += error1 + error2;
  if (error1 == 1) {
    std::cerr << "Two loop chain creation failed to execute loop 1\n";
  } 
  if (error2 == 1) {
    std::cerr << "Two loop chain creation failed to execute loop 2\n";
  }
  return numErrors;
}


int test_create_transformations() {

  int numErrors = 0;

  auto shift1 = RAJA::shift<0>(0,0,1);

  auto shift2 = RAJA::shift<3>(-2);


  auto fusion1 = RAJA::Fusion<0,1,2>();
  auto fusion2 = RAJA::Fusion<0,1>();
  auto fusion3 = RAJA::Fusion<2,4,5>();



  
   return numErrors;


} //test_create_transformations()

int test_apply_shift() {

  int numErrors = 0;
  double * _a = allocate<double>(100);
  double * _b = allocate<double>(100);

  using VIEW_TYPE1 = RAJA::View<double, RAJA::Layout<1, int, 1>>;
  
  VIEW_TYPE1 a1(_a, 100);
  VIEW_TYPE1 b1(_b, 100);

  using KERNEL_POL1 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::Lambda<0>
    >
  >;

  for(int i = 0; i < 100; i++) {
    _a[i] = i;
    _b[i] = 0;
  }
  auto knl1 = RAJA::make_kernel<KERNEL_POL1>(
                RAJA::make_tuple(RAJA::RangeSegment(10,20)), 
                [&](auto i) {b1(i) = a1(i) + i;}
              );
   
  auto chain1 = RAJA::chain(knl1, RAJA::shift<0>(5));

  chain1();

  int chain1Error1 = 0;
  for(int i = 0; i < 10; i++) {
    if(b1(i) != 0) {
      chain1Error1 = 1;
    }
  }

  int chain1Error2 = 0;
  for(int i = 10; i < 20; i++) {
    if(b1(i) != a1(i) + i) {
      chain1Error2 = 1;
    }
  }

  int chain1Error3 = 0;
  for(int i = 20; i < 100; i++) {
    if(b1(i) != 0) {
      chain1Error3 = 1;
    }
  }

  if(chain1Error1) {
    std::cerr << "Shift application error with pre-range\n";
  }
  if(chain1Error2) {
    std::cerr << "Shift application error with range\n";
  }
  if(chain1Error3) {
    std::cerr << "Shift application error with post-range\n";
  }
  numErrors += chain1Error1 + chain1Error2 + chain1Error3;

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

  //TODO: 2d shift
  return numErrors;
} //test_apply_shift

int test_apply_fuse() {

  int numErrors = 0;
  double * _a = allocate<double>(100);
  double * _b = allocate<double>(100);

  using VIEW_TYPE1 = RAJA::View<double, RAJA::Layout<1, int, 1>>;

  VIEW_TYPE1 a1(_a, 100);
  VIEW_TYPE1 b1(_b, 100);

  using KERNEL_POL1 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::Lambda<0>
    >
  >;

  for(int i = 0; i < 100; i++) {
    _a[i] = 10;
    _b[i] = 0;
  }
  auto knl1 = RAJA::make_kernel<KERNEL_POL1>(
                RAJA::make_tuple(RAJA::RangeSegment(10,20)),
                [&](auto i) {b1(i) = a1(i);}
              );
  auto knl2 = RAJA::make_kernel<KERNEL_POL1>(
                RAJA::make_tuple(RAJA::RangeSegment(10,20)),
                [=](auto i) {a1(i+1) = 1;}
              );


  auto chain1 = RAJA::chain(knl1, knl2, RAJA::fuse<0,1>());


  chain1();

  int c1err1 = 0;
  for(int i = 10; i < 11; i++) {
    if(_b[i] != 10) {
      c1err1 = 1;
    }
  }

  int c1err2 = 0;
  for(int i = 11; i < 20; i++) {
    if(_b[i] != 1) {
      c1err2 = 1;
    }
  }

  if(c1err1) {
    std::cerr << "1d fused knl fails for b(10)\n";
  }

  if(c1err2) {
    std::cerr << "1d fused knl fails for b11 to b20\n";
  }

  numErrors += c1err1 + c1err2;
  
  auto knl3 =  RAJA::make_kernel<KERNEL_POL1>(
                RAJA::make_tuple(RAJA::RangeSegment(8,15)),
                [&](auto i) {b1(i) = a1(i);}
              );


  for(int i = 0; i < 100; i++) {
    _a[i] = 10;
    _b[i] = 0;
  }
  
  auto chain2 = RAJA::chain(knl3,knl2, RAJA::fuse<0,1>());

  chain2();

  int c2err1 = 0;
  for(int i = 8; i < 10; i++) {
    if(_b[i] != 10) {
      c2err1 = 1;
    }
  }

  if(c2err1) {
    std::cerr << "Fusion error with starting non-overlap segment of loop 1\n";
  }

  int c2err2 = 0;
  for(int i = 11; i < 15; i++) {
    if(_b[i] != 1) {
      c2err2 = 1;
    }
  }
  
  if(c2err2) {
    std::cerr << "2d fusion error with overlapping segment\n";
  } 
  
  int c2err3 = 0;
  int c2err4 = 0;
  for(int i = 15; i < 20; i++) {
    if(_b[i] != 0) {
      c2err3 = 1;
    }
    if(_a[i] != 1) {
      c2err4 = 1;
    }
  }
 
  if(c2err3) {
    std::cerr << "First loop executed past its range in fusion\n";
  }
  if(c2err4) {
    std::cerr << "Second loop did not execute its non-overlap in fusion\n";
  }

  numErrors += c2err1 + c2err2 + c2err3 + c2err4;


  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;

  VIEW_TYPE2 a2(_a,10,10);
  VIEW_TYPE2 b2(_b,10,10);
   
  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

 
  

  return numErrors;
} //test_apply_fuse()


int test_apply_tile() {
  int numErrors = 0;



  return numErrors;
} //test_apply_tile

int test_apply_overlapped_tile() {
  int numErrors = 0;
  double * _a = allocate<double>(64);
  double * _b = allocate<double>(64);

  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a2(_a,8,8);
  VIEW_TYPE2 b2(_b,8,8);
   
  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

  std::stringstream s;
  
  auto l = [&](auto i, auto j) {
    s << i << j << ".";
  };

  auto segments = RAJA::make_tuple(RAJA::RangeSegment(4,6), RAJA::RangeSegment(4,6));

  auto knl = RAJA::make_kernel<KERNEL_POL2>(segments, l);

  auto overlapAmounts = RAJA::make_tuple(1,1);
  auto tileSizes = RAJA::make_tuple(1,1);

  auto chain = RAJA::chain(knl, RAJA::overlapped_tile<0>(overlapAmounts, tileSizes));

  chain();

  std::string chain1Correct = "33.34.43.44." "34.35.44.45." "43.44.53.54." "44.45.54.55.";

  if(s.str() != chain1Correct) {
    std::cerr << "2D OverlappedTile Error.\nShould be: " << chain1Correct << "\nIs       : " << s.str() << "\n";
    numErrors +=1;
  }

  return numErrors;
}
int printing_fuse_test() {

  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;




  std::cout <<"\n\n";

  auto l1 = [=](auto i, auto j) {std::cout << "1" << i << j << ".";};

  auto l2 = [=](auto i, auto j) {std::cout << "2" << i << j << ".";};

  auto knl1 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,2), RAJA::RangeSegment(0,2)), l1);
  auto knl2 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(1,3), RAJA::RangeSegment(1,3)), l2);

  std::cout << "executing knl1\n";
  knl1();
  std::cout << "\n\nexecuting knl2\n";
  knl2();
  std::cout << "\n\n";

  auto chain = RAJA::chain(knl1, knl2, RAJA::fuse<0,1>());
  std::cout << "calling fused version\n";
  chain();
  std::cout << "\n\n";
 
  std::cout << "done with printing version\n"; 
 
  return 0; 
}


int test_isl() {
  int numErrors = 0;
  double * _a = allocate<double>(64);
  double * _b = allocate<double>(64);

  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a2(_a,8,8);
  VIEW_TYPE2 b2(_b,8,8);
   
  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

  auto knl = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    a2(i) = b2(i);
  });
  
  std::cout << "calling iterspace_from_knl\n";
  RAJA::iterspace_from_knl(knl);

  std::cout << "calling dataspace_from_knl\n";
  RAJA::dataspace_from_knl(knl);

}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[])) {

  printing_fuse_test();

  int numErrors = 0;
  
  int forallExecErrors = test_forall_exec();
  std::cerr << "Forall Execution Errors: " << forallExecErrors << "\n\n";
  numErrors += forallExecErrors;


  int kernelExecErrors = test_kernel_exec();
  std::cerr << "Kernel Execution Errors: " << kernelExecErrors << "\n\n";
  numErrors += kernelExecErrors;

  int symExecErrors = test_sym_exec();
  std::cerr << "Symbolic Execution Errors: " << symExecErrors << "\n\n";
  numErrors += symExecErrors;

  int chainCreationErrors = test_chain_creation();
  std::cout << "Chain Creation Errors: " << chainCreationErrors << "\n\n";
  numErrors += chainCreationErrors;

  int createTransErrors = test_create_transformations();
  std::cout << "Transformation Creation Errors: " << createTransErrors << "\n\n";
  numErrors += createTransErrors;

  int shiftErrors = test_apply_shift();
  std::cerr << "Shift Application Errors: " << shiftErrors << "\n\n";
  numErrors += shiftErrors;

  int fuseErrors = test_apply_fuse();
  std::cerr << "Fuse Application Errors: " << fuseErrors << "\n\n";
  numErrors += fuseErrors;

  int tileErrors = test_apply_tile();
  std::cerr << "Tile Application Errors: " << tileErrors << "\n\n";
  numErrors += tileErrors;

  int overlappedTileErrors = test_apply_overlapped_tile();
  std::cerr << "Overlapped Tile Application Errors: " << overlappedTileErrors << "\n\n";
  numErrors += overlappedTileErrors;

  int islErrors = test_isl();
  std::cout << "Total error count: " << numErrors;

  return numErrors;
}
