//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//////////////////////////////////////////////////////////////////////////////
#include "RAJA/RAJA.hpp"
#include "RAJA/util/all-isl.h"

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


// Tests the function that returns a polyhedron representing the iteration space of a kernel
int test_iteration_space_isl() {

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
  
  isl_ctx * ctx = isl_ctx_alloc();
  isl_union_set * iterspace = RAJA::iterspace_from_knl<0>(ctx, knl);
  isl_union_set * correctSet = isl_union_set_read_from_str(ctx, "{L0[i0,i1] : 0 <= i0 < 8 and 0 <= i1 < 8}"); 
  
  if(! isl_union_set_is_equal(correctSet, iterspace) ){
    std::cerr << "iterspace_from_knl returns the wrong iteration space\n";
    numErrors += 1;
  }
  return numErrors;

} //test_iteration_space_isl

int test_access_functions_isl() {

  int numErrors = 0;
  double * _a = allocate<double>(64);
  double * _b = allocate<double>(64);
  double * _c = allocate<double>(64);
  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a2(_a,8,8);
  VIEW_TYPE2 b2(_b,8,8);
  VIEW_TYPE2 c2(_c,8,8);
  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;
  
  auto knl1 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    a2(i,j) = b2(i,j) + c2(i,j);
  });
  auto knl2 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    b2(i,j) = a2(i,j);
    c2(i,j) = a2(i,j);
  });
  
  isl_ctx * ctx = isl_ctx_alloc();

  isl_union_map * reads1 = RAJA::read_relation_from_knl<0>(ctx, knl1);
  isl_union_map * writes1 = RAJA::write_relation_from_knl<0>(ctx, knl1);

  isl_union_map * reads2 = RAJA::read_relation_from_knl<1>(ctx, knl2);
  isl_union_map * writes2 = RAJA::write_relation_from_knl<1>(ctx, knl2);

  if(!isl_union_map_is_equal(reads1, writes2)){
    std::cerr << "Error with read and write relations for kernels. reads in a2(i,j) = b2(i,j) + c2(i,j) should be the same as the writes in b2(i,j) = a2(i,j); c2(i,j) = a2(i,j);\n";
    numErrors += 1;
  }
  if(!isl_union_map_is_equal(reads2, writes1)) {
    std::cerr << "Error with read and write relations for kernels. writes in a2(i,j) = b2(i,j) + c2(i,j) should be the same as the reads in b2(i,j) = a2(i,j); c2(i,j) = a2(i,j);\n";
    numErrors += 1;
  }
   return numErrors;

}

int test_dep_relations_isl() {

  int numErrors = 0;
  double * _a = allocate<double>(64);
  double * _b = allocate<double>(64);
  double * _c = allocate<double>(64);
  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a2(_a,8,8);
  VIEW_TYPE2 b2(_b,8,8);
  VIEW_TYPE2 c2(_c,8,8);
  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;
  
  auto knl1 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    a2(i,j) = b2(i,j) + c2(i,j);
  });
  auto knl2 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    b2(i,j) = a2(i,j);
    c2(i,j) = a2(i,j);
  });
  isl_ctx * ctx = isl_ctx_alloc();
  
  isl_union_map * flowDeps = RAJA::flow_dep_relation_from_knls<0,1>(ctx, knl1, knl2);

  isl_union_map * deps = RAJA::data_dep_relation_from_knls<0,1>(ctx, knl1, knl2);

  isl_printer * p = isl_printer_to_file(ctx, stdout);

  p = isl_printer_print_union_map(p, deps);

  return 0;
} // test_dep_relations_isl()


int test_original_schedule_isl() {


  int numErrors = 0;
  double * _a = allocate<double>(64);
  double * _b = allocate<double>(64);
  double * _c = allocate<double>(64);
  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a2(_a,8,8);
  VIEW_TYPE2 b2(_b,8,8);
  VIEW_TYPE2 c2(_c,8,8);
  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;
  
  auto knl1 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    a2(i,j) = b2(i,j) + c2(i,j);
  });
  auto knl2 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    b2(i,j) = a2(i,j);
    c2(i,j) = a2(i,j);
  });

  isl_ctx * ctx = isl_ctx_alloc();

  isl_union_map * osched1 = RAJA::original_schedule<0>(ctx, knl1);
  isl_union_map * osched2 = RAJA::original_schedule<1>(ctx, knl2);


  return 0;
} //test_original_schedule_isl

int test_can_fuse_isl() {



  int numErrors = 0;
  double * _a = allocate<double>(64);
  double * _b = allocate<double>(64);
  double * _c = allocate<double>(64);
  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a2(_a,8,8);
  VIEW_TYPE2 b2(_b,8,8);
  VIEW_TYPE2 c2(_c,8,8);
  using KERNEL_POL2 = RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;
  
  auto knl1 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    a2(i,j) = b2(i,j) + c2(i,j);
  });
  auto knl2 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    b2(i,j) = a2(i,j);
    c2(i,j) = a2(i,j);
  });

  isl_ctx * ctx = isl_ctx_alloc();

  int canFuse = RAJA::can_fuse<0,1>(knl1, knl2);
  if(!canFuse) {

    numErrors += 1;
    std::cerr << "Fusable loops are returning not fusable\n";
  }

  auto knl3 = RAJA::make_kernel<KERNEL_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,8),RAJA::RangeSegment(0,8)), [=](auto i, auto j) {
    a2(i,j) = b2(i+1,j+1);
  });

  int canFuse2 = RAJA::can_fuse<0,1>(knl2, knl3);

  if(canFuse2) {
    numErrors += 1;
    std::cerr << "Non-fusable loops are returning fusable\n";
  }
  
  auto shifted3 = RAJA::shift_kernel(knl3, RAJA::shift<0>(1, 1));

  int canFuseShifted = RAJA::can_fuse<0,1>(knl2, shifted3);

  if(!canFuseShifted) {
    std::cerr << "Cannot fuse 'b2(i,j) = a2(i,j)' and 'a2(i,j) = b2(i+1,j+1)' after shifted knl2 by 1 1\n";
    numErrors += 1;
  }
  return numErrors;
} //test_can_fuse_isl




int test_isl() {
  int numErrors = 0;


  numErrors += test_iteration_space_isl();

  std::cout << "Testing access functions\n";
  numErrors += test_access_functions_isl();
 
  std::cout << "Testing dependence relations\n"; 
  numErrors += test_dep_relations_isl();
  
  numErrors += test_original_schedule_isl();

  numErrors += test_can_fuse_isl();

  
  return numErrors;
}//test_isl

template <camp::idx_t currIdx, camp::idx_t maxIdx>
void print_iteration_spaces(auto knlTuple) {
  
  if constexpr (currIdx > maxIdx) {return;}
  else {
  auto knl = camp::get<currIdx>(knlTuple);

  isl_ctx * ctx = isl_ctx_alloc();

  isl_printer * p = isl_printer_to_file(ctx, stdout);
 
  std::cout << "Kernel" << currIdx << ":";
 
  p = isl_printer_print_union_set(p, RAJA::iterspace_from_knl<currIdx>(ctx, knl));
  std::cout << "\n";
  print_iteration_spaces<currIdx+1, maxIdx>(knlTuple);
  
  
  }
} //print_iteration_spaces


//example for fusion issue from 7.2.2020 notes
int triple_fusion_example() {

  std::cout << "\n\n\n\n===== Triple Fusion Example======\n";  
  int numErrors = 0;
  double * _a = allocate<double>(16*16);
  double * _b = allocate<double>(16*16);
  double * _c = allocate<double>(16*16);
  double * _d = allocate<double>(16*16);
  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a(_a,16,16);
  VIEW_TYPE2 b(_b,16,16);
  VIEW_TYPE2 c(_c,16,16);
  VIEW_TYPE2 d(_d,16,16);

  auto rangeSegment = RAJA::make_tuple(RAJA::RangeSegment(1,15), RAJA::RangeSegment(1,15));

  auto lambda1 = [=](auto i, auto j) {
    b(i,j) = (a(i-1,j) + a(i,j-1) + a(i,j) + a(i+1,j) + a(i,j+1));
    
  };
  auto lambda2 = [=](auto i, auto j) {
    c(i,j) = (b(i-1,j) + b(i,j-1) + b(i,j) + b(i+1,j) + b(i,j+1));
  };
  auto lambda3 = [=](auto i, auto j) {
    d(i,j) = (c(i-1,j) + c(i,j-1) + c(i,j) + c(i+1,j) + c(i,j+1));
  };

  using KPOL =  RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

  auto knl1 = RAJA::make_kernel<KPOL>(rangeSegment, lambda1);
  auto knl2 = RAJA::make_kernel<KPOL>(rangeSegment, lambda2);
  auto knl3 = RAJA::make_kernel<KPOL>(rangeSegment, lambda3);

  std::cout << "Seeking to fuse three Jacobi style kernels. a -> b, b -> c, and c -> d.\n";

  std::cout << "First, we need to fuse knl1 and knl2.\n";

  int canFuse12 = RAJA::can_fuse<0,1>(knl1, knl2);

  std::cout << "Return value for can_fuse(knl1,knl2): " << canFuse12 << "\n";

  std::cout << "Thus, we need to shift knl2. The amount to shift turns out to be 1 in both directions.\n";

  auto shifted2 = RAJA::shift_kernel(knl2, RAJA::shift<1>(1,1));

  canFuse12 = RAJA::can_fuse<0,1>(knl1, shifted2);
  std::cout << "Return value for can_fuse(knl1,shifted2): " << canFuse12 << "\n";;

  std::cout << "So we can fuse the knl1 with the shifted knl2! However, there is now the problem that the iteration spaces for the two kernels are not the same.\n";

  std::cout << "Lets run fuse and see where it takes us. We should have a tuple of 9 kernels. One for the fused bit, then one for each loop, in each dimension, for both before and after the fused bit.\n";
  

  auto fused12 = fuse_kernels(knl1, shifted2);
  
  auto firstInFused12 = camp::get<0>(fused12);
  auto secondInFused12 = camp::get<1>(fused12);
  auto thirdInFused12 = camp::get<2>(fused12);
  auto fourthInFused12 = camp::get<3>(fused12);
  auto fifthInFused12 = camp::get<4>(fused12);
  auto sixthInFused12 = camp::get<5>(fused12); 
  auto seventhInFused12 = camp::get<6>(fused12);
  auto eighthInFused12 = camp::get<7>(fused12);
  auto ninthInFused12 = camp::get<8>(fused12);
  
  
  std::cout << "For each of the 9 resulting kernels, lets look at their iteration space.\n";

  print_iteration_spaces<0,8>(fused12);
  
  std::cout << "Kernel 0 and 1 are the initial bit of knl1, while 7 and 8 are the final bits of knl2\n";

  std::cout << "As it turns out, this is a valid ordering of iterations for these kernels. We can do the initial piece of knl1, the fused bit, then the final bit of knl2. Question is: What about when we add the third knl?\n";

  std::cout << "Let's fuse Kernel4 with the third kernel. This will create the 'F12 bits', 'L3 bits', and the 'F123'.\n";

  auto f12 = fifthInFused12;

  int canFuse123 = RAJA::can_fuse<1,2>(f12, knl3);
  std::cout << "Can we fuse without a shift?: " << canFuse123 << "\n";

  std::cout << "What about if we shift knl3 by 2 2: ";
  
  auto shifted3 = RAJA::shift_kernel(knl3, RAJA::shift<2>(2,2));

  canFuse123 = RAJA::can_fuse<1,2>(f12, shifted3);

  std::cout << canFuse123 << "\n";

  auto fused123 = fuse_kernels(f12, shifted3);

   
  
} //triple_fusion_example

//example from 7.6.2020 notes
int fixed_fusion_example() {

  std::cout << "\n\n\n\n===== Fixed Fusion Example =====\n";
  int numErrors = 0;
  double * _a = allocate<double>(16*16);
  double * _b = allocate<double>(16*16);
  double * _c = allocate<double>(16*16);
  double * _d = allocate<double>(16*16);
  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a(_a,16,16);
  VIEW_TYPE2 b(_b,16,16);
  VIEW_TYPE2 c(_c,16,16);
  VIEW_TYPE2 d(_d,16,16);

  auto rangeSegment = RAJA::make_tuple(RAJA::RangeSegment(1,15), RAJA::RangeSegment(1,15));

  auto lambda1 = [=](auto i, auto j) {
    b(i,j) = (a(i-1,j) + a(i,j-1) + a(i,j) + a(i+1,j) + a(i,j+1));
    
  };
  auto lambda2 = [=](auto i, auto j) {
    c(i,j) = (b(i-1,j) + b(i,j-1) + b(i,j) + b(i+1,j) + b(i,j+1));
  };
  auto lambda3 = [=](auto i, auto j) {
    d(i,j) = (c(i-1,j) + c(i,j-1) + c(i,j) + c(i+1,j) + c(i,j+1));
  };

  using KPOL =  RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

  auto knl1 = RAJA::make_kernel<KPOL>(rangeSegment, lambda1);
  auto knl2 = RAJA::make_kernel<KPOL>(rangeSegment, lambda2);
  auto knl3 = RAJA::make_kernel<KPOL>(rangeSegment, lambda3);


  isl_ctx * ctx = isl_ctx_alloc();
  isl_printer * p = isl_printer_to_file(ctx, stdout);

  
  std::cout << "\nThis example walks through using the fixed fusion schedule approach.\n";
  std::cout << "For this example, we use three 2d jacobis into successive arrays, \n"
               "also used in triple_fusion_example\n";
  std::cout << "The iterator boundary for the loops is the 1 <= i < 15 square,\n"
              " with 16x16 arrays a, b, c, and d.";

  std::cout << "The purpose of this example is to explore functionality for generating a specific loop partition order from a sequence of kernels.\n";
 
  std::cout << "The assumptions being made in this function are:\n";
  std::cout << "  All dependences between loops are completely non-negative\n"
               "    which can be achieved by shifting\n";
  std::cout << "  Each loop is independently parallel (no deps between iterations)\n";
  
  std::cout << "Now, to the kernels.\n";

  std::cout << "We start by shifting the second kernel by (1,1) and the third by (2,2)\n"; 
  auto shifted2 = RAJA::shift_kernel(knl2, RAJA::shift<1>(1,1));
  auto shifted3 = RAJA::shift_kernel(knl3, RAJA::shift<2>(2,2));
 
  using namespace RAJA;
  std::cout << "Next, we extract the iteration spaces. ";

  auto ispace1 = iterspace_from_knl<1>(ctx, knl1);
  auto ispace2 = iterspace_from_knl<2>(ctx, shifted2);
  auto ispace3 = iterspace_from_knl<3>(ctx, shifted3);

  std::cout << "For the three loops we have:\n";

  p = isl_printer_print_union_set(p, ispace1);
  std::cout << "\n";
  p = isl_printer_print_union_set(p, ispace2);
  std::cout <<  "\n";
  p = isl_printer_print_union_set(p, ispace3);
  std::cout << "\n";  

  std::cout << "Now that the iteration spaces do not align, we intersect the iteration spaces \n"
               "to get the part of each iteration space that will be mapped to the fully fused\n"
               "loop. We should expect this space to be the 3 <= i < 15 square\n";

  isl_union_set * fusedIterSpace = fused_iterspace<2>(ctx, make_tuple(ispace1, ispace2, ispace3));
  //isl_union_set * fusedIterSpace = intersect_union_sets(ispace1, ispace2, ispace3);

  p = isl_printer_print_union_set(p, fusedIterSpace);

  std::cout << "\n";
  std::cout << "  This part of the iteration spaces will be executed by the fused kernel.  The\n"
               "rest of the iteration spaces for the loops must be executed either before or\n"
               "after the fused kernel.\n";

  std::cout << "The problem this example tackles is as follows: Partition and order the sets in\n"
               "{IterationSpace_i - Intersection(I_1,I_2,...,I_n) | i <= n}.\n";
  std::cout << "The partition must split the iteration spaces of all the loops into\n"
               "hyper-rectangles. The ordering must respect the dependences across loops.\n";
  
  std::cout << "\nThus, we want to create a schedule which is possibly such an order.\n";
  std::cout << "Part of this schedule is the fused loop, and the rest is from the\n" 
               "partition of the iteration space leftovers.\n"; 
 
  std::cout << "The first part of the partition is the initial segment of each kernel.\n";
  std::cout << "This is the stuff from the lowest point of the iteration space up to the start of the fused part of the kernel in each dimension.\n";
  std::cout << "There is one kernel to execute this part of the iteration space for each dimension of each original kernel, so for this example, we expect to see 6 such kernels\n";

  std::cout << "We are seeking the schedule constraints for these kernels, so in our eventual schedule, these kernels will occupy the first n * d loops in the schedule's range, where the range is [0,i0,0,i1,0] for the first statement of the first 2d loop, and [0,i0,0,i1,1] for the second.\n";
  std::cout << "These constraints have two parts: mappings and iterator bounds. We start with the iterator bounds.\n";
  

  std::cout << "\n\nThe very first thing executed is the initial segment of the first loop.\n";
  std::cout << "We expect the bounds of this loop to be the part of the 1 <= i < 15 square where i0 is less than 3";
  
  std::cout << "\nFirst we take the entire iteration space for the first kernel. Then we make the first cut using the i0 dimension. We maintain the same working iteration space, cutting pieces off which are the iteration spaces of the initial segment loops.\n";


  isl_union_set * workingIterSpace = iterspace_from_knl<1>(ctx, knl1);
    
  std::cout << "Starting Iteration Space: ";
  p = isl_printer_print_union_set(p, workingIterSpace);

  isl_union_set * constraintSet1 = pre_constraint_set<1,0,2>(ctx, fusedIterSpace);

  std::cout << "\nFirst Constraint Set: ";
  p= isl_printer_print_union_set(p, constraintSet1);
  
  isl_union_set * knl1pre1IterationSpace = isl_union_set_intersect(isl_union_set_copy(workingIterSpace), constraintSet1);

  std::cout << "\nFirst Initial Segment Iteration Space: ";
  p = isl_printer_print_union_set(p, knl1pre1IterationSpace);

  workingIterSpace = isl_union_set_subtract(workingIterSpace, knl1pre1IterationSpace);
  
  std::cout << "\nNew Working Iteration Space: ";
  p = isl_printer_print_union_set(p, workingIterSpace);

  std::cout << "\nNow we apply the second constraint set, which is the remaining part where i1 < 3.\n";

  isl_union_set * constraintSet2 = pre_constraint_set<1,1,2>(ctx, fusedIterSpace);

  std::cout << "\nConstraint Set 2:\n";
  
  p = isl_printer_print_union_set(p , constraintSet2);
  std::cout << "\n"; 

  isl_union_set * knl1pre2IterationSpace = isl_union_set_intersect(isl_union_set_copy(workingIterSpace), constraintSet2);
 
   std::cout << "\nSecond Initial Segment Iteration Space: ";
  p = isl_printer_print_union_set(p, knl1pre2IterationSpace);

  workingIterSpace = isl_union_set_subtract(workingIterSpace, knl1pre2IterationSpace);

  std::cout << "\nNew Working Iteration Space: ";
  p = isl_printer_print_union_set(p, workingIterSpace);

  auto knl1preIterspaces = pre_iterspaces_from_knl<1>(ctx, fusedIterSpace, knl1);

  std::cout << "\nInitial segment iteration spaces for kernel 1 using the function instead of doing it by hand:\n";
 
  p = isl_printer_print_union_set(p, camp::get<0>(knl1preIterspaces));
  std::cout << "\n";  
  p = isl_printer_print_union_set(p, camp::get<1>(knl1preIterspaces));
  std::cout << "\n";

  std::cout << "Now, using the pre_iterspaces_from_knl function across each kernel, we end up with a list of iteration spaces\n";

  auto initialSegmentIterationSpaces = pre_iterspaces_from_knls(ctx, fusedIterSpace, make_tuple(knl1, shifted2, shifted3));

  auto segs = initialSegmentIterationSpaces;

  std::cout << "Iteration spaces in list:\n";
  p = isl_printer_print_union_set(p, camp::get<0>(segs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<1>(segs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<2>(segs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<3>(segs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<4>(segs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<5>(segs));
  std::cout << "\n";

  std::cout << "These results are a bit surprising, because I expect the L1 intersecting with L0 emptying the entire set.\n";

 
  std::cout << "\nRegardless, next we extract the same info for the post-fused kernels. This can't just be the flipped sign version of the initial segment way. We have to change the initial working set to not include the corner bits twice.\n";

  
  std::cout << "\nSo we start with the complete iteration space for the kernel. For knl1:\n";

  auto fullIterspace = iterspace_from_knl<1>(ctx, knl1);

  p = isl_printer_print_union_set(p, fullIterspace);

  std::cout << "Really, the initial set should be the very last working set while doing the initial segs\n";

  
  











  auto concludingSegmentIterationSpaces = post_iterspaces_from_knls(ctx, fusedIterSpace, make_tuple(knl1, shifted2, shifted3));

  auto cegs = concludingSegmentIterationSpaces;
  p = isl_printer_print_union_set(p, camp::get<0>(cegs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<1>(cegs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<2>(cegs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<3>(cegs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<4>(cegs));
  std::cout << "\n";
  p = isl_printer_print_union_set(p, camp::get<5>(cegs));
  std::cout << "\n";


  auto partitionedAndOrderedIterspaceTuple = tuple_cat(segs, make_tuple(fusedIterSpace), cegs);

  std::cout << "\nProblem Complete. the iteration spaces have been partitioned and ordered\n";
  
  std::cout << "\nNext step is to turn this ordering into a schedule\n";

  std::cout << "\nEach iteration space in the non-fused area is mappped to [index, i,0,j,0] and the fused area is mapped to [index, i,0,j,loopnum]\n";

  std::string scheduleString1 = "";//partition_to_schedule<0,1>(camp::get<0>(partitionedAndOrderedIterspaceTuple));	     
  scheduleString1 += "{";
  scheduleString1 += range_vector(2,0);
  scheduleString1 += " -> ";
  scheduleString1 += "[";
  scheduleString1 += "0";
  scheduleString1 += ",i0,0,i1,0] }";
  
  isl_union_map * schedule1 = isl_union_map_read_from_str(ctx, scheduleString1.c_str());
  
  std::cout << "Schedule for the first loop to be executed, without the iterator bounds\n";
  p = isl_printer_print_union_map(p, schedule1);

  std::cout << "\n";

  std::cout << "Iterspace that provide the iterator bounds\n";
  p = isl_printer_print_union_set(p, camp::get<0>(partitionedAndOrderedIterspaceTuple));
  std::cout << "\n";
  
  schedule1 = isl_union_map_intersect_domain(schedule1, isl_union_set_copy(camp::get<0>(partitionedAndOrderedIterspaceTuple)));

  std::cout << "Schedule after the iterator bounds have been added\n";

  p = isl_printer_print_union_map(p,schedule1);
  std::cout << "\n";
  
  std::cout << "Schedule for first loop from function\n";
  p = isl_printer_print_union_map(p,schedule_from_partition<0,2,0>(ctx, camp::get<0>(partitionedAndOrderedIterspaceTuple)));

  std::cout << "\n";

  auto schedules = schedules_from_partitions(ctx, partitionedAndOrderedIterspaceTuple);

  std::cout << "Schedules\n";
  p = isl_printer_print_union_map(p, camp::get<0>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<1>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<2>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<3>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<4>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<5>(schedules));
  std::cout << "\n";
  std::cout << "Fused Schedule:\n";
  p = isl_printer_print_union_map(p, camp::get<6>(schedules));
  std::cout << "\nSchedules:\n";
  p = isl_printer_print_union_map(p, camp::get<7>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<8>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<9>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<10>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<11>(schedules));
  std::cout << "\n";
  p = isl_printer_print_union_map(p, camp::get<12>(schedules));
  std::cout << "\n";

  auto fusedSchedule = union_union_map_tuple(schedules);

  std::cout << "Combined Schedule:\n";
  p = isl_printer_print_union_map(p, fusedSchedule);
} //fixed_fusion_example


//doing the same thing as the previous example function again
int fixed_fusion_example_2() {

  std::cout << "\n\n\n\n===== Fixed Fusion Example 2=====\n";
  int numErrors = 0;
  double * _a = allocate<double>(16*16);
  double * _b = allocate<double>(16*16);
  double * _c = allocate<double>(16*16);
  double * _d = allocate<double>(16*16);

 
  isl_ctx * ctx = isl_ctx_alloc();
  isl_printer * p = isl_printer_to_file(ctx, stdout);

  using namespace RAJA;

  using VIEW_TYPE2 = View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a(_a,16,16);
  VIEW_TYPE2 b(_b,16,16);
  VIEW_TYPE2 c(_c,16,16);
  VIEW_TYPE2 d(_d,16,16);

  auto rangeSegment1 = RAJA::make_tuple(RAJA::RangeSegment(3,12), RAJA::RangeSegment(3,12));
  auto rangeSegment2 = RAJA::make_tuple(RAJA::RangeSegment(2,10), RAJA::RangeSegment(1,15));
  auto rangeSegment3 = RAJA::make_tuple(RAJA::RangeSegment(1,15), RAJA::RangeSegment(1,15));
 

  auto lambda1 = [=](auto i, auto j) {
    b(i,j) = (a(i-1,j) + a(i,j-1) + a(i,j) + a(i+1,j) + a(i,j+1));
    
  };
  auto lambda2 = [=](auto i, auto j) {
    c(i,j) = (b(i-1,j) + b(i,j-1) + b(i,j) + b(i+1,j) + b(i,j));
  };
  auto lambda3 = [=](auto i, auto j) {
    d(i,j) = (c(i-1,j) + c(i,j-1) + c(i,j) + c(i+1,j) + c(i,j+1)) + b(i,j);
  };

  using KPOL =  RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

  auto knl1 = RAJA::make_kernel<KPOL>(rangeSegment1, lambda1);
  auto knl2 = RAJA::make_kernel<KPOL>(rangeSegment2, lambda2);
  auto knl3 = RAJA::make_kernel<KPOL>(rangeSegment3, lambda3);

   auto rangeSegs = make_tuple(rangeSegment1, rangeSegment2, rangeSegment3);
  auto knls = make_tuple(knl1,knl2,knl3);

  std::cout << "In this example, two complications are added.\n"
            << "1st: The range segments are not equal. I.E. The iteration spaces are the not same for each loop.\n"; 
  std::cout << "2nd: The dependences aren't all 1->2 2->3 ... n-1->n. We also have 1->3.\n";

  std::cout << "The final product here is to get a yes or no for fusion legality.\n";
  std::cout << "We will be doing so using the dependences between the loops in their original forms, and a fused schedule for the knls.\n";
  
  std::cout << "We get the dependences between the kernels by taking the dependences between the pairs of them.\n";

  isl_union_map * knlDeps = dependence_relation_from_kernels(ctx, knls);
  auto originalSchedule = original_schedule_from_kernels(ctx, knls);

  std::cout << "Dependence Relation Among Kernels\n";

  p = isl_printer_print_union_map(p, knlDeps);
  std::cout << "\n";
 
  
  




}//fixed_fusion_example_2

int fused_scheduling_example() {

  std::cout << "\n\n\n\n===== Fused Scheduling Example =====\n";
  int numErrors = 0;
  double * _a = allocate<double>(16*16);
  double * _b = allocate<double>(16*16);
  double * _c = allocate<double>(16*16);
  double * _d = allocate<double>(16*16);
  using VIEW_TYPE2 = RAJA::View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a(_a,16,16);
  VIEW_TYPE2 b(_b,16,16);
  VIEW_TYPE2 c(_c,16,16);
  VIEW_TYPE2 d(_d,16,16);

  auto rangeSegment = RAJA::make_tuple(RAJA::RangeSegment(1,15), RAJA::RangeSegment(1,15));

  auto lambda1 = [=](auto i, auto j) {
    b(i,j) = (a(i-1,j) + a(i,j-1) + a(i,j) + a(i+1,j) + a(i,j+1));
    
  };
  auto lambda2 = [=](auto i, auto j) {
    c(i,j) = (b(i-1,j) + b(i,j-1) + b(i,j) + b(i+1,j) + b(i,j+1));
  };
  auto lambda3 = [=](auto i, auto j) {
    d(i,j) = (c(i-1,j) + c(i,j-1) + c(i,j) + c(i+1,j) + c(i,j+1));
  };

  using KPOL =  RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::omp_parallel_for_exec,
      RAJA::statement::For<1,RAJA::omp_parallel_for_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

  auto knl1 = RAJA::make_kernel<KPOL>(rangeSegment, lambda1);
  auto knl2 = RAJA::make_kernel<KPOL>(rangeSegment, lambda2);
  auto knl3 = RAJA::make_kernel<KPOL>(rangeSegment, lambda3);
 
  using namespace RAJA;
  auto shifted2 = shift_kernel(knl2, shift<2>(2,2));
  auto shifted3 = shift_kernel(knl3, shift<2>(-3,1));

  
  auto knlTuple = make_tuple(knl1, shifted2, shifted3);

  isl_ctx * ctx = isl_ctx_alloc();
  isl_printer * p = isl_printer_to_file(ctx, stdout);

  auto ispace1 = iterspace_from_knl<1>(ctx, knl1);
  auto ispace2 = iterspace_from_knl<2>(ctx, shifted2);
  auto ispace3 = iterspace_from_knl<3>(ctx, shifted3);

  std::cout << "Iteration spaces of loops to fuse\n";
  p = isl_printer_print_union_set(p, ispace1);
  std::cout << "\n";
  p = isl_printer_print_union_set(p, ispace2);
  std::cout <<  "\n";
  p = isl_printer_print_union_set(p, ispace3);
  std::cout << "\n";  

  auto iterspaceTuple = make_tuple(ispace1,ispace2, ispace3);
  std::cout << "1) get the boundary values for the fused portion (min/max for each dimension)\n";
  
  isl_union_set * fusedSpace = fused_iterspace<2>(ctx, iterspaceTuple);
  
  std::cout << "Fused space:\n";
  p = isl_printer_print_union_set(p, fusedSpace);

  std::cout << "\n";
  std::cout << "2) change the boundaries of each iteration space to those\n";

  std::cout << "3) union the iteration space for each loop\n";

  
  
  
} //fused_scheduling_example()


int shift_amount_example() {

  

} //shift_amount_example


int overlapped_tile_example() {

  std::cout << "\n\n\n\n===== Overlapped Tile Example=====\n";
  int numErrors = 0;
  double * _a = allocate<double>(16*16);
  double * _b = allocate<double>(16*16);
  double * _c = allocate<double>(16*16);
  double * _d = allocate<double>(16*16);

 
  isl_ctx * ctx = isl_ctx_alloc();
  isl_printer * p = isl_printer_to_file(ctx, stdout);

  using namespace RAJA;

  using VIEW_TYPE2 = View<double, RAJA::Layout<2>>;
  VIEW_TYPE2 a(_a,16,16);
  VIEW_TYPE2 b(_b,16,16);
  VIEW_TYPE2 c(_c,16,16);
  VIEW_TYPE2 d(_d,16,16);

  auto rangeSegment = RAJA::make_tuple(RAJA::RangeSegment(1,3), RAJA::RangeSegment(1,3));
 

  auto lambda1 = [=](auto i, auto j) {
    b(i,j) = (a(i-1,j) + a(i,j-1) + a(i,j) + a(i+1,j) + a(i,j+1));
    
  };
  auto lambda2 = [=](auto i, auto j) {
    c(i,j) = (b(i-1,j) + b(i,j-1) + b(i,j) + b(i+1,j) + b(i,j));
  };
  auto lambda3 = [=](auto i, auto j) {
    d(i,j) = (c(i-1,j) + c(i,j-1) + c(i,j) + c(i+1,j) + c(i,j+1)) + b(i,j);
  };

  using KPOL =  RAJA::KernelPolicy<
    RAJA::statement::For<0,RAJA::seq_exec,
      RAJA::statement::For<1,RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >
  >;

  auto knl1 = RAJA::make_kernel<KPOL>(rangeSegment, lambda1);
  auto knl2 = RAJA::make_kernel<KPOL>(rangeSegment, lambda2);
  auto knl3 = RAJA::make_kernel<KPOL>(rangeSegment, lambda3);

  auto shifted2 = RAJA::shift_kernel(knl2, RAJA::shift<1>(1,1));
  auto shifted3 = RAJA::shift_kernel(knl3, RAJA::shift<2>(2,2));

  using namespace RAJA;
 
  auto knlTuple = make_tuple(knl1, shifted2, shifted3);

  std::cout << "The overlap amounts are the following tuples:\n";

  auto overlaps1 = make_tuple(4,4);
  auto overlaps2 = make_tuple(2,2);
  auto overlaps3 = make_tuple(0,0); 

  auto overlaps = make_tuple(overlaps1, overlaps2, overlaps3);
  std::cout << "(4,4), (2,2), (0,0)\n";
  
  auto forNestExecPol = for_nest_exec_pol<2,0,0>();


  auto pk = [=](auto i, auto j)  {std::cout << "called: " << i << ", " << j << "\n";};

  RAJA::kernel<RAJA::KernelPolicy<decltype(forNestExecPol)>>(make_tuple(RangeSegment(0,3), RangeSegment(0,3)), pk);


  auto forNestKernelPol = overlapped_tiling_inner_exec_pol<1,2>();
  
  RAJA::kernel<decltype(forNestKernelPol)>(make_tuple(RangeSegment(0,3), RangeSegment(0,3)), pk);

  std::cout << "Creating execution policy for 2 2d for loops.\n";

  auto forNestKernelPol2 = overlapped_tiling_inner_exec_pol<2,2>();
  

  std::cout << "Cannot have a bunch of range segments because the lambdas need to have arguments for as many range segment there are.\n";

  std::cout << "\n the new plan is to have a lambda which takes RangeSegments as arguments and executes the tile for those range segments\n";

  std::cout << "executorLambda is such a lambda for 3 functions: pk1, pk2 and pk3\n";

  auto pk1 = make_kernel<KPOL>(rangeSegment, [=](auto i, auto j)  {std::cout << "pk1: " << i << ", " << j << "\n";});
  auto pk2 = make_kernel<KPOL>(rangeSegment, [=](auto i, auto j)  {std::cout << "pk2: " << i << ", " << j << "\n";});
  auto pk3 = make_kernel<KPOL>(rangeSegment, [=](auto i, auto j)  {std::cout << "pk3: " << i << ", " << j << "\n";});

  std::cout << "Executing the three kernels without the executor lambda\n";

  pk1();
  pk2();
  pk3();

  std::cout << "Now executing them with the executor lambda\n";
  
  auto executorLambda = [=](auto iRange, auto jRange) {
    kernel<KPOL>(make_tuple(iRange,jRange), camp::get<0>(pk1.bodies));
    kernel<KPOL>(make_tuple(iRange,jRange), camp::get<0>(pk2.bodies));
    kernel<KPOL>(make_tuple(iRange,jRange), camp::get<0>(pk3.bodies));
  };  

  executorLambda(RangeSegment(1,2), RangeSegment(1,3));
  executorLambda(RangeSegment(2,3), RangeSegment(1,3));

  std::cout << "Now that I have this idea, I want to incorporate the overlap amounts into the lambda\n";

  auto executorWithOverlaps = [=](auto iRange, auto jRange) {
   auto iRange1 = add_overlap(iRange, camp::get<0>(camp::get<0>(overlaps)));
   auto jRange1 = add_overlap(jRange, camp::get<1>(camp::get<0>(overlaps)));
   auto segments1 = make_tuple(iRange1, jRange1);
   auto newKnl1 = make_kernel<KPOL>(segments1, camp::get<0>(pk1.bodies));

   auto iRange2 = add_overlap(iRange, camp::get<0>(camp::get<1>(overlaps)));
   auto jRange2 = add_overlap(jRange, camp::get<1>(camp::get<1>(overlaps)));
   auto segments2 = make_tuple(iRange2, jRange2);
   auto newKnl2 = make_kernel<KPOL>(segments2, camp::get<0>(pk2.bodies));

   auto iRange3 = add_overlap(iRange, camp::get<0>(camp::get<2>(overlaps)));
   auto jRange3 = add_overlap(jRange, camp::get<1>(camp::get<2>(overlaps)));
   auto segments3 = make_tuple(iRange3, jRange3);
   auto newKnl3 = make_kernel<KPOL>(segments3, camp::get<0>(pk3.bodies));

   newKnl1();
   newKnl2();
   newKnl3();
  };


  std::cout << "Executing pks with overlap included\n";

  std::cout << "Tile 1\n";
  executorWithOverlaps(RangeSegment(1,2), RangeSegment(1,3));
  std::cout << "Tile 2\n";
  executorWithOverlaps(RangeSegment(2,3), RangeSegment(1,3));
  
  std::cout << "What we want in the end is a function that takes kernels and overlap tuples and does the execution for a tile, aka a function that takes kernels and overlaps and returns the overlap executor run above.\n";

  std::cout << "That function is overlapped_tile_executor\n";
  auto executorFunc = overlapped_tile_executor(make_tuple(pk1,pk2,pk3), overlaps);

  executorFunc(make_tuple(RangeSegment(-4,0), RangeSegment(-2,2)));
  
  /*
  std::cout << "Next question: Can we execute this executor function as the lambda within a kernel of its own?\n";

  std::cout << "Lets start by creating an iterator of range segments\n";

  std::vector<RangeSegment> iRanges = {RangeSegment(1,2), RangeSegment(2,3)};
  std::vector<RangeSegment> jRanges = {RangeSegment(1,2), RangeSegment(2,3)};
  
  std::cout << "Executing overlapped tile executor using raja kernel over two iterators of range segments\n";

  kernel<KPOL>(RAJA::make_tuple(iRanges,jRanges), executorFunc);
  */


  using TPOL = KernelPolicy<
    statement::OverlappedTile<0,RAJA::statement::tile_fixed<2>, RAJA::seq_exec,
      statement::OverlappedTile<1,RAJA::statement::tile_fixed<2>, RAJA::seq_exec,
        statement::TiledLambda<0>
      >
    >
  >;
  std::cout << "Executing overlapped tiled lambdas using the new statement types\n";
  kernel<TPOL>(make_tuple(RangeSegment(0,4), RangeSegment(0,4)), executorFunc);

  

}//overlapped_tile_example

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
  std::cerr << "ISL Errors: " << islErrors<< "\n\n";
  numErrors += islErrors;

  std::cout << "Total error count: " << numErrors;

  //fixed_fusion_example_2();
  //fused_scheduling_example();
  overlapped_tile_example();
  return numErrors;
}
