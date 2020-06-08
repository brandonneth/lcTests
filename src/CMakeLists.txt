###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(tests_depends RAJA)

if (ENABLE_OPENMP)
  set(tests_depends 
    ${tests_depends}
    openmp)
endif ()

if (ENABLE_CUDA)
  set(tests_depends 
    ${tests_depends}
    cuda)
endif ()

blt_add_executable(
  NAME tests.exe
  SOURCES tests.cpp
  DEPENDS_ON ${tests_depends})
