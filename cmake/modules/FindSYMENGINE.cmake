## ---------------------------------------------------------------------
##
## Copyright (C) 2019 - 2022 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------

#
# - Try to find SymEngine
#
# This module exports
#
#   SYMENGINE_INCLUDE_DIR
#   SYMENGINE_LIBRARY
#   SYMENGINE_WITH_LLVM
#

set(SYMENGINE_DIR "" CACHE PATH "An optional hint to a SymEngine installation")
set_if_empty(SYMENGINE_DIR "$ENV{SYMENGINE_DIR}")

#
# SymEngine overwrites the CMake module path, so we save
# and restore it after this library is found and configured.
#
set (DEAL_II_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})

#
# Include the SymEngine:
#
find_package(SymEngine
  CONFIG QUIET
  HINTS ${SYMENGINE_DIR}
  PATH_SUFFIXES lib/cmake/symengine
  NO_SYSTEM_ENVIRONMENT_PATH
  )

#
# Reset the CMake module path
#
set (CMAKE_MODULE_PATH ${DEAL_II_CMAKE_MODULE_PATH})


#
# Look for symengine_config.h - we'll query it to determine supported features:
#
if(SymEngine_FOUND)
  deal_ii_find_file(SYMENGINE_SETTINGS_H symengine_config.h
    HINTS ${SYMENGINE_INCLUDE_DIRS}
    PATH_SUFFIXES symengine
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    )
endif()

#
# Version check
#
if(EXISTS ${SYMENGINE_SETTINGS_H})

  file(STRINGS "${SYMENGINE_SETTINGS_H}" SYMENGINE_VERSION_MAJOR_STRING
    REGEX "#define.*SYMENGINE_MAJOR_VERSION")
  string(REGEX REPLACE "^.*SYMENGINE_MAJOR_VERSION.*([0-9]+).*" "\\1"
    SYMENGINE_VERSION_MAJOR "${SYMENGINE_VERSION_MAJOR_STRING}"
    )
  file(STRINGS "${SYMENGINE_SETTINGS_H}" SYMENGINE_VERSION_MINOR_STRING
    REGEX "#define.*SYMENGINE_MINOR_VERSION")
  string(REGEX REPLACE "^.*SYMENGINE_MINOR_VERSION.*([0-9]+).*" "\\1"
    SYMENGINE_VERSION_MINOR "${SYMENGINE_VERSION_MINOR_STRING}"
    )
  file(STRINGS "${SYMENGINE_SETTINGS_H}" SYMENGINE_VERSION_PATCH_STRING
    REGEX "#define.*SYMENGINE_PATCH_VERSION")
  string(REGEX REPLACE "^.*SYMENGINE_PATCH_VERSION.*([0-9]+).*" "\\1"
    SYMENGINE_VERSION_PATCH "${SYMENGINE_VERSION_PATCH_STRING}"
    )
    
  set(SYMENGINE_VERSION ${SymEngine_VERSION})
endif()

#
# Feature checks
#

macro(_symengine_feature_check _var _regex)
  if(EXISTS ${SYMENGINE_SETTINGS_H})
    file(STRINGS "${SYMENGINE_SETTINGS_H}" SYMENGINE_${_var}_STRING
      REGEX "${_regex}")
    if("${SYMENGINE_${_var}_STRING}" STREQUAL "")
      set(SYMENGINE_WITH_${_var} FALSE)
    else()
      set(SYMENGINE_WITH_${_var} TRUE)
    endif()
  endif()
endmacro()

# Other possible features of interest: BOOST, GMP
_symengine_feature_check(LLVM "#define.*HAVE_SYMENGINE_LLVM")

#
# Sanitize include dirs:
#

string(REGEX REPLACE
  "(lib64|lib)\\/cmake\\/symengine\\/\\.\\.\\/\\.\\.\\/\\.\\.\\/" ""
  SYMENGINE_INCLUDE_DIRS  "${SYMENGINE_INCLUDE_DIRS}"
  )
remove_duplicates(SYMENGINE_INCLUDE_DIRS)

#
# Get the full path for the SYMENGINE_LIBRARIES. Some of these libraries are
# CMake targets, so we can query them directly for this information.
#
foreach(SYMENGINE_LIBRARY_NAME ${SYMENGINE_LIBRARIES})
   if (TARGET ${SYMENGINE_LIBRARY_NAME})
       get_property(SYMENGINE_LIBRARY TARGET ${SYMENGINE_LIBRARY_NAME} PROPERTY LOCATION)
   else ()
       set(SYMENGINE_LIBRARY ${SYMENGINE_LIBRARY_NAME})
   endif()

  set(_symengine_libraries ${_symengine_libraries} ${SYMENGINE_LIBRARY})
endforeach()
set(SYMENGINE_LIBRARIES ${_symengine_libraries})

deal_ii_package_handle(SYMENGINE
  LIBRARIES REQUIRED SYMENGINE_LIBRARIES
  INCLUDE_DIRS REQUIRED SYMENGINE_INCLUDE_DIRS
  USER_INCLUDE_DIRS REQUIRED SYMENGINE_INCLUDE_DIRS
  CLEAR SYMENGINE_SETTINGS_H SYMENGINE_SKIP_DEPENDENCIES SymEngine_DIR
)
