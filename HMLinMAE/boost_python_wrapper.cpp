//
//  boost_python_wrapper.cpp
//  HMVAR
//
//  Created by Brandon Kelly on 9/20/13.
//  Copyright (c) 2013 Brandon Kelly. All rights reserved.
//

#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif

#include "Python.h"
#include <boost/intrusive/options.hpp>
#include <boost/python.hpp>
#include <boost/python/object.hpp>
#include <boost/python/object/function_object.hpp>
#include <boost/python/object/py_function.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <armadillo>
#include <utility>
#include "numpy/ndarrayobject.h"
#include "MaeGibbs.hpp"

typedef std::vector<double> vec1d;
typedef std::vector<vec1d> vec2d;
typedef std::vector<vec2d> vec3d;

using namespace boost::python;
using namespace HMLinMAE;

BOOST_PYTHON_MODULE(lib_hmlinmae){
    import_array();
    numeric::array::set_module_and_type("numpy", "ndarray");
    
    class_<vec1d>("vec1D")
    .def(vector_indexing_suite<vec1d>());
    
    class_<vec2d>("vec2D")
    .def(vector_indexing_suite<vec2d>());
    
    class_<vec3d>("vec3D")
    .def(vector_indexing_suite<vec3d>());
    
    // MaeGibbs.hpp
    class_<MaeGibbs, boost::noncopyable>("MaeGibbs", no_init)
    .def(init<int, vec2d, vec3d>())
    .def("RunMCMC", &MaeGibbs::RunMCMC)
    .def("GetCoefsMean", &MaeGibbs::GetCoefsMean)
    .def("GetNoiseMean", &MaeGibbs::GetNoiseMean)
    .def("GetCoefs", &MaeGibbs::GetCoefs)
    .def("GetSigSqr", &MaeGibbs::GetSigSqr)
    .def("GetWeights", &MaeGibbs::GetWeights);
};

