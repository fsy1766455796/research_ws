#include <cmath>
#include <memory>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::tuple align_gicp(py::array_t<double, py::array::c_style | py::array::forcecast> source,
                     py::array_t<double, py::array::c_style | py::array::forcecast> target,
                     double max_correspondence, int iterations) {
  auto s = source.unchecked<2>();
  auto t = target.unchecked<2>();
  if (s.shape(1) != 3 || t.shape(1) != 3) throw std::runtime_error("clouds must be Nx3");
  auto src = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  auto dst = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  src->reserve(s.shape(0)); dst->reserve(t.shape(0));
  for (py::ssize_t i = 0; i < s.shape(0); ++i)
    src->emplace_back(s(i, 0), s(i, 1), s(i, 2));
  for (py::ssize_t i = 0; i < t.shape(0); ++i)
    dst->emplace_back(t(i, 0), t(i, 1), t(i, 2));

  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setInputSource(src); gicp.setInputTarget(dst);
  gicp.setMaxCorrespondenceDistance(max_correspondence);
  gicp.setMaximumIterations(iterations);
  gicp.setTransformationEpsilon(1e-5);
  gicp.setRotationEpsilon(1e-5);
  pcl::PointCloud<pcl::PointXYZ> aligned;
  {
    py::gil_scoped_release release;
    gicp.align(aligned);
  }
  const Eigen::Matrix4f m = gicp.getFinalTransformation();
  const double yaw = std::atan2(m(1, 0), m(0, 0));
  return py::make_tuple(static_cast<double>(m(0, 3)), static_cast<double>(m(1, 3)), yaw,
                        gicp.getFitnessScore(max_correspondence), gicp.hasConverged());
}

PYBIND11_MODULE(_pcl_gicp, module) {
  module.doc() = "Minimal PCL GICP binding for planar odometry initialization";
  module.def("align", &align_gicp, py::arg("source"), py::arg("target"),
             py::arg("max_correspondence") = 0.45, py::arg("iterations") = 30);
}
