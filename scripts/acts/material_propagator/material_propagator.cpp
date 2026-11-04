// acts_material_propagator: ACTS (unmodified, linked as an external library) propagating track states to a
// perigee surface through the tracking geometry WITH material effects, for acts_worker.py. ACTS's core Python
// bindings only offer propagateToSurface without actors (no MaterialInteractor); this module adds that one
// call, nothing else. It takes the geometry and field objects ACTS's own Python bindings build (same pybind11,
// built against the same ACTS libraries), so geometry, material map and field map stay configured in Python.
//
// Units are ACTS's (mm, GeV, e). Tracks are propagated in their real direction: a state whose perigee lies
// behind it (e.g. at the innermost hit) is propagated with Direction::Backward(), so energy loss is added back
// (MaterialInteractor: "in forward(backward) propagation, energy decreases(increases)"). Without covariance,
// multiple scattering changes nothing (it only inflates the covariance); energy loss changes q/p.
#include <Acts/Definitions/Algebra.hpp>
#include <Acts/Definitions/Direction.hpp>
#include <Acts/EventData/BoundTrackParameters.hpp>
#include <Acts/EventData/ParticleHypothesis.hpp>
#include <Acts/Geometry/GeometryContext.hpp>
#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/MagneticField/MagneticFieldContext.hpp>
#include <Acts/MagneticField/MagneticFieldProvider.hpp>
#include <Acts/Propagator/ActorList.hpp>
#include <Acts/Propagator/EigenStepper.hpp>
#include <Acts/Propagator/MaterialInteractor.hpp>
#include <Acts/Propagator/Navigator.hpp>
#include <Acts/Propagator/Propagator.hpp>
#include <Acts/Propagator/StandardAborters.hpp>
#include <Acts/Surfaces/PerigeeSurface.hpp>
#include <Acts/Surfaces/Surface.hpp>
#include <Acts/Utilities/Logger.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

class MaterialPropagator {
 public:
  using Propagator = Acts::Propagator<Acts::EigenStepper<>, Acts::Navigator>;
  using Options = Propagator::Options<Acts::ActorList<Acts::MaterialInteractor, Acts::EndOfWorldReached>>;

  MaterialPropagator(std::shared_ptr<Acts::TrackingGeometry> geometry,
                     std::shared_ptr<Acts::MagneticFieldProvider> field, std::array<double, 3> reference,
                     bool energyLoss, bool multipleScattering, std::size_t maxSteps)
      : m_geometry(std::move(geometry)),
        m_field(std::move(field)),
        m_propagator(Acts::EigenStepper<>(m_field), makeNavigator(m_geometry),
                     Acts::getDefaultLogger("MaterialPropagator", Acts::Logging::WARNING)),
        m_surface(Acts::Surface::makeShared<Acts::PerigeeSurface>(
            Acts::Vector3(reference[0], reference[1], reference[2]))),
        m_energyLoss(energyLoss),
        m_multipleScattering(multipleScattering),
        m_maxSteps(maxSteps) {}

  // (n, 6) perigee parameters (d0, z0, phi, theta, q/p, t; NaN where ACTS failed or the input is not
  // finite), ok (n,) and {error message: count}
  py::tuple propagate(py::array_t<double, py::array::c_style | py::array::forcecast> position,
                      py::array_t<double, py::array::c_style | py::array::forcecast> direction,
                      py::array_t<double, py::array::c_style | py::array::forcecast> qop,
                      py::array_t<bool, py::array::c_style | py::array::forcecast> backward) const {
    const py::ssize_t n = qop.size();
    if (position.ndim() != 2 || position.shape(0) != n || position.shape(1) != 3 || direction.ndim() != 2 ||
        direction.shape(0) != n || direction.shape(1) != 3 || backward.size() != n) {
      throw std::invalid_argument("position, direction (n, 3), qop, backward (n,) expected");
    }
    py::array_t<double> params({n, py::ssize_t{6}});
    py::array_t<bool> ok(n);
    auto P = position.unchecked<2>();
    auto D = direction.unchecked<2>();
    auto Q = qop.unchecked<1>();
    auto B = backward.unchecked<1>();
    auto out = params.mutable_unchecked<2>();
    auto good = ok.mutable_unchecked<1>();
    std::map<std::string, std::size_t> errors;
    {
      py::gil_scoped_release release;
      for (py::ssize_t i = 0; i < n; ++i) {
        for (int k = 0; k < 6; ++k) {
          out(i, k) = std::numeric_limits<double>::quiet_NaN();
        }
        good(i) = false;
        const Acts::Vector3 pos(P(i, 0), P(i, 1), P(i, 2));
        const Acts::Vector3 dir(D(i, 0), D(i, 1), D(i, 2));
        if (!pos.allFinite() || !dir.allFinite() || !std::isfinite(Q(i)) || dir.norm() == 0.0) {
          continue;
        }
        try {
          const auto start = Acts::BoundTrackParameters::createCurvilinear(
              Acts::Vector4(pos.x(), pos.y(), pos.z(), 0.0), dir.normalized(), Q(i), std::nullopt,
              Acts::ParticleHypothesis::pion());
          Options options(m_geometryContext, m_fieldContext);
          options.direction = B(i) ? Acts::Direction::Backward() : Acts::Direction::Forward();
          if (m_maxSteps > 0) {
            options.maxSteps = m_maxSteps;
          }
          auto& interactor = options.actorList.get<Acts::MaterialInteractor>();
          interactor.energyLoss = m_energyLoss;
          interactor.multipleScattering = m_multipleScattering;
          auto result = m_propagator.propagate<Options, Acts::ForcedSurfaceReached, Acts::PathLimitReached>(
              start, *m_surface, options);
          if (!result.ok()) {
            ++errors[result.error().message()];
            continue;
          }
          const auto& end = result.value().endParameters;
          if (!end.has_value()) {
            ++errors["perigee surface not reached"];
            continue;
          }
          const auto values = end->parameters();
          for (int k = 0; k < 6; ++k) {
            out(i, k) = values[k];
          }
          good(i) = values.allFinite();
        } catch (const std::exception& e) {
          ++errors[e.what()];
        }
      }
    }
    return py::make_tuple(params, ok, errors);
  }

 private:
  static Acts::Navigator makeNavigator(std::shared_ptr<const Acts::TrackingGeometry> geometry) {
    Acts::Navigator::Config config;
    config.trackingGeometry = std::move(geometry);
    config.resolveMaterial = true;
    config.resolveSensitive = true;
    config.resolvePassive = false;
    return Acts::Navigator(config, Acts::getDefaultLogger("Navigator", Acts::Logging::WARNING));
  }

  std::shared_ptr<Acts::TrackingGeometry> m_geometry;
  std::shared_ptr<Acts::MagneticFieldProvider> m_field;
  Acts::GeometryContext m_geometryContext = Acts::GeometryContext::dangerouslyDefaultConstruct();
  Acts::MagneticFieldContext m_fieldContext{};
  Propagator m_propagator;
  std::shared_ptr<Acts::PerigeeSurface> m_surface;
  bool m_energyLoss;
  bool m_multipleScattering;
  std::size_t m_maxSteps;
};

PYBIND11_MODULE(acts_material_propagator, m) {
  m.doc() = "ACTS propagation to a perigee surface with material effects (MaterialInteractor), for acts_worker.py";
  py::module_::import("acts");  // registers TrackingGeometry and MagneticFieldProvider with pybind11
  py::class_<MaterialPropagator>(m, "MaterialPropagator")
      .def(py::init<std::shared_ptr<Acts::TrackingGeometry>, std::shared_ptr<Acts::MagneticFieldProvider>,
                    std::array<double, 3>, bool, bool, std::size_t>(),
           py::arg("geometry"), py::arg("field"), py::arg("reference"), py::arg("energy_loss") = true,
           py::arg("multiple_scattering") = true, py::arg("max_steps") = 0)
      .def("propagate", &MaterialPropagator::propagate, py::arg("position"), py::arg("direction"), py::arg("qop"),
           py::arg("backward"));
}
