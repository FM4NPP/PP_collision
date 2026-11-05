// acts_perigee_propagator: ACTS (unmodified, linked as an external library) propagating track states to a perigee
// surface, for acts_worker.py. ACTS's core Python bindings only offer propagateToSurface without actors and
// without a direction or covariance; this module adds that one call, nothing else. It takes the field (and
// geometry) objects ACTS's own Python bindings build (same pybind11, built against the same ACTS libraries), so
// field map, geometry and material map stay configured in Python.
//
// Without a geometry the track is stepped through the field alone (VoidNavigator). With one it is navigated
// through the tracking geometry with ACTS's MaterialInteractor: energy loss changes q/p, multiple scattering
// widens a given covariance (it changes nothing without one).
//
// Units are ACTS's (mm, GeV, e). Tracks are propagated in their real direction: a state whose perigee lies
// behind it (e.g. at the innermost hit) is propagated with Direction::Backward(), so energy loss is added back
// (MaterialInteractor: "in forward(backward) propagation, energy decreases(increases)").
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
#include <Acts/Propagator/VoidNavigator.hpp>
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
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;

class PerigeePropagator {
 public:
  using FieldPropagator = Acts::Propagator<Acts::EigenStepper<>, Acts::VoidNavigator>;
  using FieldOptions = FieldPropagator::Options<>;
  using GeometryPropagator = Acts::Propagator<Acts::EigenStepper<>, Acts::Navigator>;
  using GeometryOptions = GeometryPropagator::Options<Acts::ActorList<Acts::MaterialInteractor, Acts::EndOfWorldReached>>;

  PerigeePropagator(std::shared_ptr<Acts::MagneticFieldProvider> field, std::array<double, 3> reference,
                    std::shared_ptr<Acts::TrackingGeometry> geometry, bool energyLoss, bool multipleScattering,
                    std::size_t maxSteps)
      : m_field(std::move(field)),
        m_geometry(std::move(geometry)),
        m_surface(Acts::Surface::makeShared<Acts::PerigeeSurface>(
            Acts::Vector3(reference[0], reference[1], reference[2]))),
        m_energyLoss(energyLoss),
        m_multipleScattering(multipleScattering),
        m_maxSteps(maxSteps) {
    if (m_geometry) {
      Acts::Navigator::Config config;
      config.trackingGeometry = m_geometry;
      config.resolveMaterial = true;
      config.resolveSensitive = true;
      config.resolvePassive = false;
      m_geometryPropagator.emplace(Acts::EigenStepper<>(m_field),
                                   Acts::Navigator(config, Acts::getDefaultLogger("Navigator", Acts::Logging::WARNING)),
                                   Acts::getDefaultLogger("PerigeePropagator", Acts::Logging::WARNING));
    } else {
      m_fieldPropagator.emplace(Acts::EigenStepper<>(m_field), Acts::VoidNavigator(),
                                Acts::getDefaultLogger("PerigeePropagator", Acts::Logging::WARNING));
    }
  }

  // (n, 6) perigee parameters (d0, z0, phi, theta, q/p, t; NaN where ACTS failed or the input is not
  // finite), ok (n,), {error message: count} and, given a start covariance (n, 6, 6) (curvilinear, in the
  // track's real direction), the perigee covariance (n, 6, 6) (NaN where none), else None
  py::tuple propagate(py::array_t<double, py::array::c_style | py::array::forcecast> position,
                      py::array_t<double, py::array::c_style | py::array::forcecast> direction,
                      py::array_t<double, py::array::c_style | py::array::forcecast> qop,
                      py::array_t<bool, py::array::c_style | py::array::forcecast> backward,
                      std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>> covariance) const {
    const py::ssize_t n = qop.size();
    if (position.ndim() != 2 || position.shape(0) != n || position.shape(1) != 3 || direction.ndim() != 2 ||
        direction.shape(0) != n || direction.shape(1) != 3 || backward.size() != n) {
      throw std::invalid_argument("position, direction (n, 3), qop, backward (n,) expected");
    }
    if (covariance && (covariance->ndim() != 3 || covariance->shape(0) != n || covariance->shape(1) != 6 ||
                       covariance->shape(2) != 6)) {
      throw std::invalid_argument("covariance (n, 6, 6) expected");
    }
    py::array_t<double> params({n, py::ssize_t{6}});
    py::array_t<double> covOut({covariance ? n : py::ssize_t{0}, py::ssize_t{6}, py::ssize_t{6}});
    py::array_t<bool> ok(n);
    auto P = position.unchecked<2>();
    auto D = direction.unchecked<2>();
    auto Q = qop.unchecked<1>();
    auto B = backward.unchecked<1>();
    auto out = params.mutable_unchecked<2>();
    auto covWrite = covOut.mutable_unchecked<3>();
    std::optional<py::detail::unchecked_reference<double, 3>> C;
    if (covariance) {
      C.emplace(covariance->unchecked<3>());
    }
    auto good = ok.mutable_unchecked<1>();
    std::map<std::string, std::size_t> errors;
    {
      py::gil_scoped_release release;
      for (py::ssize_t i = 0; i < n; ++i) {
        for (int k = 0; k < 6; ++k) {
          out(i, k) = std::numeric_limits<double>::quiet_NaN();
          for (int l = 0; C && l < 6; ++l) {
            covWrite(i, k, l) = std::numeric_limits<double>::quiet_NaN();
          }
        }
        good(i) = false;
        const Acts::Vector3 pos(P(i, 0), P(i, 1), P(i, 2));
        const Acts::Vector3 dir(D(i, 0), D(i, 1), D(i, 2));
        std::optional<Acts::BoundMatrix> startCov;
        if (C) {
          Acts::BoundMatrix m;
          for (int k = 0; k < 6; ++k) {
            for (int l = 0; l < 6; ++l) {
              m(k, l) = (*C)(i, k, l);
            }
          }
          if (!m.allFinite()) {
            continue;
          }
          startCov = m;
        }
        if (!pos.allFinite() || !dir.allFinite() || !std::isfinite(Q(i)) || dir.norm() == 0.0) {
          continue;
        }
        try {
          const auto start = Acts::BoundTrackParameters::createCurvilinear(
              Acts::Vector4(pos.x(), pos.y(), pos.z(), 0.0), dir.normalized(), Q(i), startCov,
              Acts::ParticleHypothesis::pion());
          const auto dirn = B(i) ? Acts::Direction::Backward() : Acts::Direction::Forward();
          std::string error;
          std::optional<Acts::BoundTrackParameters> end;
          if (m_geometryPropagator) {
            GeometryOptions options(m_geometryContext, m_fieldContext);
            auto& interactor = options.actorList.get<Acts::MaterialInteractor>();
            interactor.energyLoss = m_energyLoss;
            interactor.multipleScattering = m_multipleScattering;
            end = run(*m_geometryPropagator, options, start, dirn, error);
          } else {
            end = run(*m_fieldPropagator, FieldOptions(m_geometryContext, m_fieldContext), start, dirn, error);
          }
          if (!end) {
            ++errors[error];
            continue;
          }
          const auto values = end->parameters();
          for (int k = 0; k < 6; ++k) {
            out(i, k) = values[k];
          }
          good(i) = values.allFinite();
          if (C && end->covariance().has_value()) {
            const auto& endCov = *end->covariance();
            for (int k = 0; k < 6; ++k) {
              for (int l = 0; l < 6; ++l) {
                covWrite(i, k, l) = endCov(k, l);
              }
            }
          }
        } catch (const std::exception& e) {
          ++errors[e.what()];
        }
      }
    }
    return py::make_tuple(params, ok, errors, covariance ? py::object(covOut) : py::object(py::none()));
  }

 private:
  template <typename propagator_t, typename options_t>
  std::optional<Acts::BoundTrackParameters> run(const propagator_t& propagator, options_t options,
                                                const Acts::BoundTrackParameters& start, Acts::Direction direction,
                                                std::string& error) const {
    options.direction = direction;
    if (m_maxSteps > 0) {
      options.maxSteps = m_maxSteps;
    }
    auto result = propagator.template propagate<options_t, Acts::ForcedSurfaceReached, Acts::PathLimitReached>(
        start, *m_surface, options);
    if (!result.ok()) {
      error = result.error().message();
      return std::nullopt;
    }
    if (!result.value().endParameters.has_value()) {
      error = "perigee surface not reached";
      return std::nullopt;
    }
    return *result.value().endParameters;
  }

  std::shared_ptr<Acts::MagneticFieldProvider> m_field;
  std::shared_ptr<Acts::TrackingGeometry> m_geometry;
  Acts::GeometryContext m_geometryContext = Acts::GeometryContext::dangerouslyDefaultConstruct();
  Acts::MagneticFieldContext m_fieldContext{};
  std::optional<FieldPropagator> m_fieldPropagator;
  std::optional<GeometryPropagator> m_geometryPropagator;
  std::shared_ptr<Acts::PerigeeSurface> m_surface;
  bool m_energyLoss;
  bool m_multipleScattering;
  std::size_t m_maxSteps;
};

PYBIND11_MODULE(acts_perigee_propagator, m) {
  m.doc() = "ACTS propagation to a perigee surface, through a field or a tracking geometry with material";
  py::module_::import("acts");  // registers TrackingGeometry and MagneticFieldProvider with pybind11
  py::class_<PerigeePropagator>(m, "PerigeePropagator")
      .def(py::init<std::shared_ptr<Acts::MagneticFieldProvider>, std::array<double, 3>,
                    std::shared_ptr<Acts::TrackingGeometry>, bool, bool, std::size_t>(),
           py::arg("field"), py::arg("reference"), py::arg("geometry") = py::none(), py::arg("energy_loss") = true,
           py::arg("multiple_scattering") = true, py::arg("max_steps") = 0)
      .def("propagate", &PerigeePropagator::propagate, py::arg("position"), py::arg("direction"), py::arg("qop"),
           py::arg("backward"), py::arg("covariance") = py::none());
}
