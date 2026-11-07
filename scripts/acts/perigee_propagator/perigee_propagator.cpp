// acts_perigee_propagator: ACTS propagation of track states backward to a perigee surface, with covariance, for
// acts_worker.py; the one call ACTS's Python bindings lack. Takes the field and geometry objects the bindings
// build. Without a geometry: field only (VoidNavigator); with one: navigated, with material (MaterialInteractor).
// Units mm, GeV, e.
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
#include <Acts/Utilities/Logger.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using RowMatrix6 = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;

class PerigeePropagator {
 public:
  using FieldPropagator = Acts::Propagator<Acts::EigenStepper<>, Acts::VoidNavigator>;
  using GeometryPropagator = Acts::Propagator<Acts::EigenStepper<>, Acts::Navigator>;
  using GeometryOptions = GeometryPropagator::Options<Acts::ActorList<Acts::MaterialInteractor, Acts::EndOfWorldReached>>;

  PerigeePropagator(std::shared_ptr<Acts::MagneticFieldProvider> field, std::array<double, 3> reference,
                    std::shared_ptr<Acts::TrackingGeometry> geometry, bool energyLoss, std::size_t maxSteps)
      : m_field(std::move(field)),
        m_geometry(std::move(geometry)),
        m_surface(Acts::Surface::makeShared<Acts::PerigeeSurface>(Acts::Vector3(reference[0], reference[1], reference[2]))),
        m_energyLoss(energyLoss),
        m_maxSteps(maxSteps) {
    // errors only: per-state warnings (e.g. a field lookup outside the map) are counted in the worker's failure summary
    auto logger = [](const char* name) { return Acts::getDefaultLogger(name, Acts::Logging::ERROR); };
    if (m_geometry) {
      Acts::Navigator::Config config;
      config.trackingGeometry = m_geometry;
      config.resolveMaterial = true;
      config.resolveSensitive = true;
      config.resolvePassive = false;
      m_geometryPropagator.emplace(Acts::EigenStepper<>(m_field), Acts::Navigator(config, logger("Navigator")),
                                   logger("PerigeePropagator"));
    } else {
      m_fieldPropagator.emplace(Acts::EigenStepper<>(m_field), Acts::VoidNavigator(), logger("PerigeePropagator"));
    }
  }

  // position, direction (n, 3), q/p (n,) and covariance (n, 6, 6) (curvilinear, in the track's direction of
  // travel) -> perigee parameters (d0, z0, phi, theta, q/p, t) (n, 6), their covariance (n, 6, 6) (both NaN where
  // ACTS failed or an input is not finite), ok (n,) and {error message: count}
  py::tuple propagate(Array position, Array direction, Array qop, Array covariance) const {
    const py::ssize_t n = qop.size();
    if (position.size() != 3 * n || direction.size() != 3 * n || covariance.size() != 36 * n) {
      throw std::invalid_argument("position, direction (n, 3), qop (n,), covariance (n, 6, 6) expected");
    }
    Array params({n, py::ssize_t{6}});
    Array covOut({n, py::ssize_t{6}, py::ssize_t{6}});
    py::array_t<bool> ok(n);
    const double* P = position.data();
    const double* D = direction.data();
    const double* Q = qop.data();
    const double* C = covariance.data();
    double* out = params.mutable_data();
    double* cov = covOut.mutable_data();
    bool* good = ok.mutable_data();
    std::fill(out, out + 6 * n, std::numeric_limits<double>::quiet_NaN());
    std::fill(cov, cov + 36 * n, std::numeric_limits<double>::quiet_NaN());
    std::fill(good, good + n, false);
    std::map<std::string, std::size_t> errors;
    {
      py::gil_scoped_release release;
      for (py::ssize_t i = 0; i < n; ++i) {
        const Acts::Vector3 pos(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
        const Acts::Vector3 dir(D[3 * i], D[3 * i + 1], D[3 * i + 2]);
        const Acts::BoundMatrix startCov = Eigen::Map<const RowMatrix6>(C + 36 * i);
        if (!startCov.allFinite() || !pos.allFinite() || !dir.allFinite() || !std::isfinite(Q[i]) ||
            dir.norm() == 0.0) {
          continue;
        }
        try {
          const auto start = Acts::BoundTrackParameters::createCurvilinear(
              Acts::Vector4(pos.x(), pos.y(), pos.z(), 0.0), dir.normalized(), Q[i], startCov,
              Acts::ParticleHypothesis::pion());
          std::string error;
          std::optional<Acts::BoundTrackParameters> end;
          if (m_geometryPropagator) {
            GeometryOptions options(m_geometryContext, m_fieldContext);
            options.actorList.get<Acts::MaterialInteractor>().energyLoss = m_energyLoss;
            end = run(*m_geometryPropagator, options, start, error);
          } else {
            end = run(*m_fieldPropagator, FieldPropagator::Options<>(m_geometryContext, m_fieldContext), start, error);
          }
          if (!end) {
            ++errors[error];
            continue;
          }
          Eigen::Map<Acts::BoundVector>(out + 6 * i) = end->parameters();
          good[i] = end->parameters().allFinite();
          if (end->covariance().has_value()) {
            Eigen::Map<RowMatrix6>(cov + 36 * i) = *end->covariance();
          }
        } catch (const std::exception& e) {
          ++errors[e.what()];
        }
      }
    }
    return py::make_tuple(params, covOut, ok, errors);
  }

 private:
  template <typename propagator_t, typename options_t>
  std::optional<Acts::BoundTrackParameters> run(const propagator_t& propagator, options_t options,
                                                const Acts::BoundTrackParameters& start, std::string& error) const {
    options.direction = Acts::Direction::Backward();
    if (m_maxSteps > 0) {
      options.maxSteps = m_maxSteps;
    }
    auto result = propagator.template propagate<options_t, Acts::ForcedSurfaceReached>(start, *m_surface, options);
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
  std::size_t m_maxSteps;
};

PYBIND11_MODULE(acts_perigee_propagator, m) {
  m.doc() = "ACTS propagation backward to a perigee surface, through a field or a tracking geometry with material";
  py::module_::import("acts");  // registers TrackingGeometry and MagneticFieldProvider with pybind11
  py::class_<PerigeePropagator>(m, "PerigeePropagator")
      .def(py::init<std::shared_ptr<Acts::MagneticFieldProvider>, std::array<double, 3>,
                    std::shared_ptr<Acts::TrackingGeometry>, bool, std::size_t>(),
           py::arg("field"), py::arg("reference"), py::arg("geometry") = py::none(), py::arg("energy_loss") = true,
           py::arg("max_steps") = 0)
      .def("propagate", &PerigeePropagator::propagate, py::arg("position"), py::arg("direction"), py::arg("qop"),
           py::arg("covariance"));
}
