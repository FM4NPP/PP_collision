// acts_perigee_propagator: ACTS propagation of track states backward to a perigee surface, with covariance, for
// acts_worker.py; the one call ACTS's Python bindings lack. Takes the field and geometry objects the bindings
// build. Without a geometry: field only (VoidNavigator); with one: navigated, with material (MaterialInteractor).
// With the Kalman refit: each track's hits are first fitted by ACTS's Kalman fitter (each hit an (r phi, z)
// measurement on its own cylinder, DirectNavigator, field only: no TPC gas material) outward from the track's
// state just inside its innermost hit; the smoothed state at the innermost hit is then propagated to the perigee.
// States are split across threads (the propagator is const and keeps its state per call). Tracks are singly charged
// particles of one hypothesis (PDG id, mass), which sets the energy loss and multiple scattering in material.
// Units mm, GeV, e.
#include <Acts/Definitions/Algebra.hpp>
#include <Acts/Definitions/Direction.hpp>
#include <Acts/Definitions/ParticleData.hpp>
#include <Acts/Definitions/PdgParticle.hpp>
#include <Acts/Definitions/TrackParametrization.hpp>
#include <Acts/EventData/BoundTrackParameters.hpp>
#include <Acts/EventData/ParticleHypothesis.hpp>
#include <Acts/EventData/SourceLink.hpp>
#include <Acts/EventData/TrackContainer.hpp>
#include <Acts/EventData/VectorMultiTrajectory.hpp>
#include <Acts/EventData/VectorTrackContainer.hpp>
#include <Acts/Geometry/GeometryContext.hpp>
#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/MagneticField/MagneticFieldContext.hpp>
#include <Acts/MagneticField/MagneticFieldProvider.hpp>
#include <Acts/Propagator/ActorList.hpp>
#include <Acts/Propagator/DirectNavigator.hpp>
#include <Acts/Propagator/EigenStepper.hpp>
#include <Acts/Propagator/MaterialInteractor.hpp>
#include <Acts/Propagator/Navigator.hpp>
#include <Acts/Propagator/Propagator.hpp>
#include <Acts/Propagator/StandardAborters.hpp>
#include <Acts/Propagator/VoidNavigator.hpp>
#include <Acts/Surfaces/CylinderSurface.hpp>
#include <Acts/Surfaces/PerigeeSurface.hpp>
#include <Acts/TrackFitting/GainMatrixSmoother.hpp>
#include <Acts/TrackFitting/GainMatrixUpdater.hpp>
#include <Acts/TrackFitting/KalmanFitter.hpp>
#include <Acts/Utilities/CalibrationContext.hpp>
#include <Acts/Utilities/Logger.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using RowMatrix6 = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>;

// the Kalman refit: a hit's cylinder is unbounded in z (mm); the fit starts this far (mm) before the innermost hit
constexpr double kHitCylinderHalfZ = 1e6;
constexpr double kKalmanStartOffset = 10.0;

// one hit as a Kalman fit measurement: (r phi, z) = (0, z) on its own cylinder, rotated so the hit is at phi = 0
struct HitLink {
  const Acts::Surface* surface;
  Acts::Vector2 value;
  Acts::SquareMatrix<2> covariance;
};

const Acts::Surface* hitSurface(const Acts::SourceLink& link) { return link.get<HitLink>().surface; }

void calibrateHit(const Acts::GeometryContext& /*gctx*/, const Acts::CalibrationContext& /*cctx*/,
                  const Acts::SourceLink& link, Acts::VectorMultiTrajectory::TrackStateProxy state) {
  const HitLink& hit = link.get<HitLink>();
  state.setUncalibratedSourceLink(Acts::SourceLink{link});
  state.allocateCalibrated(2);
  state.calibrated<2>() = hit.value;
  state.calibratedCovariance<2>() = hit.covariance;
  state.setProjectorSubspaceIndices(std::array{Acts::eBoundLoc0, Acts::eBoundLoc1});
}

class PerigeePropagator {
 public:
  using FieldPropagator = Acts::Propagator<Acts::EigenStepper<>, Acts::VoidNavigator>;
  using GeometryPropagator = Acts::Propagator<Acts::EigenStepper<>, Acts::Navigator>;
  using GeometryOptions = GeometryPropagator::Options<Acts::ActorList<Acts::MaterialInteractor, Acts::EndOfWorldReached>>;
  using KalmanPropagator = Acts::Propagator<Acts::EigenStepper<>, Acts::DirectNavigator>;
  using KalmanFitter = Acts::KalmanFitter<KalmanPropagator, Acts::VectorMultiTrajectory>;

  PerigeePropagator(std::shared_ptr<Acts::MagneticFieldProvider> field, std::array<double, 3> reference,
                    std::shared_ptr<Acts::TrackingGeometry> geometry, bool energyLoss, std::size_t maxSteps,
                    std::size_t threads, int particlePdg, std::optional<float> particleMass, bool kalmanRefit,
                    std::array<double, 2> hitSigma)
      : m_particle(particleHypothesis(particlePdg, particleMass)),
        m_surface(Acts::Surface::makeShared<Acts::PerigeeSurface>(Acts::Vector3(reference[0], reference[1], reference[2]))),
        m_energyLoss(energyLoss),
        m_maxSteps(maxSteps),
        m_threads(threads) {
    if (threads == 0) {
      throw std::invalid_argument("threads must be at least 1");
    }
    // errors only: per-state warnings (e.g. a field lookup outside the map) are counted in the worker's failure summary
    auto logger = [](const char* name) { return Acts::getDefaultLogger(name, Acts::Logging::ERROR); };
    // the stepper and the navigator keep the field and the geometry alive
    if (geometry) {
      Acts::Navigator::Config config;
      config.trackingGeometry = geometry;
      config.resolveMaterial = true;
      config.resolveSensitive = true;
      config.resolvePassive = false;
      m_geometryPropagator.emplace(Acts::EigenStepper<>(field), Acts::Navigator(config, logger("Navigator")),
                                   logger("PerigeePropagator"));
    } else {
      m_fieldPropagator.emplace(Acts::EigenStepper<>(field), Acts::VoidNavigator(), logger("PerigeePropagator"));
    }
    if (kalmanRefit) {
      m_kalman.emplace(KalmanPropagator(Acts::EigenStepper<>(field), Acts::DirectNavigator(logger("DirectNavigator")),
                                        logger("KalmanPropagator")),
                       logger("KalmanFitter"));
      m_hitCovariance = Acts::Vector2(hitSigma[0] * hitSigma[0], hitSigma[1] * hitSigma[1]).asDiagonal();
    }
  }

  // position, direction (n, 3), q/p (n,) and covariance (n, 6, 6) (curvilinear, in the track's direction of
  // travel; at the innermost hit) and the hits (m, 3) with their state's index hit_track (m,)
  // (sorted by it, each track's hits in order along it; none without the refit) -> perigee parameters (d0, z0, phi,
  // theta, q/p, t) (n, 6), their covariance (n, 6, 6) (both NaN where ACTS failed or an input is not finite),
  // ok (n,) and {error message: count}
  py::tuple propagate(Array position, Array direction, Array qop, Array covariance, Array hitTrack, Array hits) const {
    const py::ssize_t n = qop.size();
    const py::ssize_t m = hitTrack.size();
    auto shaped = [](const Array& a, const std::vector<py::ssize_t>& shape) {
      return a.ndim() == static_cast<py::ssize_t>(shape.size()) && std::equal(shape.begin(), shape.end(), a.shape());
    };
    if (qop.ndim() != 1 || !shaped(position, {n, 3}) || !shaped(direction, {n, 3}) || !shaped(covariance, {n, 6, 6}) ||
        hitTrack.ndim() != 1 || !shaped(hits, {m, 3})) {
      throw std::invalid_argument(
          "position, direction (n, 3), qop (n,), covariance (n, 6, 6), hit_track (m,), hits (m, 3) expected");
    }
    // each state's hits: [hitBegin[i], hitBegin[i + 1])
    const double* T = hitTrack.data();
    std::vector<py::ssize_t> hitBegin(n + 1, 0);
    for (py::ssize_t j = 0; j < m; ++j) {
      if (!(T[j] >= 0 && T[j] < n && T[j] == std::floor(T[j])) || (j > 0 && T[j] < T[j - 1])) {
        throw std::invalid_argument("hit_track: state indices in [0, n), sorted");
      }
      ++hitBegin[static_cast<py::ssize_t>(T[j]) + 1];
    }
    std::partial_sum(hitBegin.begin(), hitBegin.end(), hitBegin.begin());
    const double* H = hits.data();
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
    const std::size_t nThreads = std::min<std::size_t>(m_threads, std::max<py::ssize_t>(n, 1));
    std::vector<std::map<std::string, std::size_t>> threadErrors(nThreads);
    std::vector<std::exception_ptr> threadFailures(nThreads);
    // states [first, last) on one thread; each state writes only its own output rows. An ACTS exception counts as
    // that state's failure; any other is rethrown after the threads finish
    auto work = [&](py::ssize_t first, py::ssize_t last, std::map<std::string, std::size_t>& errors,
                    std::exception_ptr& failure) {
      try {
        for (py::ssize_t i = first; i < last; ++i) {
          const Acts::Vector3 pos(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
          const Acts::Vector3 dir(D[3 * i], D[3 * i + 1], D[3 * i + 2]);
          const Acts::BoundMatrix startCov = Eigen::Map<const RowMatrix6>(C + 36 * i);
          // no track (a NaN state) is skipped; a track ACTS cannot start counts as a failure
          if (!pos.allFinite() || !dir.allFinite() || !std::isfinite(Q[i])) {
            continue;
          }
          if (!startCov.allFinite() || dir.norm() == 0.0) {
            ++errors["non-finite start covariance or zero direction"];
            continue;
          }
          try {
            const Acts::Vector3 at = m_kalman ? Acts::Vector3(pos - kKalmanStartOffset * dir.normalized()) : pos;
            const auto state = Acts::BoundTrackParameters::createCurvilinear(
                Acts::Vector4(at.x(), at.y(), at.z(), 0.0), dir.normalized(), Q[i], startCov, m_particle);
            std::string error;
            const std::optional<Acts::BoundTrackParameters> start =
                m_kalman ? refit(state, H + 3 * hitBegin[i], hitBegin[i + 1] - hitBegin[i], error) : state;
            if (!start) {
              ++errors[error];
              continue;
            }
            std::optional<Acts::BoundTrackParameters> end;
            if (m_geometryPropagator) {
              GeometryOptions options(m_geometryContext, m_fieldContext);
              options.actorList.get<Acts::MaterialInteractor>().energyLoss = m_energyLoss;
              end = run(*m_geometryPropagator, options, *start, error);
            } else {
              end = run(*m_fieldPropagator, FieldPropagator::Options<>(m_geometryContext, m_fieldContext), *start, error);
            }
            if (!end) {
              ++errors[error];
              continue;
            }
            // the start covariance is always transported
            if (!end->parameters().allFinite() || !end->covariance().value().allFinite()) {
              ++errors["non-finite perigee parameters or covariance"];
              continue;
            }
            Eigen::Map<RowMatrix6>(cov + 36 * i) = end->covariance().value();
            Eigen::Map<Acts::BoundVector>(out + 6 * i) = end->parameters();
            good[i] = true;
          } catch (const std::exception& e) {
            ++errors[e.what()];
          }
        }
      } catch (...) {
        failure = std::current_exception();
      }
    };
    {
      py::gil_scoped_release release;
      // jthreads join when the pool goes, also when starting one throws
      std::vector<std::jthread> pool;
      for (std::size_t t = 0; t < nThreads; ++t) {
        pool.emplace_back(work, n * t / nThreads, n * (t + 1) / nThreads, std::ref(threadErrors[t]),
                          std::ref(threadFailures[t]));
      }
    }
    for (const auto& failure : threadFailures) {
      if (failure) {
        std::rethrow_exception(failure);
      }
    }
    std::map<std::string, std::size_t> errors;
    for (const auto& counts : threadErrors) {
      for (const auto& [message, count] : counts) {
        errors[message] += count;
      }
    }
    return py::make_tuple(params, covOut, ok, errors);
  }

 private:
  // the Kalman fit of a track's count hits (in order along it), travelling forward from start (before the
  // innermost hit): the smoothed state at the innermost hit
  std::optional<Acts::BoundTrackParameters> refit(const Acts::BoundTrackParameters& start, const double* hits,
                                                  py::ssize_t count, std::string& error) const {
    if (count == 0) {
      error = "Kalman refit: no hits";
      return std::nullopt;
    }
    std::vector<std::shared_ptr<Acts::CylinderSurface>> surfaces;
    std::vector<const Acts::Surface*> sequence;
    std::vector<Acts::SourceLink> links;
    for (py::ssize_t j = 0; j < count; ++j) {
      const Acts::Vector3 hit(hits[3 * j], hits[3 * j + 1], hits[3 * j + 2]);
      Acts::Transform3 transform = Acts::Transform3::Identity();
      transform.rotate(Acts::AngleAxis3(std::atan2(hit.y(), hit.x()), Acts::Vector3::UnitZ()));
      surfaces.push_back(Acts::Surface::makeShared<Acts::CylinderSurface>(transform, hit.head<2>().norm(),
                                                                          kHitCylinderHalfZ));
      sequence.push_back(surfaces.back().get());
      links.emplace_back(HitLink{surfaces.back().get(), Acts::Vector2(0.0, hit.z()), m_hitCovariance});
    }
    Acts::KalmanFitterExtensions<Acts::VectorMultiTrajectory> extensions;
    extensions.surfaceAccessor.connect<&hitSurface>();
    extensions.calibrator.connect<&calibrateHit>();
    extensions.updater.connect<&Acts::GainMatrixUpdater::operator()<Acts::VectorMultiTrajectory>>(&m_updater);
    extensions.smoother.connect<&Acts::GainMatrixSmoother::operator()<Acts::VectorMultiTrajectory>>(&m_smoother);
    Acts::PropagatorPlainOptions plain(m_geometryContext, m_fieldContext);
    if (m_maxSteps > 0) {
      plain.maxSteps = m_maxSteps;
    }
    const Acts::KalmanFitterOptions<Acts::VectorMultiTrajectory> options(
        m_geometryContext, m_fieldContext, m_calibrationContext, extensions, plain, nullptr, false, false);
    Acts::TrackContainer tracks{Acts::VectorTrackContainer{}, Acts::VectorMultiTrajectory{}};
    auto result = m_kalman->fit(links.begin(), links.end(), start, options, sequence, tracks);
    if (!result.ok()) {
      error = "Kalman refit: " + result.error().message();
      return std::nullopt;
    }
    // the innermost measurement: the last one from the tip back
    std::optional<Acts::BoundTrackParameters> innermost;
    for (const auto state : result.value().trackStatesReversed()) {
      if (state.hasCalibrated() && state.hasSmoothed()) {
        innermost.emplace(state.referenceSurface().getSharedPtr(), Acts::BoundVector(state.smoothed()),
                          Acts::BoundMatrix(state.smoothedCovariance()), m_particle);
      }
    }
    if (!innermost) {
      error = "Kalman refit: not smoothed";
    }
    return innermost;
  }

  // a singly charged particle (the hit fit's q/p): the PDG id's mass from ACTS's particle data, or mass (GeV)
  static Acts::ParticleHypothesis particleHypothesis(int pdg, std::optional<float> mass) {
    const Acts::PdgParticle absPdg = Acts::makeAbsolutePdgParticle(static_cast<Acts::PdgParticle>(pdg));
    const std::optional<float> pdgMass = Acts::findMass(absPdg);
    if (!mass && !pdgMass) {
      throw std::invalid_argument("particle_mass is needed: ACTS has no mass for PDG id " + std::to_string(pdg));
    }
    return Acts::ParticleHypothesis(absPdg, mass ? *mass : *pdgMass, 1.0f);
  }

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

  Acts::GeometryContext m_geometryContext = Acts::GeometryContext::dangerouslyDefaultConstruct();
  Acts::MagneticFieldContext m_fieldContext{};
  std::optional<FieldPropagator> m_fieldPropagator;
  std::optional<GeometryPropagator> m_geometryPropagator;
  std::optional<KalmanFitter> m_kalman;
  Acts::CalibrationContext m_calibrationContext{};
  Acts::GainMatrixUpdater m_updater;
  Acts::GainMatrixSmoother m_smoother;
  Acts::SquareMatrix<2> m_hitCovariance = Acts::SquareMatrix<2>::Zero();
  Acts::ParticleHypothesis m_particle;
  std::shared_ptr<Acts::PerigeeSurface> m_surface;
  bool m_energyLoss;
  std::size_t m_maxSteps;
  std::size_t m_threads;
};

PYBIND11_MODULE(acts_perigee_propagator, m) {
  m.doc() = "ACTS propagation backward to a perigee surface, through a field or a tracking geometry with material";
  py::module_::import("acts");  // registers TrackingGeometry and MagneticFieldProvider with pybind11
  py::class_<PerigeePropagator>(m, "PerigeePropagator")
      .def(py::init<std::shared_ptr<Acts::MagneticFieldProvider>, std::array<double, 3>,
                    std::shared_ptr<Acts::TrackingGeometry>, bool, std::size_t, std::size_t, int, std::optional<float>,
                    bool, std::array<double, 2>>(),
           py::arg("field"), py::arg("reference"), py::arg("geometry"), py::arg("energy_loss"), py::arg("max_steps"),
           py::arg("threads"), py::arg("particle_pdg"), py::arg("particle_mass"), py::arg("kalman_refit"),
           py::arg("hit_sigma"))
      .def("propagate", &PerigeePropagator::propagate, py::arg("position"), py::arg("direction"), py::arg("qop"),
           py::arg("covariance"), py::arg("hit_track"), py::arg("hits"));
}
