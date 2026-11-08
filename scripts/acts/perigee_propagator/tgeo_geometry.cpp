// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Vendored from ACTS v48.2.0 Examples/Detectors/TGeoDetector (src/TGeoDetector.cpp and
// include/ActsExamples/TGeoDetector/JsonTGeoDetectorConfig.hpp), modified: the examples framework (Detector base,
// Options::Interval, boost program options) and the ITk module splitter removed (a configuration asking for it is
// an error), the JSON writers dropped, the JSON material decorator built here. Only ACTS Core and its Root and Json
// plugins are used; the geometry is built as TGeoDetector builds it.

#include "tgeo_geometry.hpp"

#include <Acts/Geometry/CylinderVolumeBuilder.hpp>
#include <Acts/Geometry/CylinderVolumeHelper.hpp>
#include <Acts/Geometry/GeometryContext.hpp>
#include <Acts/Geometry/GeometryIdentifier.hpp>
#include <Acts/Geometry/ITrackingVolumeBuilder.hpp>
#include <Acts/Geometry/LayerArrayCreator.hpp>
#include <Acts/Geometry/LayerCreator.hpp>
#include <Acts/Geometry/PassiveLayerBuilder.hpp>
#include <Acts/Geometry/ProtoLayerHelper.hpp>
#include <Acts/Geometry/SurfaceArrayCreator.hpp>
#include <Acts/Geometry/SurfaceBinningMatcher.hpp>
#include <Acts/Geometry/TrackingGeometryBuilder.hpp>
#include <Acts/Geometry/TrackingVolumeArrayCreator.hpp>
#include <Acts/Material/IMaterialDecorator.hpp>
#include <Acts/Utilities/BinningType.hpp>
#include <ActsPlugins/Json/ActsJson.hpp>
#include <ActsPlugins/Json/JsonMaterialDecorator.hpp>
#include <ActsPlugins/Json/MaterialMapJsonConverter.hpp>
#include <ActsPlugins/Root/TGeoAxes.hpp>
#include <ActsPlugins/Root/TGeoCylinderDiscSplitter.hpp>
#include <ActsPlugins/Root/TGeoLayerBuilder.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "TGeoManager.h"

using namespace Acts;
using namespace ActsPlugins;

namespace ActsPlugins {

/// Read config for cylinder/disc module splitter
void from_json(const nlohmann::json& j, TGeoCylinderDiscSplitter::Config& cdc) {
  cdc.cylinderPhiSegments = j.at("geo-tgeo-cyl-nphi-segs");
  cdc.cylinderLongitudinalSegments = j.at("geo-tgeo-cyl-nz-segs");
  cdc.discPhiSegments = j.at("geo-tgeo-disc-nphi-segs");
  cdc.discRadialSegments = j.at("geo-tgeo-disc-nr-segs");
}

// enum specialization by nlohman library
NLOHMANN_JSON_SERIALIZE_ENUM(Acts::BinningType,
                             {
                                 {Acts::BinningType::equidistant, "equidistant"},
                                 {Acts::BinningType::arbitrary, "arbitrary"},
                             })

}  // namespace ActsPlugins

namespace tgeo_geometry {

namespace {

/// A range with optional ends (ActsExamples::Options::Interval).
struct Interval {
  std::optional<double> lower;
  std::optional<double> upper;
};

void from_json(const nlohmann::json& j, Interval& interval) {
  interval.lower = j.at("lower");
  interval.upper = j.at("upper");
}

struct Config {
  Logging::Level surfaceLogLevel = Logging::WARNING;
  Logging::Level layerLogLevel = Logging::WARNING;
  Logging::Level volumeLogLevel = Logging::WARNING;

  std::string fileName;
  bool buildBeamPipe = false;
  double beamPipeRadius{0};
  double beamPipeHalflengthZ{0};
  double beamPipeLayerThickness{0};
  double beamPipeEnvelopeR{1.0};
  double layerEnvelopeR{1.0};

  double unitScalor = 1.0;

  std::shared_ptr<const GeometryIdentifierHook> geometryIdentifierHook =
      std::make_shared<GeometryIdentifierHook>();

  enum SubVolume : std::size_t { Negative = 0, Central, Positive };

  template <typename T>
  struct LayerTriplet {
    LayerTriplet() = default;
    explicit LayerTriplet(const T& value) : negative{value}, central{value}, positive{value} {}

    T negative;
    T central;
    T positive;

    const T& at(SubVolume i) const {
      switch (i) {
        case Negative:
          return negative;
        case Central:
          return central;
        case Positive:
          return positive;
        default:
          throw std::invalid_argument{"Unknown index"};
      }
    }
  };

  struct Volume {
    std::string name;
    LayerTriplet<bool> layers{false};
    LayerTriplet<std::string> subVolumeName;
    LayerTriplet<std::vector<std::string>> sensitiveNames;
    LayerTriplet<std::string> sensitiveAxes;
    LayerTriplet<Interval> rRange;
    LayerTriplet<Interval> zRange;
    LayerTriplet<double> splitTolR{0};
    LayerTriplet<double> splitTolZ{0};
    LayerTriplet<std::vector<std::pair<int, BinningType>>> binning0;
    LayerTriplet<std::vector<std::pair<int, BinningType>>> binning1;

    Interval binToleranceR;
    Interval binTolerancePhi;
    Interval binToleranceZ;

    bool cylinderDiscSplit = false;
    unsigned int cylinderNZSegments = 0;
    unsigned int cylinderNPhiSegments = 0;
    unsigned int discNRSegments = 0;
    unsigned int discNPhiSegments = 0;
  };

  std::vector<Volume> volumes;
};

/// Read layer configuration triplets
template <typename T>
void from_json(const nlohmann::json& j, Config::LayerTriplet<T>& ltr) {
  ltr.negative = j.at("negative").get<T>();
  ltr.central = j.at("central").get<T>();
  ltr.positive = j.at("positive").get<T>();
}

/// Read volume struct
void from_json(const nlohmann::json& j, Config::Volume& vol) {
  // subdetector selection
  vol.name = j.at("geo-tgeo-volume-name");

  // configure surface autobinning
  vol.binToleranceR = j.at("geo-tgeo-sfbin-r-tolerance");
  vol.binToleranceZ = j.at("geo-tgeo-sfbin-z-tolerance");
  vol.binTolerancePhi = j.at("geo-tgeo-sfbin-phi-tolerance");

  // Fill layer triplets
  vol.layers = j.at("geo-tgeo-volume-layers");
  vol.subVolumeName = j.at("geo-tgeo-subvolume-names");
  vol.sensitiveNames = j.at("geo-tgeo-sensitive-names");
  vol.sensitiveAxes = j.at("geo-tgeo-sensitive-axes");
  vol.rRange = j.at("geo-tgeo-layer-r-ranges");
  vol.zRange = j.at("geo-tgeo-layer-z-ranges");
  vol.splitTolR = j.at("geo-tgeo-layer-r-split");
  vol.splitTolZ = j.at("geo-tgeo-layer-z-split");
  // Set binning manually
  vol.binning0 = j.at("geo-tgeo-binning0");
  vol.binning1 = j.at("geo-tgeo-binning1");

  vol.cylinderDiscSplit = j.at("geo-tgeo-cyl-disc-split");
  if (vol.cylinderDiscSplit) {
    TGeoCylinderDiscSplitter::Config cdConfig = j.at("Splitters").at("CylinderDisk");
    vol.cylinderNZSegments = cdConfig.cylinderLongitudinalSegments;
    vol.cylinderNPhiSegments = cdConfig.cylinderPhiSegments;
    vol.discNRSegments = cdConfig.discRadialSegments;
    vol.discNPhiSegments = cdConfig.discPhiSegments;
  }

  if (j.count("geo-tgeo-itk-module-split") != 0 && j.at("geo-tgeo-itk-module-split").get<bool>()) {
    throw std::invalid_argument("volume " + vol.name + ": ITk module splitting is not supported (ACTS examples only)");
  }
}

/// Read the TGeo layer builder configurations from the user configuration.
void readJson(const std::string& path, Config& config) {
  nlohmann::json djson;
  std::ifstream infile(path, std::ifstream::in | std::ifstream::binary);
  if (!infile) {
    throw std::invalid_argument("cannot read the geometry configuration " + path);
  }
  infile >> djson;

  config.unitScalor = djson["geo-tgeo-unit-scalor"];

  config.buildBeamPipe = djson["geo-tgeo-build-beampipe"];
  if (config.buildBeamPipe) {
    const auto beamPipeParameters = djson["geo-tgeo-beampipe-parameters"].get<std::array<double, 3>>();
    config.beamPipeRadius = beamPipeParameters[0];
    config.beamPipeHalflengthZ = beamPipeParameters[1];
    config.beamPipeLayerThickness = beamPipeParameters[2];
  }

  // Fill nested volume configs
  for (const auto& volume : djson["Volumes"]) {
    auto& vol = config.volumes.emplace_back();
    vol = volume;
  }
}

/// @brief Function that constructs a set of layer builder
///        configs from a central TGeo config.
std::vector<TGeoLayerBuilder::Config> makeLayerBuilderConfigs(const Config& config, const Logger& logger) {
  std::vector<TGeoLayerBuilder::Config> detLayerConfigs;

  // iterate over all configured detector volumes
  for (const auto& volume : config.volumes) {
    TGeoLayerBuilder::Config layerBuilderConfig;
    layerBuilderConfig.configurationName = volume.name;
    layerBuilderConfig.unit = config.unitScalor;
    layerBuilderConfig.detectorElementFactory = TGeoLayerBuilder::defaultElementFactory;

    // configure surface autobinning
    std::vector<std::pair<double, double>> binTolerances(numAxisDirections(), {0., 0.});
    binTolerances[toUnderlying(AxisDirection::AxisR)] = {volume.binToleranceR.lower.value_or(0.),
                                                         volume.binToleranceR.upper.value_or(0.)};
    binTolerances[toUnderlying(AxisDirection::AxisZ)] = {volume.binToleranceZ.lower.value_or(0.),
                                                         volume.binToleranceZ.upper.value_or(0.)};
    binTolerances[toUnderlying(AxisDirection::AxisPhi)] = {volume.binTolerancePhi.lower.value_or(0.),
                                                           volume.binTolerancePhi.upper.value_or(0.)};

    layerBuilderConfig.autoSurfaceBinning = true;
    layerBuilderConfig.surfaceBinMatcher = SurfaceBinningMatcher(binTolerances);

    // loop over the negative/central/positive layer configurations
    for (auto ncp : {Config::Negative, Config::Central, Config::Positive}) {
      if (!volume.layers.at(ncp)) {
        continue;
      }

      TGeoLayerBuilder::LayerConfig lConfig;
      lConfig.volumeName = volume.subVolumeName.at(ncp);
      lConfig.sensorNames = volume.sensitiveNames.at(ncp);
      lConfig.localAxes = TGeoAxes::parse(volume.sensitiveAxes.at(ncp));
      lConfig.envelope = {config.layerEnvelopeR, config.layerEnvelopeR};

      auto rR = volume.rRange.at(ncp);
      auto rMin = rR.lower.value_or(0.);
      auto rMax = rR.upper.value_or(std::numeric_limits<double>::max());
      auto zR = volume.zRange.at(ncp);
      auto zMin = zR.lower.value_or(-std::numeric_limits<double>::max());
      auto zMax = zR.upper.value_or(std::numeric_limits<double>::max());
      lConfig.parseRanges = {
          {AxisDirection::AxisR, {rMin, rMax}},
          {AxisDirection::AxisZ, {zMin, zMax}},
      };

      // Fill the layer splitting parameters in r/z
      auto str = volume.splitTolR.at(ncp);
      auto stz = volume.splitTolZ.at(ncp);
      if (0 < str) {
        lConfig.splitConfigs.emplace_back(AxisDirection::AxisR, str);
      }
      if (0 < stz) {
        lConfig.splitConfigs.emplace_back(AxisDirection::AxisZ, stz);
      }
      lConfig.binning0 = volume.binning0.at(ncp);
      lConfig.binning1 = volume.binning1.at(ncp);

      layerBuilderConfig.layerConfigurations[ncp].push_back(lConfig);
    }

    // Perform splitting of cylinders and discs
    if (volume.cylinderDiscSplit) {
      TGeoCylinderDiscSplitter::Config cdsConfig;
      cdsConfig.cylinderPhiSegments = volume.cylinderNPhiSegments;
      cdsConfig.cylinderLongitudinalSegments = volume.cylinderNZSegments;
      cdsConfig.discPhiSegments = volume.discNPhiSegments;
      cdsConfig.discRadialSegments = volume.discNRSegments;
      layerBuilderConfig.detectorElementSplitter = std::make_shared<const TGeoCylinderDiscSplitter>(
          cdsConfig, logger.clone("TGeoCylinderDiscSplitter", config.layerLogLevel));
    }

    detLayerConfigs.push_back(layerBuilderConfig);
  }

  return detLayerConfigs;
}

/// @brief Function to build the tracking geometry from a TGeo object.
std::shared_ptr<const TrackingGeometry> buildTGeoDetector(
    const Config& config, const GeometryContext& context,
    std::vector<std::shared_ptr<const SurfacePlacementBase>>& detElementStore,
    std::shared_ptr<const IMaterialDecorator> materialDecorator, const Logger& logger) {
  // configure surface array creator
  SurfaceArrayCreator::Config sacConfig;
  auto surfaceArrayCreator = std::make_shared<const SurfaceArrayCreator>(
      sacConfig, logger.clone("SurfaceArrayCreator", config.surfaceLogLevel));
  // configure the proto layer helper
  ProtoLayerHelper::Config plhConfig;
  auto protoLayerHelper =
      std::make_shared<const ProtoLayerHelper>(plhConfig, logger.clone("ProtoLayerHelper", config.layerLogLevel));
  // configure the layer creator that uses the surface array creator
  LayerCreator::Config lcConfig;
  lcConfig.surfaceArrayCreator = surfaceArrayCreator;
  auto layerCreator =
      std::make_shared<const LayerCreator>(lcConfig, logger.clone("LayerCreator", config.layerLogLevel));
  // configure the layer array creator
  LayerArrayCreator::Config lacConfig;
  auto layerArrayCreator =
      std::make_shared<const LayerArrayCreator>(lacConfig, logger.clone("LayerArrayCreator", config.layerLogLevel));
  // tracking volume array creator
  TrackingVolumeArrayCreator::Config tvacConfig;
  auto tVolumeArrayCreator = std::make_shared<const TrackingVolumeArrayCreator>(
      tvacConfig, logger.clone("TrackingVolumeArrayCreator", config.volumeLogLevel));
  // configure the cylinder volume helper
  CylinderVolumeHelper::Config cvhConfig;
  cvhConfig.layerArrayCreator = layerArrayCreator;
  cvhConfig.trackingVolumeArrayCreator = tVolumeArrayCreator;
  auto cylinderVolumeHelper = std::make_shared<const CylinderVolumeHelper>(
      cvhConfig, logger.clone("CylinderVolumeHelper", config.volumeLogLevel));

  //-------------------------------------------------------------------------------------
  // list the volume builders
  std::list<std::shared_ptr<const ITrackingVolumeBuilder>> volumeBuilders;

  // Create a beam pipe if configured to do so
  if (config.buildBeamPipe) {
    /// configure the beam pipe layer builder
    PassiveLayerBuilder::Config bplConfig;
    bplConfig.layerIdentification = "BeamPipe";
    bplConfig.centralLayerRadii = {config.beamPipeRadius};
    bplConfig.centralLayerHalflengthZ = {config.beamPipeHalflengthZ};
    bplConfig.centralLayerThickness = {config.beamPipeLayerThickness};
    auto beamPipeBuilder = std::make_shared<const PassiveLayerBuilder>(
        bplConfig, logger.clone("BeamPipeLayerBuilder", config.layerLogLevel));
    // create the volume for the beam pipe
    CylinderVolumeBuilder::Config bpvConfig;
    bpvConfig.trackingVolumeHelper = cylinderVolumeHelper;
    bpvConfig.volumeName = "BeamPipe";
    bpvConfig.layerBuilder = beamPipeBuilder;
    bpvConfig.layerEnvelopeR = {config.beamPipeEnvelopeR, config.beamPipeEnvelopeR};
    bpvConfig.buildToRadiusZero = true;
    auto beamPipeVolumeBuilder = std::make_shared<const CylinderVolumeBuilder>(
        bpvConfig, logger.clone("BeamPipeVolumeBuilder", config.volumeLogLevel));
    // add to the list of builders
    volumeBuilders.push_back(beamPipeVolumeBuilder);
  }

  // Import the file from
  TGeoManager::Import(config.fileName.c_str());

  auto layerBuilderConfigs = makeLayerBuilderConfigs(config, logger);

  // Remember the layer builders to collect the detector elements
  std::vector<std::shared_ptr<const TGeoLayerBuilder>> tgLayerBuilders;

  for (auto& lbc : layerBuilderConfigs) {
    std::shared_ptr<const LayerCreator> layerCreatorLB = nullptr;

    if (lbc.autoSurfaceBinning) {
      // Configure surface array creator (optionally) per layer builder
      // (in order to configure them to work appropriately)
      SurfaceArrayCreator::Config sacConfigLB;
      sacConfigLB.surfaceMatcher = lbc.surfaceBinMatcher;
      auto surfaceArrayCreatorLB = std::make_shared<const SurfaceArrayCreator>(
          sacConfigLB, logger.clone(lbc.configurationName + "SurfaceArrayCreator", config.surfaceLogLevel));
      // configure the layer creator that uses the surface array creator
      LayerCreator::Config lcConfigLB;
      lcConfigLB.surfaceArrayCreator = surfaceArrayCreatorLB;
      layerCreatorLB = std::make_shared<const LayerCreator>(
          lcConfigLB, logger.clone(lbc.configurationName + "LayerCreator", config.layerLogLevel));
    }

    // Configure the proto layer helper
    ProtoLayerHelper::Config plhConfigLB;
    auto protoLayerHelperLB = std::make_shared<const ProtoLayerHelper>(
        plhConfigLB, logger.clone(lbc.configurationName + "ProtoLayerHelper", config.layerLogLevel));

    //-------------------------------------------------------------------------------------
    lbc.layerCreator = (layerCreatorLB != nullptr) ? layerCreatorLB : layerCreator;
    lbc.protoLayerHelper = (protoLayerHelperLB != nullptr) ? protoLayerHelperLB : protoLayerHelper;

    auto layerBuilder = std::make_shared<const TGeoLayerBuilder>(
        lbc, logger.clone(lbc.configurationName + "LayerBuilder", config.layerLogLevel));
    // remember the layer builder
    tgLayerBuilders.push_back(layerBuilder);

    // build the pixel volume
    CylinderVolumeBuilder::Config volumeConfig;
    volumeConfig.trackingVolumeHelper = cylinderVolumeHelper;
    volumeConfig.volumeName = lbc.configurationName;
    volumeConfig.buildToRadiusZero = volumeBuilders.empty();
    volumeConfig.layerEnvelopeR = {config.layerEnvelopeR, config.layerEnvelopeR};
    auto ringLayoutConfiguration = [&](const std::vector<TGeoLayerBuilder::LayerConfig>& lConfigs) {
      for (const auto& lcfg : lConfigs) {
        for (const auto& scfg : lcfg.splitConfigs) {
          if (scfg.first == AxisDirection::AxisR && scfg.second > 0.) {
            volumeConfig.ringTolerance = std::max(volumeConfig.ringTolerance, scfg.second);
            volumeConfig.checkRingLayout = true;
          }
        }
      }
    };
    ringLayoutConfiguration(lbc.layerConfigurations[0]);
    ringLayoutConfiguration(lbc.layerConfigurations[2]);
    volumeConfig.layerBuilder = layerBuilder;
    auto volumeBuilder = std::make_shared<const CylinderVolumeBuilder>(
        volumeConfig, logger.clone(lbc.configurationName + "VolumeBuilder", config.volumeLogLevel));
    // add to the list of builders
    volumeBuilders.push_back(volumeBuilder);
  }

  //-------------------------------------------------------------------------------------
  // create the tracking geometry
  TrackingGeometryBuilder::Config tgConfig;
  // Add the builders
  tgConfig.materialDecorator = std::move(materialDecorator);
  tgConfig.geometryIdentifierHook = config.geometryIdentifierHook;

  for (auto& vb : volumeBuilders) {
    tgConfig.trackingVolumeBuilders.push_back(
        [=](const auto& gcontext, const auto& inner, const auto&) { return vb->trackingVolume(gcontext, inner); });
  }
  // Add the helper
  tgConfig.trackingVolumeHelper = cylinderVolumeHelper;
  auto cylinderGeometryBuilder = std::make_shared<const TrackingGeometryBuilder>(
      tgConfig, logger.clone("TrackerGeometryBuilder", config.volumeLogLevel));
  // get the geometry
  auto trackingGeometry = cylinderGeometryBuilder->trackingGeometry(context);
  // collect the detector element store
  for (auto& lBuilder : tgLayerBuilders) {
    auto detElements = lBuilder->detectorElements();
    detElementStore.insert(detElementStore.begin(), detElements.begin(), detElements.end());
  }

  /// return the tracking geometry
  return trackingGeometry;
}

}  // namespace

TGeoGeometry build(const std::string& geometry, const std::string& config, const std::string& materialMap,
                   Logging::Level level) {
  Config cfg;
  readJson(config, cfg);
  cfg.fileName = geometry;
  cfg.surfaceLogLevel = cfg.layerLogLevel = cfg.volumeLogLevel = level;
  std::shared_ptr<const IMaterialDecorator> decorator;
  if (!materialMap.empty()) {
    decorator = std::make_shared<const JsonMaterialDecorator>(MaterialMapJsonConverter::Config{}, materialMap, level);
  }
  TGeoGeometry built;
  auto logger = getDefaultLogger("TGeoGeometry", level);
  built.geometry = buildTGeoDetector(cfg, GeometryContext::dangerouslyDefaultConstruct(), built.elements, decorator,
                                     *logger);
  return built;
}

}  // namespace tgeo_geometry
