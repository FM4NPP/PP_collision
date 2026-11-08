// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Vendored from ACTS v48.2.0 Examples/Detectors/TGeoDetector (TGeoDetector.hpp), modified: the examples framework
// (Detector base, Options), the ITk module splitter and the beam-pipe/element-factory options removed; one function
// builds the tracking geometry from a TGeo file, the TGeoDetector JSON configuration and a JSON material map.

#pragma once

#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/Surfaces/SurfacePlacementBase.hpp>
#include <Acts/Utilities/Logger.hpp>

#include <memory>
#include <string>
#include <vector>

namespace tgeo_geometry {

/// A tracking geometry and the detector elements its surfaces point to (kept alive with it).
struct TGeoGeometry {
  std::shared_ptr<const Acts::TrackingGeometry> geometry;
  std::vector<std::shared_ptr<const Acts::SurfacePlacementBase>> elements;
};

/// The tracking geometry of a TGeo file (geometry), laid out by a TGeoDetector JSON configuration (config), with the
/// JSON material map (materialMap; empty: none); builders log at level.
TGeoGeometry build(const std::string& geometry, const std::string& config, const std::string& materialMap,
                   Acts::Logging::Level level);

}  // namespace tgeo_geometry
