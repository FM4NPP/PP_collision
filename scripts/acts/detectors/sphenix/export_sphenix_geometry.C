// export_sphenix_geometry.C: write an sPHENIX tracking-geometry file (conditions database key
// Tracking_Geometry, a PHOOL file) in portable form for the ACTS container, which has no sPHENIX code:
//   <prefix>_tgeo.root        the detector geometry as a plain ROOT TGeoManager (from RUN#GEOMETRY_IO), with
//                             the TPC measurement boxes ACTS builds its TPC surfaces from (see below)
//   <prefix>_tpc_layers.txt   the TPC layout: layer, radius, thickness in cm (from TPCGEOMCONTAINER), and the
//                             maximum drift length
//   <prefix>_volumes.txt      the geometry's volume tree (names, 4 levels), for checking the ACTS config
// Run ONCE, on a machine with the sPHENIX software set up (it needs sPHENIX's class dictionaries):
//   root -l -b -q 'scripts/acts/detectors/sphenix/export_sphenix_geometry.C("<Tracking_Geometry file>.root", "sphenix_ana559")'
// sPHENIX's classes are used through the interpreter (by name), so no sPHENIX headers are needed here.
//
// TPC measurement boxes: sPHENIX's tracking adds them to the TPC gas volume before ACTS builds its geometry
// (its material map is keyed to the surfaces they give). Same layout, written here from its description:
// per TPC layer a box "tpc_gas_measurement_<layer>" of half-sizes 0.4999 x thickness (radial),
// 0.4999 x 2 tan(dphi / 2) (r - thickness / 2) (along phi) and half the surface z length; placed at the layer
// radius, z = 0, rotated to the centre of each of n_modules x n_phi phi segments starting at -pi, with
// copy numbers counting up over layers, modules and segments; surface z length = maximum drift length -
// 0.0001 cm. Added to the gas volume, so both its placements (tpc_gas_north / _south) get them.

#include <TClass.h>
#include <TDataType.h>
#include <TFile.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>
#include <TMath.h>
#include <TROOT.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
  void write_volumes(std::ofstream &out, TGeoNode *node, int depth, int max_depth)
  {
    out << std::string(2 * depth, ' ') << node->GetName() << "  [" << node->GetVolume()->GetName() << ", "
        << node->GetVolume()->GetShape()->ClassName() << ", " << node->GetNdaughters() << " daughters]\n";
    if (depth >= max_depth)
    {
      return;
    }
    for (int i = 0; i < node->GetNdaughters(); ++i)
    {
      write_volumes(out, node->GetDaughter(i), depth + 1, max_depth);
    }
  }

  TGeoVolume *daughter_volume(TGeoVolume *mother, const char *name_start)
  {
    for (int i = 0; i < mother->GetNdaughters(); ++i)
    {
      if (TString(mother->GetNode(i)->GetName()).BeginsWith(name_start))
      {
        return mother->GetNode(i)->GetVolume();
      }
    }
    return nullptr;
  }

  // returns the number of boxes placed; 0 if the geometry already has them
  int add_tpc_measurement_boxes(TGeoManager *manager, const std::vector<double> &radius,
                                const std::vector<double> &thickness, double surface_z_length, int n_modules,
                                int n_phi)
  {
    TGeoVolume *envelope = daughter_volume(manager->GetTopVolume(), "tpc_envelope");
    TGeoVolume *gas = envelope ? daughter_volume(envelope, "tpc_gas_north") : nullptr;
    if (!gas)
    {
      std::cout << "ERROR: no World/tpc_envelope*/tpc_gas_north* volume" << std::endl;
      return -1;
    }
    for (int i = 0; i < gas->GetNdaughters(); ++i)
    {
      if (TString(gas->GetNode(i)->GetName()).BeginsWith("tpc_gas_measurement_"))
      {
        std::cout << "TPC measurement boxes already present: not added again" << std::endl;
        return 0;
      }
    }
    const double module_dphi = 2.0 * M_PI / n_modules;
    const double surface_dphi = 2.0 * M_PI / (n_phi * n_modules);
    int copy = 0;
    for (size_t layer = 0; layer < radius.size(); ++layer)
    {
      const double rphi = 2.0 * std::tan(surface_dphi / 2.0) * (radius[layer] - thickness[layer] / 2.0);
      TGeoVolume *box = manager->MakeBox(("tpc_gas_measurement_" + std::to_string(layer)).c_str(), gas->GetMedium(),
                                         0.4999 * thickness[layer], 0.4999 * rphi, 0.5 * surface_z_length);
      for (int module = 0; module < n_modules; ++module)
      {
        for (int segment = 0; segment < n_phi; ++segment)
        {
          const double phi = -M_PI + module * module_dphi + (segment + 0.5) * surface_dphi;
          auto *rotation = new TGeoRotation(("tpc_gas_rotation_" + std::to_string(copy)).c_str(),
                                            phi * TMath::RadToDeg(), 0, 0);
          gas->AddNode(box, copy, new TGeoCombiTrans(radius[layer] * std::cos(phi), radius[layer] * std::sin(phi), 0.0,
                                                     rotation));
          ++copy;
        }
      }
    }
    return copy;
  }
}  // namespace

void export_sphenix_geometry(const char *input, const char *prefix = "sphenix_geometry", bool add_tpc_boxes = true,
                             int n_modules = 12, int n_phi = 12)
{
  // sPHENIX libraries holding the stored classes' dictionaries (names differ between releases: load what exists)
  for (const char *lib : {"libphool.so", "libphgeom.so", "libg4detectors.so", "libg4tpc.so", "libtrack_io.so",
                          "libtrackbase_historic_io.so", "libmvtx_io.so", "libintt_io.so", "libmicromegas_io.so"})
  {
    if (gSystem->Load(lib) < 0)
    {
      std::cout << "(library not loaded, may not exist in this release: " << lib << ")" << std::endl;
    }
  }
  for (const char *name : {"PHGeomIOTGeo", "PHG4TpcGeomContainer"})
  {
    TClass *cls = TClass::GetClass(name);
    if (!cls || !cls->HasDictionary())
    {
      std::cout << "ERROR: no dictionary for " << name << ": set up the sPHENIX software first (e.g. "
                << "source /opt/sphenix/core/bin/sphenix_setup.sh -n)" << std::endl;
      return;
    }
  }

  TFile file(input);
  TTree *run_tree = file.Get<TTree>("T1");
  if (!run_tree)
  {
    std::cout << "ERROR: no run tree T1 in " << input << std::endl;
    return;
  }
  void *geometry_io = nullptr;
  void *tpc_container = nullptr;
  // untyped addresses, so the class is given explicitly (an object pointer: isptr = true)
  run_tree->SetBranchAddress("RUN#GEOMETRY_IO", &geometry_io, nullptr, TClass::GetClass("PHGeomIOTGeo"), kOther_t, true);
  run_tree->SetBranchAddress("RUN#RECO_TRACKING_GEOMETRY#TPCGEOMCONTAINER", &tpc_container, nullptr,
                             TClass::GetClass("PHG4TpcGeomContainer"), kOther_t, true);
  run_tree->GetEntry(0);
  if (!geometry_io || !tpc_container)
  {
    std::cout << "ERROR: RUN#GEOMETRY_IO or TPCGEOMCONTAINER not read from " << input << std::endl;
    return;
  }

  // the geometry, rebuilt by sPHENIX's own PHGeomIOTGeo
  auto *manager = reinterpret_cast<TGeoManager *>(
      gROOT->ProcessLine(Form("((PHGeomIOTGeo *) %p)->ConstructTGeoManager();", geometry_io)));
  if (!manager)
  {
    // older interface: a copy of the top volume, closed into a new geometry
    auto *top = reinterpret_cast<TGeoVolume *>(
        gROOT->ProcessLine(Form("((PHGeomIOTGeo *) %p)->GetGeometryCopy();", geometry_io)));
    if (!top)
    {
      std::cout << "ERROR: PHGeomIOTGeo gave no geometry (neither ConstructTGeoManager nor GetGeometryCopy)" << std::endl;
      return;
    }
    manager = new TGeoManager("sPHENIX", "sPHENIX tracking geometry");
    manager->SetTopVolume(top);
    manager->CloseGeometry();
  }

  // the TPC layout, layer by layer, by sPHENIX's own PHG4TpcGeom accessors (layer numbers in order); the
  // maximum drift length (the same for all layers) from layer 20
  std::vector<int> layers;
  std::vector<double> radius, thickness;
  double max_driftlength = 0.0;
  gROOT->ProcessLine(Form(
      "{ auto *c = (PHG4TpcGeomContainer *) %p; auto *l = (std::vector<int> *) %p; auto *r = (std::vector<double> *) %p;"
      "  auto *t = (std::vector<double> *) %p; auto range = c->get_begin_end();"
      "  for (auto it = range.first; it != range.second; ++it) {"
      "    l->push_back(it->first); r->push_back(it->second->get_radius()); t->push_back(it->second->get_thickness());"
      "    if (it->first == 20) *(double *) %p = it->second->get_max_driftlength(); } }",
      tpc_container, (void *) &layers, (void *) &radius, (void *) &thickness, (void *) &max_driftlength));
  if (radius.empty())
  {
    std::cout << "ERROR: no TPC layers read from TPCGEOMCONTAINER" << std::endl;
    return;
  }
  const TString tpc_path = TString(prefix) + "_tpc_layers.txt";
  std::ofstream tpc(tpc_path.Data());
  tpc << "# max_driftlength_cm " << max_driftlength << "\n# layer radius_cm thickness_cm\n";
  for (size_t i = 0; i < layers.size(); ++i)
  {
    tpc << layers[i] << " " << radius[i] << " " << thickness[i] << "\n";
  }
  tpc.close();

  if (add_tpc_boxes)
  {
    if (max_driftlength <= 0.0)
    {
      std::cout << "ERROR: no maximum drift length (layer 20 missing?): TPC boxes not added" << std::endl;
      return;
    }
    const int n_boxes = add_tpc_measurement_boxes(manager, radius, thickness, max_driftlength - 0.0001, n_modules, n_phi);
    if (n_boxes < 0)
    {
      return;
    }
    manager->CloseGeometry();
    std::cout << "added " << n_boxes << " TPC measurement boxes (" << radius.size() << " layers x " << n_modules
              << " modules x " << n_phi << " phi segments, z half-length " << 0.5 * (max_driftlength - 0.0001)
              << " cm)" << std::endl;
  }

  const TString geometry_path = TString(prefix) + "_tgeo.root";
  manager->Export(geometry_path);

  const TString volumes_path = TString(prefix) + "_volumes.txt";
  std::ofstream volumes(volumes_path.Data());
  write_volumes(volumes, manager->GetTopNode(), 0, 4);
  volumes.close();

  std::cout << "wrote " << geometry_path << ", " << tpc_path << ", " << volumes_path << std::endl;
  std::cout << "top volume " << manager->GetTopVolume()->GetName() << " with " << manager->GetTopNode()->GetNdaughters()
            << " daughters; " << manager->GetListOfVolumes()->GetEntries() << " volumes in total; maximum drift length "
            << max_driftlength << " cm" << std::endl;
}
