"""
acts_vertexing.py: ACTS's own vertex finding, as shipped (AdaptiveMultiVertexFinderAlgorithm, Gaussian track-density
seeder), on per-event perigee tracks; a reference for VertexHead, run by compare_vertexing.py inside the ACTS image:

    singularity exec acts-base-48.2.0.sif acts-python scripts/acts/acts_vertexing.py <dir> --beamspot <x y z mm> \
        (--bz_tesla 1.4 | --field_map <field.txt>)

Reads <dir>/event<N:09>-tracks.csv (CsvTrackParameterReader: perigee at --beamspot, mm) and writes the vertices
found to <dir>/event<N:09>-vertices.csv (CsvVertexWriter: x, y, z, T in mm, in the order ACTS found them).
"""
import argparse
import os
import sys

import acts
import acts.examples

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acts_worker import build_field  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--bz_tesla", type=float, default=None, help="uniform field, T (without --field_map)")
    parser.add_argument("--field_map", default=None, help="ACTS text field map (mm, T), as the worker loads it")
    parser.add_argument("--field_map_kind", default="xyz", choices=("xyz", "rz"))
    parser.add_argument("--beamspot", type=float, nargs=3, required=True, help="the tracks' perigee reference, mm")
    args = parser.parse_args()
    if args.bz_tesla is None and args.field_map is None:
        parser.error("--bz_tesla or --field_map is needed")

    field = build_field(args.bz_tesla, args.field_map, args.field_map_kind)
    level = acts.logging.WARNING
    sequencer = acts.examples.Sequencer(numThreads=1, logLevel=level, trackFpes=False)   # no FPE-mask summary
    sequencer.addReader(acts.examples.CsvTrackParameterReader(
        level=level, inputDir=args.directory, inputStem="tracks", outputTrackParameters="tracks",
        beamspot=args.beamspot))
    sequencer.addAlgorithm(acts.examples.AdaptiveMultiVertexFinderAlgorithm(
        level=level, inputTrackParameters="tracks", outputProtoVertices="protovertices",
        outputVertices="vertices", seedFinder=acts.examples.VertexSeedFinder.GaussianSeeder, bField=field))
    sequencer.addWriter(acts.examples.CsvVertexWriter(
        level=level, inputVertices="vertices", outputDir=args.directory, outputStem="vertices"))
    sequencer.run()


if __name__ == "__main__":
    main()
