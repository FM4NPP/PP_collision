"""
acts_vertexing.py: ACTS's own vertex finding, as shipped (AdaptiveMultiVertexFinderAlgorithm, Gaussian track-density
seeder), on per-event perigee tracks; a reference for VertexHead, run by compare_vertexing.py inside the ACTS image:

    singularity exec acts-base-48.2.0.sif acts-python scripts/acts/acts_vertexing.py <dir> --bz_tesla 1.4

Reads <dir>/event<N:09>-tracks.csv (CsvTrackParameterReader: perigee at the origin, mm) and writes the vertices
found to <dir>/event<N:09>-vertices.csv (CsvVertexWriter: x, y, z, T in mm, in the order ACTS found them).
"""
import argparse

import acts
import acts.examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--bz_tesla", type=float, required=True)
    args = parser.parse_args()

    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, args.bz_tesla * acts.UnitConstants.T))
    level = acts.logging.WARNING
    sequencer = acts.examples.Sequencer(numThreads=1, logLevel=level)
    sequencer.addReader(acts.examples.CsvTrackParameterReader(
        level=level, inputDir=args.directory, inputStem="tracks", outputTrackParameters="tracks",
        beamspot=[0.0, 0.0, 0.0]))
    sequencer.addAlgorithm(acts.examples.AdaptiveMultiVertexFinderAlgorithm(
        level=level, inputTrackParameters="tracks", outputProtoVertices="protovertices",
        outputVertices="vertices", seedFinder=acts.examples.VertexSeedFinder.GaussianSeeder, bField=field))
    sequencer.addWriter(acts.examples.CsvVertexWriter(
        level=level, inputVertices="vertices", outputDir=args.directory, outputStem="vertices"))
    sequencer.run()


if __name__ == "__main__":
    main()
