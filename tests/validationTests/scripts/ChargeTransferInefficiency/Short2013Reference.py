"""
Tests of the start state of the Short et al. (2013) CTI model in a subfield: the background
and the stars below the subfield.
"""

import numpy as np

from test import Test


class Short2013Base(Test):
    """
    Common configuration: a noise-free image with a constant trap density.
    """

    def setAllEffects(self):
        """
        Switch on the Short et al. (2013) model with the EOL trap densities at all times.
        """
        super().setAllEffects()
        self.sim["CCD/Position"] = 1
        self.sim["ObservingParameters/NumExposures"] = 1
        self.sim["CCD/IncludeQuantisation"] = "no"
        self.sim["CCD/IncludeDigitalSaturation"] = "no"
        self.sim["CCD/BadPixelMap/includeBadPixelMap"] = "no"
        self.sim["CCD/CTI/Model"] = "Short2013"
        self.sim["CCD/CTI/Short2013/TrapDensity/BOL"] = self.sim["CCD/CTI/Short2013/TrapDensity/EOL"]

    def runMaps(self, includeCTI):
        """
        Run the simulation and return the image and smearing map.
        """
        self.sim["CCD/IncludeCTIeffects"] = "yes" if includeCTI else "no"
        output = self.sim.run(removeOutputFile=True)
        return np.array(output.getImage(0), dtype=float), np.array(output.getSmearingMap(0), dtype=float)


class Short2013Subfield(Short2013Base):
    """
    A small subfield must equal the same rows of a subfield that extends far below it.
    """

    def setNr(self):
        self.nr = "016.4"

    def setAllEffects(self):
        super().setAllEffects()
        self.sim["Sky/SkyBackground/BackgroundValue"] = 100
        self.sim["SubField/ZeroPointColumn"] = 2000
        self.sim["SubField/NumColumns"] = 8
        self.sim["SubField/NumSmearingOverscanRows"] = 2
        self.firstRow = 3000
        self.numRows = 40
        self.extraRows = 700

    def runSimulation(self):
        catalog = self.outputDir + "/stars.txt"
        self.sim.createStarCatalogFileFromPixelCoordinates(np.array([self.firstRow + 20.3]), np.array([2004.2]),
                                                           np.array([11.0]), np.array([1]), catalog)
        self.sim["CCD/IncludeCTIeffects"] = "yes"
        self.sim["SubField/ZeroPointRow"] = self.firstRow
        self.sim["SubField/NumRows"] = self.numRows
        self.small, _ = self.runMaps(True)

        self.sim["SubField/ZeroPointRow"] = self.firstRow - self.extraRows
        self.sim["SubField/NumRows"] = self.numRows + self.extraRows
        large, _ = self.runMaps(True)
        self.large = large[self.extraRows:]

    def compare(self):
        return np.abs(self.small - self.large).max() < 0.5


class Short2013ColumnStars(Short2013Base):
    """
    A star below the subfield, in the same columns, must leave the same traps behind as in a
    subfield that contains the star.
    """

    def setNr(self):
        self.nr = "016.5"

    def setAllEffects(self):
        super().setAllEffects()
        self.sim["Sky/SkyBackground/BackgroundValue"] = 100
        self.sim["CCD/IncludeQuantumEfficiency"] = "yes"
        self.sim["SubField/ZeroPointColumn"] = 2000
        self.sim["SubField/NumColumns"] = 8
        self.sim["SubField/NumSmearingOverscanRows"] = 2
        self.sim["SubField/ColumnStarsMagnitudeLimit"] = 16
        self.firstRow = 3000
        self.numRows = 40
        self.extraRows = 700

    def runSimulation(self):
        catalog = self.outputDir + "/stars.txt"
        self.sim.createStarCatalogFileFromPixelCoordinates(np.array([self.firstRow - 30.3]), np.array([2004.2]),
                                                           np.array([10.0]), np.array([1]), catalog)

        # Small subfield above the star, with and without the stars along the columns

        self.sim["SubField/ZeroPointRow"] = self.firstRow
        self.sim["SubField/NumRows"] = self.numRows
        self.withColumnStars, _ = self.runMaps(True)

        self.sim["SubField/ColumnStarsMagnitudeLimit"] = -100
        self.withoutColumnStars, _ = self.runMaps(True)
        self.sim["SubField/ColumnStarsMagnitudeLimit"] = 16

        # Large subfield that contains the star. It has the same centre as the small one, because the
        # mapped PSF model uses the PSF of the centre of the subfield.

        self.sim["SubField/ZeroPointRow"] = self.firstRow - self.extraRows
        self.sim["SubField/NumRows"] = self.numRows + 2 * self.extraRows
        large, _ = self.runMaps(True)
        self.large = large[self.extraRows:self.extraRows + self.numRows]

    def compare(self):
        matches = np.abs(self.withColumnStars - self.large).max() < 0.1
        starMatters = np.abs(self.withoutColumnStars - self.large).max() > 10
        return matches and starMatters


if __name__ == "__main__":
    for test in (Short2013Subfield, Short2013ColumnStars):
        print(test.__name__, test().run())
