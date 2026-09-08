from starPositionOnCCD                             import StarPositionOnCCD            
from stellarVariability                            import StellarVariability           
from StellarAberration.absoluteAberration          import AbsoluteAberration           
#from StellarAberration.differentialAberration      import DifferentialAberration       
from fieldDistortion                               import FieldDistortion              
from ThermoElasticDrift.tedFromFile                import TedFromFile                  
from ThermoElasticDrift.tedYawPitchRoll            import TedYawPitchRoll              
from ThermoElasticDrift.tedOnCCDs                  import TedOnCCDs                    
from Jitter.jitterYawPitchRoll                     import JitterYawPitchRoll           
from Jitter.jitterFromFile                         import JitterFromFile               
from Jitter.jitterOnCCDs                           import JitterOnCCDs                 
from Jitter.jitterOnCameras                        import JitterOnCameras              
from skyBackground                                 import SkyBackGround                
from Convolution.mappedGaussianPSF                 import MappedGaussianPSF            
from Convolution.analyticNonGaussian               import AnalyticNonGaussianPSF       
from Convolution.analyticGaussian                  import AnalyticGaussianPSF          
from PRNU.prnu                                     import PRNU
from PRNU.mappedPRNU                               import MappedPRNU
from rebinning                                     import Rebinning                    
from ThroughputEfficiency.transmissionEfficiency   import TransmissionEfficiency       
from ThroughputEfficiency.vignetting               import Vignetting                   
from ThroughputEfficiency.polarization             import Polarization                 
from ThroughputEfficiency.quantumEfficiency        import QuantumEfficiency            
from ThroughputEfficiency.particulateContamination import ParticulateContamination     
from ThroughputEfficiency.molecularContamination   import MolecularContamination       
from DarkSignal.shotNoise                          import ShotNoise                    
from DarkSignal.darkSignalNonUniformity            import DarkSignalNonUniformity      
from DarkSignal.tempVariationOfCCD                 import TempVariationOfCCD           
from brighterFatterEffect                          import BrighterFatterEffect         
#from cosmics                                       import Cosmics                      
from openShutterSmearing                           import OpenShutterSmearing          
from ChargeTransferInefficiency.simpleCTI          import SimpleCTI                    
from ChargeTransferInefficiency.Short2013          import Short2013CTI                 
from ChargeTransferInefficiency.Short2013FromFile  import Short2013CTIFromFile         
from photonNoise                                   import PhotonNoise                  
from readOutNoise                                  import ReadoutNoise                 
from fullWellSaturation                            import FullWellSaturation           
from Quantisation.gain                             import Gain                         
from Quantisation.electronicOffset                 import ElectronicOffset             
from Quantisation.flooring                         import Flooring                     
from Quantisation.digitalSaturation                import DigitalSaturation            
from metallicShield                                import MetallicShield               
from quaternion                                    import Quaternion                   
from RefFrames.focalPlaneCoordinates               import FocalPlaneCoordinates
from nonlineargain                                 import NonlinearGainTest 
from spr                                           import SPRTest
from straylight                                    import Straylight


from contextlib import contextmanager
import sys, os
import time


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


testMessages = []

try:
    from termcolor import colored
    successMessage = colored('Success', 'green')
    failureMessage = colored('Failure', 'red', attrs=['reverse', 'blink'])
except ImportError:
    successMessage = 'Success'
    failureMessage = 'Failure'

success = {True: successMessage, False: failureMessage}





myTests = [
    (StarPositionOnCCD(),          "Star position on CCD"),
    (SkyBackGround(),              "Sky Background"),
    (FullWellSaturation(),         "Full-Well Saturation"),
    (JitterOnCCDs(),               "Jitter on different CCDs"),
    (FieldDistortion(),            "Field Distortion"),
    (StellarVariability(),         "Stellar Variability"),
    (TedOnCCDs(),                  "TED on different CCDs"),
    (SimpleCTI(),                  "Simple CTI model"),
    (AnalyticGaussianPSF(),        "Analytic Gaussian PSF"),
    (Short2013CTI(),               "Short 2013 model"),
    (SPRTest(),                    "Stellar pollution ratio (SPR)"),    
    (JitterOnCameras(),            "Jitter on different cameras"),
    (ReadoutNoise(),               "Readout Noise"),
    (JitterFromFile(),             "Jitter from file"),
    (TedFromFile(),                "Thermo-Elastic drift from file"),
    (ParticulateContamination(),   "Particulate Contamination"),
    (MolecularContamination(),     "Molecular Contamination"),
    (Quaternion(),                 "Quaternions"),
    (QuantumEfficiency(),          "Quantum Efficiency"),
    (FocalPlaneCoordinates(),      "Orientation CAM ref frame in (RA, dec) plane"),
    (Straylight(),                 "Straylight"),
    (PRNU(),                       "Pixel-Responsivity Non Uniformity"),
    (MappedPRNU(),                 "Mapped PRNU"),
#    (Short2013CTIFromFile(),       "Short 2013 from file model"),
    (AnalyticNonGaussianPSF(),     "Analytic non Gaussian PSF"),
    (ElectronicOffset(),           "Electronic Offset"),
    (Rebinning(),                  "Rebinning Subpixel"),
    (Vignetting(),                 "Vignetting"),
    (Polarization(),               "Polarization"),
    (ShotNoise(),                  "Shot Noise"),
    (OpenShutterSmearing(),        "Open-Shutter Smearing"),
    (TedYawPitchRoll(),            "Thermo-Elastic drift from noise"),
    (JitterYawPitchRoll(),         "Jitter from red noise"),
    (NonlinearGainTest(),          "Nonlinear gain"),
    (Flooring(),                   "Flooring"),
    (DigitalSaturation(),          "Digital Saturation"),
    (TransmissionEfficiency(),     "Transmission Efficiency"),
    (MappedGaussianPSF(),          "Mapped Gaussian PSF and Charge Diffusion"),
    #(Cosmics(),                    "Cosmics"),
    (BrighterFatterEffect(),       "Brighter-Fatter Effect"),
    (AbsoluteAberration(),         "Absolute Aberration"),
    (MetallicShield(),             "Metallic Shield"),
    (PhotonNoise(),                "Photon Noise"),
    # (DifferentialAberration(),     "Differential Aberration"),
    (Gain(),                       "Gain"),
    (DarkSignalNonUniformity(),    "Dark Signal Non Uniformity"),
    (TempVariationOfCCD(),         "Temperature Variation of CCD"),
]                                  



def runTheTest(description, test, i):
    t0 = time.time()
    with suppress_stdout():
        t  = test.run()

    testMessages.append("{:<9}  {:^55}:{}\ttime: {:.1f}s".format("Test{}:".format(i), description, success[t], time.time()-t0))
    print(testMessages[-1])





i = 1
startTime = time.time()
for (function, description) in myTests:
    runTheTest(description, function, i)
    i += 1



# That's it: write the output on screen also to a file

endTime = time.time()
print("[Completed in {0} s]".format(endTime - startTime))
with open(os.environ["PLATO_PROJECT_HOME"] + '/tests/validationTests/ioFiles/result.txt', 'w') as outputFile:
    for line in testMessages:
        outputFile.write(line+'\n')

