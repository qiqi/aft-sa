/*---------------------------------------------------------------------------*\
    Register SpalartAllmarasAI as an incompressible RAS model
    (runtime-loadable via libs ("libSAAIIncompressibleTurbulenceModel.so")).
\*---------------------------------------------------------------------------*/

#include "turbulentTransportModels.H"
#include "SpalartAllmarasAI.H"

makeRASModel(SpalartAllmarasAI);

// ************************************************************************* //
