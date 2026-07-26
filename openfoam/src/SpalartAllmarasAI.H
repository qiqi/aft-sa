/*---------------------------------------------------------------------------*\
    SpalartAllmarasAI -- shallow OpenFOAM port of the single-equation SA-AI
    transition model (Spalart-Allmaras with Autogenous Inception, sphere
    kernel / model v3).

    Reference implementation: Flow360 monorepo
    compute/src/Flow360Core/Applications/Solver/SpalartAllmaras/
    (SAAiTransition.h, ModelConstants.h, SpalartAllmaras.h), and the SA-AI
    paper (sa-ai/paper/sa-ai.tex).

    The SA working variable nuTilda, in its sub-O(1) chi range, doubles as an
    e^N-style amplification factor. Modifications to standard SA:

      1. Production blend (gated max):
             P = max[(1 - sigma_t) P_AI, sigma_t P_SA]
         P_SA = Cb1 Stilda nuTilda,  P_AI = a |omega| nuTilda, with the
         sphere-kernel rate a built from the local indicator triple
             X = |u|, Y = |omega| d, Z = 1/2 d^2 (lap U . uhat)
         shear fraction Shat = Y/sqrt(X^2+Y^2), inflection coordinate
         g = (Y-X-Z)/R, P = Shat*g, a = aMax clip<P>_0^1 * onset with the
         soft-min onset threshold ReOmega_c = softmin2(Ceil, A + B/P^2) and
         onset = (1 + tanh((ReOmega/ReOmega_c - 1)/rampWidth))/2,
         ReOmega = d^2 |omega| / nu.
      2. Destruction gate (sigma-d tie):
             sigma_D = 1 - R (1 - sigma_t),  R = Cb1/(kappa^2 Cw1)
         multiplies the standard SA destruction.
      3. Laminar diffusivity scaling: the nuTilda diffusion sees
         nuLamScale * nu (default 1/6) instead of nu; momentum untouched.

    sigma_t(chi) = 1 - exp(-max(chi - 1, 0)/switchWidth), switchWidth = 4.

    Freestream seed: nuTilda_inf = chi_inf * nu with
    chi_inf = Cv1 exp(-N_crit) (e.g. 8.76e-4 at N_crit = 9).

    "Shallow": everything is assembled from existing fvm/fvc operators; no
    custom numerics. The amplification rate is independent of nuTilda (frozen
    flow), entering as an explicit source exactly like stock SA production.

    Not ported (convergence aids / experimental switches, not model content):
    AI_LAMINAR_SLOWDOWN (pseudo-time scaling; SIMPLE relaxation plays this
    role), AI_FV1BYPASS (experimental, default off), AI_INVARIANT_KERNEL
    (experimental, default off), DES length scales, rotation-curvature
    correction, low-Re Cw2 correction.
\*---------------------------------------------------------------------------*/

#ifndef SpalartAllmarasAI_H
#define SpalartAllmarasAI_H

#include "RASModel.H"
#include "eddyViscosity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                      Class SpalartAllmarasAI Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class SpalartAllmarasAI
:
    public eddyViscosity<RASModel<BasicTurbulenceModel>>
{
    // Private Member Functions

        //- No copy construct
        SpalartAllmarasAI(const SpalartAllmarasAI&) = delete;

        //- No copy assignment
        void operator=(const SpalartAllmarasAI&) = delete;


protected:

    // Protected Data

        // Standard SA coefficients

            dimensionedScalar sigmaNut_;
            dimensionedScalar kappa_;
            dimensionedScalar Cb1_;
            dimensionedScalar Cb2_;
            dimensionedScalar Cw1_;
            dimensionedScalar Cw2_;
            dimensionedScalar Cw3_;
            dimensionedScalar Cv1_;
            dimensionedScalar Cs_;

        // SA-AI coefficients (defaults = paper calibration; see
        // ModelConstants.h in the Flow360 reference implementation)

            //- a_max, Michalke free-shear ceiling (AI_RATESCALE)
            dimensionedScalar aMax_;

            //- Onset threshold ceiling k*2600 (AI_REOMC_CEIL)
            dimensionedScalar reOmCeil_;

            //- Onset threshold additive floor k*175 (AI_REOMC_A)
            dimensionedScalar reOmA_;

            //- Onset threshold inverse-square coeff k*2 (AI_REOMC_B)
            dimensionedScalar reOmB_;

            //- Onset tanh ramp half-width (AI_RAMPWIDTH)
            dimensionedScalar rampWidth_;

            //- chi at laminar->turbulent handover
            dimensionedScalar switchCenter_;

            //- Handover width in chi units (AI_SWITCHWIDTH)
            dimensionedScalar switchWidth_;

            //- Laminar-viscosity scale in nuTilda diffusion (AI_NULAMSCALE)
            dimensionedScalar nuLamScale_;


        // Fields

            //- Modified kinematic viscosity [m^2/s]
            volScalarField nuTilda_;

            //- Kernel diagnostics (written with the solution for
            //- cross-code comparison against Flow360's derived slice)
            volScalarField aiRate_;
            volScalarField aiP_;
            volScalarField aiZ_;
            volScalarField aiReOmega_;

            //- Wall distance
            //  Note: different to wall distance in parent RASModel
            //  which is for near-wall cells only
            const volScalarField& y_;


    // Protected Member Functions

        tmp<volScalarField> chi() const;

        tmp<volScalarField> fv1(const volScalarField& chi) const;

        tmp<volScalarField> fv2
        (
            const volScalarField& chi,
            const volScalarField& fv1
        ) const;

        tmp<volScalarField> Stilda
        (
            const volScalarField& chi,
            const volScalarField& fv1,
            const volTensorField& gradU
        ) const;

        tmp<volScalarField::Internal> fw(const volScalarField& Stilda) const;

        //- Laminar->turbulent handover weight sigma_t(chi)
        tmp<volScalarField> sigmaT(const volScalarField& chi) const;

        //- Sphere-kernel amplification rate a(x) (dimensionless; frozen
        //- w.r.t. nuTilda). Non-const: refreshes the kernel diagnostic
        //- fields.
        tmp<volScalarField> aiRate(const volScalarField& omegaMag);

        void correctNut(const volScalarField& fv1);
        virtual void correctNut();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("SpalartAllmarasAI");


    // Constructors

        //- Construct from components
        SpalartAllmarasAI
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~SpalartAllmarasAI() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read();

        //- Return the effective diffusivity for nuTilda
        //  (laminar part scaled by nuLamScale -- SA-AI modification 3)
        tmp<volScalarField> DnuTildaEff() const;

        //- Return the turbulence kinetic energy (estimate)
        virtual tmp<volScalarField> k() const;

        //- Return the turbulence kinetic energy dissipation rate (estimate)
        virtual tmp<volScalarField> epsilon() const;

        //- Return the turbulence specific dissipation rate (estimate)
        virtual tmp<volScalarField> omega() const;

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "SpalartAllmarasAI.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
