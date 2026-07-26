/*---------------------------------------------------------------------------*\
    SpalartAllmarasAI -- see header for model description and provenance.
    Structure follows OpenFOAM v2412 SpalartAllmarasBase.C (GPL v3).
\*---------------------------------------------------------------------------*/

#include "SpalartAllmarasAI.H"
#include "fvcCurl.H"
#include "fvcLaplacian.H"
#include "wallDist.H"
#include "bound.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasAI<BasicTurbulenceModel>::chi() const
{
    return nuTilda_/this->nu();
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasAI<BasicTurbulenceModel>::fv1
(
    const volScalarField& chi
) const
{
    const volScalarField chi3("chi3", pow3(chi));
    return chi3/(chi3 + pow3(Cv1_));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasAI<BasicTurbulenceModel>::fv2
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{
    return scalar(1) - chi/(scalar(1) + chi*fv1);
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasAI<BasicTurbulenceModel>::Stilda
(
    const volScalarField& chi,
    const volScalarField& fv1,
    const volTensorField& gradU
) const
{
    const volScalarField Omega(::sqrt(2.0)*mag(skew(gradU)));

    return
        max
        (
            Omega + fv2(chi, fv1)*nuTilda_/sqr(kappa_*y_),
            Cs_*Omega
        );
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> SpalartAllmarasAI<BasicTurbulenceModel>::fw
(
    const volScalarField& Stilda
) const
{
    const dimensionedScalar eps(Stilda.dimensions(), SMALL);

    const volScalarField::Internal r
    (
        min(nuTilda_()/(max(Stilda(), eps)*sqr(kappa_*y_())), scalar(10))
    );

    const volScalarField::Internal g(r + Cw2_*(pow6(r) - r));

    return g*pow((1 + pow6(Cw3_))/(pow6(g) + pow6(Cw3_)), 1.0/6.0);
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasAI<BasicTurbulenceModel>::sigmaT
(
    const volScalarField& chi
) const
{
    // sigma_t = 1 - exp(-max(chi - switchCenter, 0)/switchWidth); exactly 0
    // for chi <= switchCenter (SAAiTransition.h __aiIsTurb)
    return scalar(1) - exp(-max(chi - switchCenter_, dimensionedScalar(dimless, Zero))/switchWidth_);
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasAI<BasicTurbulenceModel>::aiRate
(
    const volScalarField& omegaMag
)
{
    const volVectorField& U = this->U_;

    // Indicator triple (magnitude/standard form, SAAiTransition.h __aiRate):
    //   X = |u|, Y = |omega| d, Z = + 1/2 d^2 u'',  u'' = lap(U) . uhat
    // (Flow360 forms Z0 = -1/2 d^2 dwdn with dwdn = -u''; same sign here.)
    const volScalarField X(mag(U));
    const dimensionedScalar epsU(dimVelocity, SMALL);
    const volScalarField upp((fvc::laplacian(U) & U)/max(X, epsU));

    const volScalarField Y(omegaMag*y_);
    const volScalarField Z(0.5*sqr(y_)*upp);

    const volScalarField xy(sqrt(sqr(X) + sqr(Y)) + epsU);
    const volScalarField R(sqrt(sqr(X) + sqr(Y) + sqr(Z)) + epsU);

    const volScalarField ShatFrac(Y/xy);        // shear fraction (Omega-hat)
    const volScalarField g((Y - X - Z)/R);      // inflection coordinate (I-hat)
    const volScalarField P(ShatFrac*g);         // amplifying coordinate

    const volScalarField Pp(min(max(P, scalar(0)), scalar(1)));

    // Soft-min (n=2) onset threshold on the same P
    const volScalarField Pf(max(Pp, scalar(1e-6)));
    const volScalarField pw(reOmA_ + reOmB_/sqr(Pf));
    const volScalarField reOmC(reOmCeil_*pw/sqrt(sqr(reOmCeil_) + sqr(pw)));

    const volScalarField reOmega(sqr(y_)*omegaMag/this->nu());

    const volScalarField onset
    (
        0.5*(scalar(1) + tanh((reOmega/reOmC - scalar(1))/rampWidth_))
    );

    aiP_ = P;
    aiZ_ = Z;
    aiReOmega_ = reOmega;
    aiRate_ = aMax_*Pp*onset;

    return tmp<volScalarField>::New("aiRate", aiRate_);
}


template<class BasicTurbulenceModel>
void SpalartAllmarasAI<BasicTurbulenceModel>::correctNut
(
    const volScalarField& fv1
)
{
    this->nut_ = nuTilda_*fv1;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void SpalartAllmarasAI<BasicTurbulenceModel>::correctNut()
{
    correctNut(fv1(this->chi()));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SpalartAllmarasAI<BasicTurbulenceModel>::SpalartAllmarasAI
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    sigmaNut_
    (
        dimensioned<scalar>::getOrAddToDict("sigmaNut", this->coeffDict_, 0.66666)
    ),
    kappa_
    (
        dimensioned<scalar>::getOrAddToDict("kappa", this->coeffDict_, 0.41)
    ),
    Cb1_
    (
        dimensioned<scalar>::getOrAddToDict("Cb1", this->coeffDict_, 0.1355)
    ),
    Cb2_
    (
        dimensioned<scalar>::getOrAddToDict("Cb2", this->coeffDict_, 0.622)
    ),
    Cw1_(Cb1_/sqr(kappa_) + (1.0 + Cb2_)/sigmaNut_),
    Cw2_
    (
        dimensioned<scalar>::getOrAddToDict("Cw2", this->coeffDict_, 0.3)
    ),
    Cw3_
    (
        dimensioned<scalar>::getOrAddToDict("Cw3", this->coeffDict_, 2.0)
    ),
    Cv1_
    (
        dimensioned<scalar>::getOrAddToDict("Cv1", this->coeffDict_, 7.1)
    ),
    Cs_
    (
        dimensioned<scalar>::getOrAddToDict("Cs", this->coeffDict_, 0.3)
    ),

    aMax_
    (
        dimensioned<scalar>::getOrAddToDict("aMax", this->coeffDict_, 0.19)
    ),
    reOmCeil_
    (
        dimensioned<scalar>::getOrAddToDict("reOmCeil", this->coeffDict_, 1851.2)
    ),
    reOmA_
    (
        dimensioned<scalar>::getOrAddToDict("reOmA", this->coeffDict_, 124.6)
    ),
    reOmB_
    (
        dimensioned<scalar>::getOrAddToDict("reOmB", this->coeffDict_, 1.424)
    ),
    rampWidth_
    (
        dimensioned<scalar>::getOrAddToDict("rampWidth", this->coeffDict_, 0.35)
    ),
    switchCenter_
    (
        dimensioned<scalar>::getOrAddToDict("switchCenter", this->coeffDict_, 1.0)
    ),
    switchWidth_
    (
        dimensioned<scalar>::getOrAddToDict("switchWidth", this->coeffDict_, 4.0)
    ),
    nuLamScale_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "nuLamScale",
            this->coeffDict_,
            1.0/6.0
        )
    ),

    nuTilda_
    (
        IOobject
        (
            "nuTilda",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    aiRate_
    (
        IOobject("aiRate", this->runTime_.timeName(), this->mesh_,
                 IOobject::NO_READ, IOobject::AUTO_WRITE),
        this->mesh_,
        dimensionedScalar(dimless, Zero)
    ),
    aiP_
    (
        IOobject("aiP", this->runTime_.timeName(), this->mesh_,
                 IOobject::NO_READ, IOobject::AUTO_WRITE),
        this->mesh_,
        dimensionedScalar(dimless, Zero)
    ),
    aiZ_
    (
        IOobject("aiZ", this->runTime_.timeName(), this->mesh_,
                 IOobject::NO_READ, IOobject::AUTO_WRITE),
        this->mesh_,
        dimensionedScalar(dimVelocity, Zero)
    ),
    aiReOmega_
    (
        IOobject("aiReOmega", this->runTime_.timeName(), this->mesh_,
                 IOobject::NO_READ, IOobject::AUTO_WRITE),
        this->mesh_,
        dimensionedScalar(dimless, Zero)
    ),

    y_(wallDist::New(this->mesh_).y())
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool SpalartAllmarasAI<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        sigmaNut_.readIfPresent(this->coeffDict());
        kappa_.readIfPresent(this->coeffDict());
        Cb1_.readIfPresent(this->coeffDict());
        Cb2_.readIfPresent(this->coeffDict());
        Cw1_ = Cb1_/sqr(kappa_) + (1.0 + Cb2_)/sigmaNut_;
        Cw2_.readIfPresent(this->coeffDict());
        Cw3_.readIfPresent(this->coeffDict());
        Cv1_.readIfPresent(this->coeffDict());
        Cs_.readIfPresent(this->coeffDict());

        aMax_.readIfPresent(this->coeffDict());
        reOmCeil_.readIfPresent(this->coeffDict());
        reOmA_.readIfPresent(this->coeffDict());
        reOmB_.readIfPresent(this->coeffDict());
        rampWidth_.readIfPresent(this->coeffDict());
        switchCenter_.readIfPresent(this->coeffDict());
        switchWidth_.readIfPresent(this->coeffDict());
        nuLamScale_.readIfPresent(this->coeffDict());

        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
tmp<volScalarField>
SpalartAllmarasAI<BasicTurbulenceModel>::DnuTildaEff() const
{
    // SA-AI modification 3: laminar viscosity scaled by nuLamScale in the
    // nuTilda diffusion only (c_nu,ai = 1/6); momentum keeps the full nu.
    return tmp<volScalarField>::New
    (
        IOobject::groupName("DnuTildaEff", this->alphaRhoPhi_.group()),
        (nuTilda_ + nuLamScale_*this->nu())/sigmaNut_
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasAI<BasicTurbulenceModel>::k() const
{
    const scalar Cmu = 0.09;
    const auto fv1 = this->fv1(chi());

    return tmp<volScalarField>::New
    (
        IOobject::groupName("k", this->alphaRhoPhi_.group()),
        cbrt(fv1)*nuTilda_*::sqrt(scalar(2)/Cmu)*mag(symm(fvc::grad(this->U_)))
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField>
SpalartAllmarasAI<BasicTurbulenceModel>::epsilon() const
{
    const scalar Cmu = 0.09;
    const auto fv1 = this->fv1(chi());
    const dimensionedScalar nutSMALL(nuTilda_.dimensions(), SMALL);

    return tmp<volScalarField>::New
    (
        IOobject::groupName("epsilon", this->alphaRhoPhi_.group()),
        sqrt(fv1)*sqr(::sqrt(Cmu)*this->k())/(nuTilda_ + this->nut_ + nutSMALL)
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasAI<BasicTurbulenceModel>::omega() const
{
    const scalar betaStar = 0.09;
    const dimensionedScalar k0(sqr(dimLength/dimTime), SMALL);

    return tmp<volScalarField>::New
    (
        IOobject::groupName("omega", this->alphaRhoPhi_.group()),
        this->epsilon()/(betaStar*(this->k() + k0))
    );
}


template<class BasicTurbulenceModel>
void SpalartAllmarasAI<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    {
        // Local references
        const alphaField& alpha = this->alpha_;
        const rhoField& rho = this->rho_;
        const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
        fv::options& fvOptions(fv::options::New(this->mesh_));

        eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

        const volScalarField chi(this->chi());
        const volScalarField fv1(this->fv1(chi));

        tmp<volTensorField> tgradU = fvc::grad(this->U_);
        const volScalarField omegaMag(::sqrt(2.0)*mag(skew(tgradU())));
        const volScalarField Stilda(this->Stilda(chi, fv1, tgradU()));
        tgradU.clear();

        // --- SA-AI pieces (all frozen w.r.t. nuTilda except sigma_t) ---
        const volScalarField rate(this->aiRate(omegaMag));
        const volScalarField sigmaT(this->sigmaT(chi));

        // Production blend (gated max), both branches linear in nuTilda:
        //   P = max[(1 - sigma_t) a |omega|, sigma_t Cb1 Stilda] * nuTilda
        const volScalarField prodCoeff
        (
            "prodCoeff",
            max
            (
                (scalar(1) - sigmaT)*rate*omegaMag,
                sigmaT*Cb1_*Stilda
            )
        );

        // Destruction gate: sigma_D = 1 - R (1 - sigma_t), R = Cb1/(kappa^2 Cw1)
        const dimensionedScalar Rtie(Cb1_/(sqr(kappa_)*Cw1_));
        const volScalarField sigmaD
        (
            scalar(1) - Rtie*(scalar(1) - sigmaT)
        );

        tmp<fvScalarMatrix> nuTildaEqn
        (
            fvm::ddt(alpha, rho, nuTilda_)
          + fvm::div(alphaRhoPhi, nuTilda_)
          - fvm::laplacian(alpha*rho*DnuTildaEff(), nuTilda_)
          - Cb2_/sigmaNut_*alpha()*rho()*magSqr(fvc::grad(nuTilda_)()())
         ==
            prodCoeff()*alpha()*rho()*nuTilda_()
          - fvm::Sp
            (
                sigmaD()*Cw1_*fw(Stilda)*alpha()*rho()*nuTilda_()/sqr(y_()),
                nuTilda_
            )
          + fvOptions(alpha, rho, nuTilda_)
        );

        nuTildaEqn.ref().relax();
        fvOptions.constrain(nuTildaEqn.ref());
        solve(nuTildaEqn);
        fvOptions.correct(nuTilda_);
        bound(nuTilda_, dimensionedScalar(nuTilda_.dimensions(), Zero));
        nuTilda_.correctBoundaryConditions();
    }

    correctNut();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
